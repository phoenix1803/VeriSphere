"""
Advanced rate limiting implementation
"""
import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
import hashlib
import json

from defame.utils.logger import get_logger
from config.globals import get_config

logger = get_logger(__name__)
config = get_config()


class RateLimitType(str, Enum):
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    GLOBAL = "global"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int
    window_seconds: int
    burst_allowance: int = 0
    
    @property
    def key_ttl(self) -> int:
        """TTL for rate limit keys"""
        return self.window_seconds + 60  # Extra buffer


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache: Dict[str, Dict] = {}
        self.cache_cleanup_interval = 300  # 5 minutes
        
        # Rate limit configurations
        self.limits = {
            # API endpoints
            "api_general": RateLimit(100, 60, 20),  # 100/min with 20 burst
            "api_submit_claim": RateLimit(10, 60, 5),  # 10/min with 5 burst
            "api_batch": RateLimit(5, 3600, 2),  # 5/hour with 2 burst
            
            # Authentication
            "auth_login": RateLimit(5, 300, 0),  # 5 per 5 minutes, no burst
            "auth_register": RateLimit(3, 3600, 0),  # 3 per hour, no burst
            "auth_password_reset": RateLimit(3, 3600, 0),
            
            # Global limits
            "global_api": RateLimit(10000, 60, 1000),  # 10k/min globally
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Rate limiter Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache: {e}")
            self.redis_client = None
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: str,
        rate_limit_type: RateLimitType = RateLimitType.PER_IP
    ) -> RateLimitResult:
        """Check if request is within rate limits"""
        try:
            limit_config = self.limits.get(limit_type)
            if not limit_config:
                # No limit configured, allow request
                return RateLimitResult(True, 999999, int(time.time()) + 3600)
            
            # Create unique key
            key = self._create_key(identifier, limit_type, rate_limit_type)
            
            if self.redis_client:
                return await self._check_redis_rate_limit(key, limit_config)
            else:
                return await self._check_local_rate_limit(key, limit_config)
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # On error, allow request but log
            return RateLimitResult(True, 0, int(time.time()) + 60)
    
    def _create_key(self, identifier: str, limit_type: str, rate_limit_type: RateLimitType) -> str:
        """Create unique key for rate limiting"""
        # Hash identifier for privacy
        hashed_id = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"rate_limit:{rate_limit_type.value}:{limit_type}:{hashed_id}"
    
    async def _check_redis_rate_limit(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check rate limit using Redis sliding window"""
        now = time.time()
        window_start = now - limit.window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, limit.key_ttl)
        
        results = await pipe.execute()
        current_count = results[1]
        
        # Check if within limits (including burst)
        max_allowed = limit.requests + limit.burst_allowance
        allowed = current_count <= max_allowed
        
        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(now))
        
        remaining = max(0, max_allowed - current_count)
        reset_time = int(now + limit.window_seconds)
        retry_after = limit.window_seconds if not allowed else None
        
        return RateLimitResult(allowed, remaining, reset_time, retry_after)
    
    async def _check_local_rate_limit(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check rate limit using local cache (fallback)"""
        now = time.time()
        window_start = now - limit.window_seconds
        
        # Clean up old entries
        if key not in self.local_cache:
            self.local_cache[key] = {"requests": [], "last_cleanup": now}
        
        cache_entry = self.local_cache[key]
        
        # Remove old requests
        cache_entry["requests"] = [
            req_time for req_time in cache_entry["requests"] 
            if req_time > window_start
        ]
        
        current_count = len(cache_entry["requests"])
        max_allowed = limit.requests + limit.burst_allowance
        allowed = current_count < max_allowed
        
        if allowed:
            cache_entry["requests"].append(now)
        
        remaining = max(0, max_allowed - current_count - (1 if allowed else 0))
        reset_time = int(now + limit.window_seconds)
        retry_after = limit.window_seconds if not allowed else None
        
        return RateLimitResult(allowed, remaining, reset_time, retry_after)
    
    async def reset_rate_limit(self, identifier: str, limit_type: str, rate_limit_type: RateLimitType):
        """Reset rate limit for identifier (admin function)"""
        key = self._create_key(identifier, limit_type, rate_limit_type)
        
        if self.redis_client:
            await self.redis_client.delete(key)
        else:
            self.local_cache.pop(key, None)
        
        logger.info(f"Rate limit reset for key: {key}")
    
    async def get_rate_limit_info(
        self, 
        identifier: str, 
        limit_type: str, 
        rate_limit_type: RateLimitType
    ) -> Dict:
        """Get current rate limit information"""
        key = self._create_key(identifier, limit_type, rate_limit_type)
        limit_config = self.limits.get(limit_type)
        
        if not limit_config:
            return {"configured": False}
        
        now = time.time()
        window_start = now - limit_config.window_seconds
        
        if self.redis_client:
            # Count current requests in window
            current_count = await self.redis_client.zcount(key, window_start, now)
        else:
            cache_entry = self.local_cache.get(key, {"requests": []})
            current_count = len([
                req for req in cache_entry["requests"] 
                if req > window_start
            ])
        
        max_allowed = limit_config.requests + limit_config.burst_allowance
        
        return {
            "configured": True,
            "limit": limit_config.requests,
            "burst_allowance": limit_config.burst_allowance,
            "window_seconds": limit_config.window_seconds,
            "current_count": current_count,
            "remaining": max(0, max_allowed - current_count),
            "reset_time": int(now + limit_config.window_seconds)
        }
    
    async def cleanup_expired_entries(self):
        """Clean up expired local cache entries"""
        if not self.local_cache:
            return
        
        now = time.time()
        expired_keys = []
        
        for key, entry in self.local_cache.items():
            # Remove entries older than 1 hour
            if now - entry.get("last_cleanup", 0) > 3600:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit entries")
    
    async def get_statistics(self) -> Dict:
        """Get rate limiting statistics"""
        stats = {
            "backend": "redis" if self.redis_client else "local",
            "configured_limits": len(self.limits),
            "local_cache_entries": len(self.local_cache)
        }
        
        if self.redis_client:
            try:
                # Get Redis info
                info = await self.redis_client.info()
                stats["redis_connected"] = True
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            except Exception as e:
                stats["redis_connected"] = False
                stats["redis_error"] = str(e)
        
        return stats


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: AdvancedRateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting"""
        # Extract identifier (IP, user ID, API key)
        identifier = self._get_identifier(request)
        
        # Determine limit type based on endpoint
        limit_type = self._get_limit_type(request.url.path)
        
        # Determine rate limit type
        rate_limit_type = self._get_rate_limit_type(request)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(
            identifier, limit_type, rate_limit_type
        )
        
        if not result.allowed:
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            headers = {
                "X-RateLimit-Limit": str(self.rate_limiter.limits[limit_type].requests),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": str(result.reset_time),
                "Retry-After": str(result.retry_after) if result.retry_after else "60"
            }
            
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers=headers
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.limits[limit_type].requests)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_time)
        
        return response
    
    def _get_identifier(self, request) -> str:
        """Get identifier for rate limiting"""
        # Try to get user ID from auth
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:16]}"  # Use first 16 chars
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"
    
    def _get_limit_type(self, path: str) -> str:
        """Determine limit type based on endpoint"""
        if path.startswith("/api/v1/claims") and "batch" in path:
            return "api_batch"
        elif path.startswith("/api/v1/claims"):
            return "api_submit_claim"
        elif path.startswith("/login"):
            return "auth_login"
        elif path.startswith("/register"):
            return "auth_register"
        elif path.startswith("/api/"):
            return "api_general"
        else:
            return "api_general"
    
    def _get_rate_limit_type(self, request) -> RateLimitType:
        """Determine rate limit type"""
        if hasattr(request.state, 'user') and request.state.user:
            return RateLimitType.PER_USER
        elif request.headers.get("X-API-Key"):
            return RateLimitType.PER_API_KEY
        else:
            return RateLimitType.PER_IP


# Global rate limiter instance
rate_limiter: Optional[AdvancedRateLimiter] = None


async def get_rate_limiter() -> AdvancedRateLimiter:
    """Get global rate limiter instance"""
    global rate_limiter
    if not rate_limiter:
        rate_limiter = AdvancedRateLimiter()
        await rate_limiter.initialize()
    return rate_limiter


async def cleanup_rate_limiter():
    """Cleanup rate limiter resources"""
    global rate_limiter
    if rate_limiter and rate_limiter.redis_client:
        await rate_limiter.redis_client.close()
        rate_limiter = None