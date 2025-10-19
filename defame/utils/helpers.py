"""
Utility functions and helpers for VeriSphere
"""
import hashlib
import uuid
import re
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import aiohttp
import time
from functools import wraps
from urllib.parse import urlparse, urljoin
import mimetypes

from defame.utils.logger import get_logger

logger = get_logger(__name__)


def generate_claim_id() -> str:
    """Generate unique claim ID"""
    return str(uuid.uuid4())


def generate_hash(content: str) -> str:
    """Generate SHA-256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize and clean text input"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove potentially harmful characters
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\~\`]', '', text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text"""
    mention_pattern = re.compile(r'@(\w+)')
    return mention_pattern.findall(text)


def extract_hashtags(text: str) -> List[str]:
    """Extract #hashtags from text"""
    hashtag_pattern = re.compile(r'#(\w+)')
    return hashtag_pattern.findall(text)


def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_domain(url: str) -> Optional[str]:
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity score"""
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def format_confidence_score(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"


def format_processing_time(seconds: float) -> str:
    """Format processing time in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: Any = None) -> str:
    """Safely serialize object to JSON"""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(default) if default is not None else "{}"


def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async functions with retry logic"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s",
                            error=str(e)
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}",
                            error=str(e)
                        )
            
            raise last_exception
        return wrapper
    return decorator


def timeout_async(seconds: float):
    """Decorator to add timeout to async functions"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if we can make a call
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


async def fetch_url(url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fetch URL content with error handling"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers=headers or {}) as response:
                content = await response.text()
                return {
                    "url": url,
                    "status_code": response.status,
                    "content": content,
                    "headers": dict(response.headers),
                    "success": True
                }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "success": False
        }


def validate_image_file(file_path: Union[str, Path], max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """Validate image file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    # Check file size
    if file_path.stat().st_size > max_size:
        return {"valid": False, "error": f"File too large (max {max_size} bytes)"}
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type or not mime_type.startswith('image/'):
        return {"valid": False, "error": "Not a valid image file"}
    
    return {
        "valid": True,
        "size": file_path.stat().st_size,
        "mime_type": mime_type
    }


def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average of values"""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    if sum(weights) == 0:
        return sum(values) / len(values)  # Simple average if no weights
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to specified range"""
    if max_val <= min_val:
        return min_val
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of file"""
    file_path = Path(file_path)
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def create_directory_if_not_exists(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def time_ago(timestamp: datetime) -> str:
    """Get human-readable time difference"""
    now = datetime.utcnow()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"


class ConfigValidator:
    """Validate configuration dictionaries"""
    
    @staticmethod
    def validate_required_keys(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate that all required keys are present"""
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        return missing_keys
    
    @staticmethod
    def validate_types(config: Dict[str, Any], type_specs: Dict[str, type]) -> List[str]:
        """Validate that values have correct types"""
        type_errors = []
        for key, expected_type in type_specs.items():
            if key in config and not isinstance(config[key], expected_type):
                type_errors.append(f"{key} should be {expected_type.__name__}, got {type(config[key]).__name__}")
        return type_errors


# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test text processing
    text = "This is a test with @mention and #hashtag and https://example.com"
    print(f"URLs: {extract_urls(text)}")
    print(f"Mentions: {extract_mentions(text)}")
    print(f"Hashtags: {extract_hashtags(text)}")
    
    # Test similarity
    text1 = "The quick brown fox"
    text2 = "A quick brown fox jumps"
    print(f"Similarity: {calculate_text_similarity(text1, text2):.2f}")
    
    # Test formatting
    print(f"Confidence: {format_confidence_score(0.856)}")
    print(f"Time: {format_processing_time(125.7)}")
    
    print("Utility functions test completed.")