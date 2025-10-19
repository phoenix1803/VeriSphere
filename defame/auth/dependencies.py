"""
FastAPI authentication dependencies
"""
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session
from typing import Optional

from defame.core.database import DatabaseSession
from defame.auth.models import User, APIKey
from defame.utils.logger import get_logger

logger = get_logger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Optional[User]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    
    try:
        user = User.verify_jwt_token(credentials.credentials)
        if user and user.is_active:
            return user
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
    
    return None


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get current user from API key"""
    if not api_key:
        return None
    
    try:
        with DatabaseSession() as db:
            # Find API key
            api_key_obj = db.query(APIKey).filter(APIKey.is_active == True).all()
            
            for key_obj in api_key_obj:
                if key_obj.check_key(api_key):
                    # Update last used
                    from datetime import datetime
                    key_obj.last_used = datetime.utcnow()
                    db.commit()
                    
                    # Return user if active
                    if key_obj.user.is_active:
                        return key_obj.user
                    break
    except Exception as e:
        logger.warning(f"API key verification failed: {e}")
    
    return None


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> Optional[User]:
    """Get current user from either JWT token or API key"""
    return token_user or api_key_user


async def require_authentication(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_permission(permission: str):
    """Require user to have specific permission"""
    async def permission_checker(
        current_user: User = Depends(require_authentication)
    ) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker


def require_role(role: str):
    """Require user to have specific role"""
    async def role_checker(
        current_user: User = Depends(require_authentication)
    ) -> User:
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    
    return role_checker


async def get_optional_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    return current_user