"""
Authentication service
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from defame.core.database import DatabaseSession
from defame.auth.models import (
    User, Role, Permission, APIKey,
    DEFAULT_PERMISSIONS, DEFAULT_ROLES, DEFAULT_ROLE_PERMISSIONS
)
from defame.utils.logger import get_logger

logger = get_logger(__name__)


class AuthService:
    """Authentication and authorization service"""
    
    @staticmethod
    def create_user(
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None
    ) -> User:
        """Create new user"""
        with DatabaseSession() as db:
            # Check if user exists
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise ValueError("User with this username or email already exists")
            
            # Create user
            user = User(
                username=username,
                email=email,
                full_name=full_name
            )
            user.set_password(password)
            
            db.add(user)
            db.flush()  # Get user ID
            
            # Assign roles
            if roles:
                for role_name in roles:
                    role = db.query(Role).filter(Role.name == role_name).first()
                    if role:
                        user.roles.append(role)
            else:
                # Default role
                default_role = db.query(Role).filter(Role.name == "user").first()
                if default_role:
                    user.roles.append(default_role)
            
            db.commit()
            return user
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        with DatabaseSession() as db:
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if user and user.is_active and user.check_password(password):
                # Update last login
                user.last_login = datetime.utcnow()
                db.commit()
                return user
            
            return None
    
    @staticmethod
    def create_api_key(
        user_id: str,
        name: str,
        expires_days: Optional[int] = None
    ) -> tuple[APIKey, str]:
        """Create API key for user"""
        with DatabaseSession() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            # Generate key
            key_value = APIKey.generate_key()
            
            # Create API key object
            api_key = APIKey(
                user_id=user_id,
                name=name,
                expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
            )
            api_key.set_key(key_value)
            
            db.add(api_key)
            db.commit()
            
            return api_key, key_value
    
    @staticmethod
    def revoke_api_key(api_key_id: str) -> bool:
        """Revoke API key"""
        with DatabaseSession() as db:
            api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
            if api_key:
                api_key.is_active = False
                db.commit()
                return True
            return False
    
    @staticmethod
    def get_user_permissions(user_id: str) -> List[str]:
        """Get all permissions for user"""
        with DatabaseSession() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return []
            
            permissions = set()
            for role in user.roles:
                for permission in role.permissions:
                    permissions.add(permission.name)
            
            return list(permissions)
    
    @staticmethod
    def initialize_default_data():
        """Initialize default roles and permissions"""
        with DatabaseSession() as db:
            # Create permissions
            existing_permissions = {p.name for p in db.query(Permission).all()}
            
            for name, description, resource, action in DEFAULT_PERMISSIONS:
                if name not in existing_permissions:
                    permission = Permission(
                        name=name,
                        description=description,
                        resource=resource,
                        action=action
                    )
                    db.add(permission)
            
            db.commit()
            
            # Create roles
            existing_roles = {r.name for r in db.query(Role).all()}
            
            for name, description in DEFAULT_ROLES:
                if name not in existing_roles:
                    role = Role(name=name, description=description)
                    db.add(role)
            
            db.commit()
            
            # Assign permissions to roles
            for role_name, permission_names in DEFAULT_ROLE_PERMISSIONS.items():
                role = db.query(Role).filter(Role.name == role_name).first()
                if role:
                    # Clear existing permissions
                    role.permissions.clear()
                    
                    # Add new permissions
                    for permission_name in permission_names:
                        permission = db.query(Permission).filter(
                            Permission.name == permission_name
                        ).first()
                        if permission:
                            role.permissions.append(permission)
            
            db.commit()
            logger.info("Default authentication data initialized")
    
    @staticmethod
    def create_admin_user(
        username: str = "admin",
        email: str = "admin@verisphere.local",
        password: str = "admin123"
    ) -> User:
        """Create default admin user"""
        try:
            return AuthService.create_user(
                username=username,
                email=email,
                password=password,
                full_name="System Administrator",
                roles=["admin"]
            )
        except ValueError as e:
            logger.warning(f"Admin user creation failed: {e}")
            # Return existing user
            with DatabaseSession() as db:
                return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_stats() -> Dict[str, Any]:
        """Get user statistics"""
        with DatabaseSession() as db:
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            verified_users = db.query(User).filter(User.is_verified == True).count()
            
            # Recent logins (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_logins = db.query(User).filter(
                User.last_login >= recent_cutoff
            ).count()
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "verified_users": verified_users,
                "recent_logins": recent_logins
            }