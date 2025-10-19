"""
Authentication and authorization models
"""
from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timedelta
import uuid
import bcrypt
import jwt
from typing import List, Optional

from defame.core.database import Base
from config.globals import get_config

config = get_config()

# Association table for user roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True)
)

# Association table for role permissions
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id'), primary_key=True)
)


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """Hash and set password"""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Check if password is correct"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if user has specific permission"""
        for role in self.roles:
            if role.has_permission(permission_name):
                return True
        return False
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return any(role.name == role_name for role in self.roles)
    
    def generate_jwt_token(self, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT token for user"""
        if expires_delta is None:
            expires_delta = timedelta(minutes=config.jwt_expire_minutes)
        
        expire = datetime.utcnow() + expires_delta
        payload = {
            'user_id': str(self.id),
            'username': self.username,
            'exp': expire,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
    
    @staticmethod
    def verify_jwt_token(token: str) -> Optional['User']:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, config.jwt_secret_key, algorithms=[config.jwt_algorithm])
            user_id = payload.get('user_id')
            if user_id:
                from defame.core.database import DatabaseSession
                with DatabaseSession() as db:
                    return db.query(User).filter(User.id == user_id).first()
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        return None


class Role(Base):
    """Role model"""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if role has specific permission"""
        return any(perm.name == permission_name for perm in self.permissions)


class Permission(Base):
    """Permission model"""
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    resource = Column(String(50), nullable=False)  # claims, agents, system, etc.
    action = Column(String(50), nullable=False)    # create, read, update, delete, execute
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class APIKey(Base):
    """API Key model"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    @staticmethod
    def generate_key() -> str:
        """Generate new API key"""
        return f"vrs_{uuid.uuid4().hex}"
    
    def set_key(self, key: str):
        """Hash and set API key"""
        salt = bcrypt.gensalt()
        self.key_hash = bcrypt.hashpw(key.encode('utf-8'), salt).decode('utf-8')
    
    def check_key(self, key: str) -> bool:
        """Check if API key is correct"""
        return bcrypt.checkpw(key.encode('utf-8'), self.key_hash.encode('utf-8'))


# Default permissions
DEFAULT_PERMISSIONS = [
    # Claims permissions
    ("claims.create", "Create new claims", "claims", "create"),
    ("claims.read", "Read claim data", "claims", "read"),
    ("claims.update", "Update claim data", "claims", "update"),
    ("claims.delete", "Delete claims", "claims", "delete"),
    ("claims.cancel", "Cancel claim processing", "claims", "cancel"),
    
    # System permissions
    ("system.status", "View system status", "system", "read"),
    ("system.metrics", "View system metrics", "system", "read"),
    ("system.health", "View health checks", "system", "read"),
    ("system.admin", "System administration", "system", "admin"),
    
    # Agent permissions
    ("agents.view", "View agent status", "agents", "read"),
    ("agents.manage", "Manage agents", "agents", "update"),
    
    # API permissions
    ("api.access", "Access API endpoints", "api", "read"),
    ("api.batch", "Use batch processing", "api", "create"),
]

# Default roles
DEFAULT_ROLES = [
    ("admin", "System administrator with full access"),
    ("user", "Regular user with basic access"),
    ("api_user", "API-only user for integrations"),
    ("readonly", "Read-only access to system"),
]

# Role-permission mappings
DEFAULT_ROLE_PERMISSIONS = {
    "admin": [
        "claims.create", "claims.read", "claims.update", "claims.delete", "claims.cancel",
        "system.status", "system.metrics", "system.health", "system.admin",
        "agents.view", "agents.manage",
        "api.access", "api.batch"
    ],
    "user": [
        "claims.create", "claims.read", "claims.cancel",
        "system.status", "system.health",
        "api.access"
    ],
    "api_user": [
        "claims.create", "claims.read", "claims.cancel",
        "api.access", "api.batch"
    ],
    "readonly": [
        "claims.read",
        "system.status", "system.health",
        "agents.view"
    ]
}