"""
Web UI routes for VeriSphere dashboard
"""
from fastapi import APIRouter, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional

from defame.auth.dependencies import get_current_user, require_authentication
from defame.auth.models import User
from defame.auth.service import AuthService
from defame.utils.logger import get_logger

logger = get_logger(__name__)

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Create router
web_router = APIRouter()


@web_router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Main dashboard page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": current_user}
    )


@web_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@web_router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle login form submission"""
    try:
        user = AuthService.authenticate_user(username, password)
        if not user:
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Invalid username or password"
                }
            )
        
        # Generate JWT token
        token = user.generate_jwt_token()
        
        # Create response with redirect
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        
        # Set token in cookie (in production, use secure, httponly cookies)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Login failed. Please try again."
            }
        )


@web_router.get("/logout")
async def logout():
    """Logout user"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response


@web_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    return templates.TemplateResponse(
        "register.html",
        {"request": request}
    )


@web_router.post("/register")
async def register_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    full_name: Optional[str] = Form(None)
):
    """Handle registration form submission"""
    try:
        # Validate passwords match
        if password != confirm_password:
            return templates.TemplateResponse(
                "register.html",
                {
                    "request": request,
                    "error": "Passwords do not match"
                }
            )
        
        # Create user
        user = AuthService.create_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name
        )
        
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "success": "Account created successfully. Please log in."
            }
        )
        
    except ValueError as e:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Registration failed. Please try again."
            }
        )


@web_router.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: User = Depends(require_authentication)
):
    """User profile page"""
    return templates.TemplateResponse(
        "profile.html",
        {"request": request, "user": current_user}
    )


@web_router.get("/api-keys", response_class=HTMLResponse)
async def api_keys_page(
    request: Request,
    current_user: User = Depends(require_authentication)
):
    """API keys management page"""
    return templates.TemplateResponse(
        "api_keys.html",
        {"request": request, "user": current_user}
    )


@web_router.post("/api-keys/create")
async def create_api_key(
    request: Request,
    name: str = Form(...),
    expires_days: Optional[int] = Form(None),
    current_user: User = Depends(require_authentication)
):
    """Create new API key"""
    try:
        api_key, key_value = AuthService.create_api_key(
            user_id=str(current_user.id),
            name=name,
            expires_days=expires_days
        )
        
        return templates.TemplateResponse(
            "api_keys.html",
            {
                "request": request,
                "user": current_user,
                "new_key": key_value,
                "success": f"API key '{name}' created successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        return templates.TemplateResponse(
            "api_keys.html",
            {
                "request": request,
                "user": current_user,
                "error": "Failed to create API key"
            }
        )


# Static files setup function
def setup_static_files(app):
    """Setup static file serving"""
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")