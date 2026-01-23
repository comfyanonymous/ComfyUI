from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from .auth import (
    register_user, authenticate_user, create_user_session,
    refresh_access_token, logout_user, get_current_user,
    get_db, require_role
)
from .models import User

router = APIRouter(prefix="/api/auth")

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None
    
    model_config = {
        "from_attributes": True
    }

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = register_user(db, user.username, user.email, user.password)
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        role=db_user.role,
        is_active=db_user.is_active,
        created_at=db_user.created_at.isoformat() if db_user.created_at else None,
        last_login=db_user.last_login.isoformat() if db_user.last_login else None
    )

@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    remember_me: bool = False
):
    # Get client IP
    client_ip = request.client.host if request.client else None
    
    user = authenticate_user(db, form_data.username, form_data.password, client_ip)
    tokens = create_user_session(db, user.id, remember_me)
    return tokens

@router.post("/refresh")
async def refresh(
    request: Request,
    db: Session = Depends(get_db)
):
    # Get refresh token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found"
        )
    
    refresh_token = auth_header.split(" ")[1]
    new_tokens = refresh_access_token(db, refresh_token)
    return new_tokens

@router.post("/logout")
async def logout(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get refresh token from Authorization header
    auth_header = request.headers.get("Authorization")
    refresh_token = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else None
    
    logout_user(db, refresh_token, current_user.id)
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
        last_login=current_user.last_login.isoformat() if current_user.last_login else None
    )

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {
        "message": "This is a protected route",
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role
        }
    }

@router.get("/admin-only")
async def admin_route(current_user: User = Depends(require_role("admin"))):
    return {
        "message": "This is an admin-only route",
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role
        }
    }

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "auth"}