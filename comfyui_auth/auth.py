from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Dict, Any
import uuid
import os

from .models import User, Session as DB_Session, FailedLoginAttempt
from .utils import (
    create_access_token, create_refresh_token, verify_token,
    validate_email, validate_username, validate_password,
    JWT_SECRET, JWT_ALGORITHM
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Security configuration
MAX_FAILED_ATTEMPTS = int(os.getenv('MAX_FAILED_ATTEMPTS', '5'))
LOCKOUT_DURATION_MINUTES = int(os.getenv('LOCKOUT_DURATION_MINUTES', '15'))

def get_db():
    from .models import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def register_user(db: Session, username: str, email: str, password: str) -> User:
    # Validate input
    if not validate_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 3-50 characters long and can only contain letters, numbers, and underscores"
        )
    
    if not validate_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    if not validate_password(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, and one number"
        )
    
    # Check if user already exists
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = User(username=username, email=email)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

def authenticate_user(db: Session, username: str, password: str, ip_address: str = None) -> User:
    # Check for failed login attempts
    check_failed_attempts(db, username, ip_address)
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        log_failed_login_attempt(db, username, ip_address)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.check_password(password):
        log_failed_login_attempt(db, username, ip_address)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Reset failed attempts on successful login
    reset_failed_login_attempts(db, username, ip_address)
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user

def create_user_session(db: Session, user_id: int, remember_me: bool = False) -> Dict[str, str]:
    # Create access token
    access_token = create_access_token(data={"sub": str(user_id)})
    
    # Create refresh token
    refresh_token = create_refresh_token(data={"sub": str(user_id)})
    
    # Store refresh token in database
    expires_days = 30 if remember_me else 7
    expires_at = datetime.utcnow() + timedelta(days=expires_days)
    
    db_session = DB_Session(
        user_id=user_id,
        refresh_token=refresh_token,
        expires_at=expires_at
    )
    
    db.add(db_session)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

def refresh_access_token(db: Session, refresh_token: str) -> Dict[str, str]:
    try:
        payload = verify_token(refresh_token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if session exists and is valid
        db_session = db.query(DB_Session).filter(
            DB_Session.refresh_token == refresh_token,
            DB_Session.is_revoked == False
        ).first()
        
        if not db_session or db_session.is_expired():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired or is invalid"
            )
        
        # Create new access token
        access_token = create_access_token(data={"sub": user_id})
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

def logout_user(db: Session, refresh_token: str = None, user_id: int = None):
    if refresh_token:
        db_session = db.query(DB_Session).filter(
            DB_Session.refresh_token == refresh_token
        ).first()
        if db_session:
            db_session.is_revoked = True
            db.commit()
    elif user_id:
        # Revoke all sessions for user
        db.query(DB_Session).filter(
            DB_Session.user_id == user_id
        ).update({"is_revoked": True})
        db.commit()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No token or user ID provided"
        )

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    try:
        payload = verify_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def require_role(required_role: str):
    def role_checker(user: User = Depends(get_current_user)):
        if user.role != required_role and user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return role_checker

def log_failed_login_attempt(db: Session, username: str, ip_address: str = None):
    attempt = FailedLoginAttempt(
        username=username,
        ip_address=ip_address
    )
    db.add(attempt)
    db.commit()

def check_failed_attempts(db: Session, username: str, ip_address: str = None):
    # Check by username
    recent_attempts = db.query(FailedLoginAttempt).filter(
        FailedLoginAttempt.username == username,
        FailedLoginAttempt.attempt_time >= datetime.utcnow() - timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    ).count()
    
    if recent_attempts >= MAX_FAILED_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Please try again after {LOCKOUT_DURATION_MINUTES} minutes."
        )
    
    # Check by IP address if provided
    if ip_address:
        ip_attempts = db.query(FailedLoginAttempt).filter(
            FailedLoginAttempt.ip_address == ip_address,
            FailedLoginAttempt.attempt_time >= datetime.utcnow() - timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        ).count()
        
        if ip_attempts >= MAX_FAILED_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed login attempts from this IP. Please try again after {LOCKOUT_DURATION_MINUTES} minutes."
            )

def reset_failed_login_attempts(db: Session, username: str, ip_address: str = None):
    # Reset attempts for username
    db.query(FailedLoginAttempt).filter(
        FailedLoginAttempt.username == username
    ).delete()
    
    # Reset attempts for IP if provided
    if ip_address:
        db.query(FailedLoginAttempt).filter(
            FailedLoginAttempt.ip_address == ip_address
        ).delete()
    
    db.commit()