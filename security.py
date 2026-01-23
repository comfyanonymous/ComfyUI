#!/usr/bin/env python3
"""
Security utilities for ComfyUI authentication module
"""

import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .models import FailedLoginAttempt, User

# Password policy
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGITS = True
PASSWORD_REQUIRE_SYMBOLS = False

# Rate limiting
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# Session management
SESSION_EXPIRY_DAYS = 7
REMEMBER_ME_EXPIRY_DAYS = 30

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'",
}

def generate_secure_password(length: int = 12) -> str:
    """Generate a secure random password"""
    if length < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password length must be at least {PASSWORD_MIN_LENGTH}")
    
    # Define character sets
    uppercase = string.ascii_uppercase
    lowercase = string.ascii_lowercase
    digits = string.digits
    symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Start with one character from each required set
    password = []
    
    if PASSWORD_REQUIRE_UPPERCASE:
        password.append(secrets.choice(uppercase))
    if PASSWORD_REQUIRE_LOWERCASE:
        password.append(secrets.choice(lowercase))
    if PASSWORD_REQUIRE_DIGITS:
        password.append(secrets.choice(digits))
    if PASSWORD_REQUIRE_SYMBOLS:
        password.append(secrets.choice(symbols))
    
    # Fill the rest with random characters from all sets
    all_chars = uppercase + lowercase + digits + symbols
    remaining_length = length - len(password)
    password += [secrets.choice(all_chars) for _ in range(remaining_length)]
    
    # Shuffle the password
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)

def validate_password(password: str) -> bool:
    """Validate password against policy"""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False
    
    if PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        return False
    
    if PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        return False
    
    if PASSWORD_REQUIRE_DIGITS and not re.search(r'[0-9]', password):
        return False
    
    if PASSWORD_REQUIRE_SYMBOLS and not re.search(r'[!@#$%^&*()_+-=\[\]{}|;:,.<>?]', password):
        return False
    
    return True

def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_pattern, email) is not None

def validate_username(username: str) -> bool:
    """Validate username format"""
    if len(username) < 3 or len(username) > 50:
        return False
    username_pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(username_pattern, username) is not None

def check_failed_login_attempts(db: Session, username: str, ip_address: str = None) -> None:
    """Check for failed login attempts and enforce lockout"""
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

def log_failed_login_attempt(db: Session, username: str, ip_address: str = None) -> None:
    """Log a failed login attempt"""
    attempt = FailedLoginAttempt(
        username=username,
        ip_address=ip_address
    )
    db.add(attempt)
    db.commit()

def reset_failed_login_attempts(db: Session, username: str, ip_address: str = None) -> None:
    """Reset failed login attempts for a user/IP"""
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

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent XSS attacks"""
    if not input_str:
        return ""
    
    # Basic HTML escaping
    input_str = input_str.replace('&', '&amp;')
    input_str = input_str.replace('<', '&lt;')
    input_str = input_str.replace('>', '&gt;')
    input_str = input_str.replace('"', '&quot;')
    input_str = input_str.replace("'", '&#39;')
    
    return input_str

def generate_csrf_token() -> str:
    """Generate a secure CSRF token"""
    return secrets.token_urlsafe(32)

def is_secure_password(password: str) -> bool:
    """Check if password is secure (additional checks beyond validation)"""
    # Check for common passwords
    common_passwords = {
        'password', '123456', '123456789', '12345678', '12345',
        'qwerty', 'abc123', '111111', '123123', '1234567',
        '1234567890', 'password1', 'iloveyou', 'princess', '1234'
    }
    
    if password.lower() in common_passwords:
        return False
    
    # Check for sequential characters
    sequential_patterns = [
        '123456', '234567', '345678', '456789', '567890',
        'abcdef', 'bcdefg', 'cdefgh', 'defghi', 'efghij',
        'fedcba', 'zyxwvuts', 'tsrqpo', 'ponmlk', 'lkjihg'
    ]
    
    for pattern in sequential_patterns:
        if pattern in password.lower():
            return False
    
    return True

def get_security_headers() -> Dict[str, str]:
    """Get security headers for responses"""
    return SECURITY_HEADERS.copy()

def secure_response(response) -> None:
    """Add security headers to a response"""
    headers = get_security_headers()
    for key, value in headers.items():
        response.headers[key] = value

def audit_log(db: Session, action: str, user_id: Optional[int] = None, details: Optional[Dict] = None) -> None:
    """Log security-related actions"""
    # In a production system, this would write to an audit log table
    # For now, we'll just print to console
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user_id': user_id,
        'details': details
    }
    print(f"AUDIT LOG: {log_entry}")