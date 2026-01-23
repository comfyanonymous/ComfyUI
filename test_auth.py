#!/usr/bin/env python3
"""
Test script for ComfyUI authentication module
"""

from httpx import Client as TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_auth import create_app
from comfyui_auth.models import Base, User
from comfyui_auth.auth import get_db

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app = create_app()
app.dependency_overrides[get_db] = override_get_db

# Create test database
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_register_user():
    """Test user registration"""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test12345"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert data["role"] == "user"

def test_register_duplicate_username():
    """Test registration with duplicate username"""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test2@example.com",
            "password": "Test12345"
        }
    )
    assert response.status_code == 400
    assert "Username already registered" in response.json()["detail"]

def test_register_duplicate_email():
    """Test registration with duplicate email"""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser2",
            "email": "test@example.com",
            "password": "Test12345"
        }
    )
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]

def test_login_user():
    """Test user login"""
    # First register a user
    client.post(
        "/api/auth/register",
        json={
            "username": "testlogin",
            "email": "login@example.com",
            "password": "Test12345"
        }
    )
    
    # Then login
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "Test12345"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_password():
    """Test login with invalid password"""
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "WrongPassword123"
        }
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

def test_protected_route():
    """Test protected route access"""
    # Login to get token
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "Test12345"
        }
    )
    token = response.json()["access_token"]
    
    # Access protected route
    response = client.get(
        "/api/auth/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "This is a protected route" in response.json()["message"]

def test_protected_route_no_auth():
    """Test protected route without authentication"""
    response = client.get("/api/auth/protected")
    assert response.status_code == 401

def test_refresh_token():
    """Test token refresh"""
    # Login to get tokens
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "Test12345"
        }
    )
    refresh_token = response.json()["refresh_token"]
    
    # Refresh token
    response = client.post(
        "/api/auth/refresh",
        headers={"Authorization": f"Bearer {refresh_token}"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_logout_user():
    """Test user logout"""
    # Login to get token
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "Test12345"
        }
    )
    token = response.json()["access_token"]
    
    # Logout
    response = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "Successfully logged out" in response.json()["message"]

def test_admin_route():
    """Test admin-only route"""
    # Login with regular user
    response = client.post(
        "/api/auth/login",
        data={
            "username": "testlogin",
            "password": "Test12345"
        }
    )
    token = response.json()["access_token"]
    
    # Access admin route
    response = client.get(
        "/api/auth/admin-only",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["detail"]

if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    
    # Test registration
    print("\n1. Testing user registration...")
    test_register_user()
    print("✓ Registration test passed")
    
    # Test duplicate registration
    print("\n2. Testing duplicate registration...")
    test_register_duplicate_username()
    test_register_duplicate_email()
    print("✓ Duplicate registration tests passed")
    
    # Test login
    print("\n3. Testing user login...")
    test_login_user()
    print("✓ Login test passed")
    
    # Test invalid login
    print("\n4. Testing invalid login...")
    test_login_invalid_password()
    print("✓ Invalid login test passed")
    
    # Test protected routes
    print("\n5. Testing protected routes...")
    test_protected_route()
    test_protected_route_no_auth()
    print("✓ Protected routes tests passed")
    
    # Test token refresh
    print("\n6. Testing token refresh...")
    test_refresh_token()
    print("✓ Token refresh test passed")
    
    # Test logout
    print("\n7. Testing logout...")
    test_logout_user()
    print("✓ Logout test passed")
    
    # Test admin route
    print("\n8. Testing admin route...")
    test_admin_route()
    print("✓ Admin route test passed")
    
    print("\n✅ All tests passed!")
    
    # Clean up test database
    os.remove("test.db")