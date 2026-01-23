# ComfyUI Authentication Module

A complete authentication and session management system for ComfyUI, built with Python, FastAPI, JWT, and bcrypt.

## Features

### Core Authentication
- **User Registration**: Secure registration with username, email, and password (bcrypt hashed storage)
- **User Login**: JWT token-based authentication with "Remember Me" functionality
- **Session Management**: Token refresh, logout, and session revocation
- **Password Security**: bcrypt hashing with configurable work factor

### Security Features
- **Rate Limiting**: Prevent brute force attacks with login attempt limits
- **Account Lockout**: Temporary lockout after multiple failed attempts
- **Input Validation**: Sanitize and validate all user inputs
- **Security Headers**: Comprehensive HTTP security headers
- **CSRF Protection**: Built-in CSRF token generation and validation

### Authorization
- **RBAC (Role-Based Access Control)**: Admin and user roles
- **Route Protection**: Decorators to protect API endpoints
- **Middleware**: Automatic token validation for protected routes

### Database Support
- **SQLite**: Default database for development
- **PostgreSQL**: Production-ready database support
- **SQLAlchemy**: ORM for database interactions

## Project Structure

```
comfyui_auth/
├── __init__.py          # Main application factory
├── models.py            # Database models (User, Session, FailedLoginAttempt)
├── auth.py              # Authentication core logic
├── routes.py            # API endpoints
├── middleware.py        # Authentication middleware
├── utils.py             # Utility functions (JWT, password hashing)
└── security.py          # Security utilities

web/
├── login.html           # Login page
├── register.html        # Registration page
└── auth-utils.js        # Frontend authentication utilities

.env.example             # Environment variables template
requirements.txt         # Python dependencies
README.md               # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize database**:
The database will be automatically created when the application starts.

## Usage

### Starting the Application

```bash
uvicorn comfyui_auth:app --host 0.0.0.0 --port 8188
```

### API Endpoints

#### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get tokens
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Logout and revoke session
- `GET /api/auth/me` - Get current user info

#### Protected Routes
- `GET /api/auth/protected` - Example protected route
- `GET /api/auth/admin-only` - Admin-only route

### Frontend Integration

The authentication module includes a complete frontend implementation:

- **Login Page**: `/web/login.html`
- **Registration Page**: `/web/register.html`
- **Auth Utilities**: `/web/auth-utils.js`

The frontend handles:
- Token storage (localStorage/sessionStorage)
- Automatic token refresh
- API requests with Authorization headers
- Login state management
- Route protection

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./comfyui_auth.db` |
| `JWT_SECRET` | Secret key for JWT signing | `your-secret-key-here` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime | `7` |
| `MAX_FAILED_ATTEMPTS` | Max failed login attempts before lockout | `5` |
| `LOCKOUT_DURATION_MINUTES` | Lockout duration after failed attempts | `15` |
| `DEBUG` | Debug mode | `False` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:8188` |

### Security Configuration

Password policy:
- Minimum length: 8 characters
- Require uppercase letter
- Require lowercase letter
- Require digit
- Optional symbols (configurable)

## Testing

Run the test suite:

```bash
python test_auth.py
```

Check routes and security configuration:

```bash
python check_routes.py
```

## Integration with ComfyUI

To integrate with an existing ComfyUI installation:

1. **Copy the `comfyui_auth` directory** to your ComfyUI installation
2. **Update ComfyUI's main application** to include the auth module
3. **Protect existing routes** with the `@require_auth` decorator
4. **Update frontend** to include authentication state management

### Example Integration

```python
# In your ComfyUI main app
from comfyui_auth import create_app
from comfyui_auth.auth import require_auth

app = create_app()

# Protect existing routes
@app.get("/api/workflows")
@require_auth
def get_workflows():
    # Your workflow logic here
    pass
```

## Security Best Practices

1. **Use HTTPS**: Always use HTTPS in production to protect tokens in transit
2. **Strong JWT Secret**: Use a long, random secret key (at least 32 characters)
3. **Rotate Secrets**: Regularly rotate JWT secrets and database credentials
4. **Monitor Logs**: Monitor authentication logs for suspicious activity
5. **Keep Dependencies Updated**: Regularly update Python dependencies
6. **Limit Failed Attempts**: Configure appropriate lockout settings
7. **Use Environment Variables**: Never hardcode secrets in code

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Note**: This is a standalone authentication module designed for ComfyUI. It can be used as a starting point or integrated into existing ComfyUI installations.