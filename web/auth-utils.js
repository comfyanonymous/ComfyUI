/**
 * Authentication utilities for ComfyUI
 * Handles token storage, API requests with auth headers, and login state management
 */

class AuthManager {
    constructor() {
        this.accessToken = localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token') || sessionStorage.getItem('refresh_token');
        this.user = null;
    }
    
    isLoggedIn() {
        return !!this.accessToken;
    }
    
    async loadUser() {
        if (!this.isLoggedIn()) return null;
        
        try {
            const response = await fetch('/api/auth/me', {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });
            
            if (response.ok) {
                this.user = await response.json();
                return this.user;
            }
            
            // Token might be expired, try to refresh
            if (response.status === 401 && this.refreshToken) {
                const refreshed = await this.refreshAccessToken();
                if (refreshed) {
                    return await this.loadUser();
                }
            }
            
            return null;
        } catch (error) {
            console.error('Failed to load user:', error);
            return null;
        }
    }
    
    async refreshAccessToken() {
        if (!this.refreshToken) return false;
        
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.refreshToken}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.accessToken = data.access_token;
                
                // Store token in same place as refresh token
                if (localStorage.getItem('refresh_token')) {
                    localStorage.setItem('access_token', this.accessToken);
                } else {
                    sessionStorage.setItem('access_token', this.accessToken);
                }
                
                return true;
            }
            
            // Refresh token expired or invalid
            this.logout();
            return false;
        } catch (error) {
            console.error('Failed to refresh token:', error);
            this.logout();
            return false;
        }
    }
    
    async fetchWithAuth(url, options = {}) {
        if (!this.isLoggedIn()) {
            throw new Error('Not authenticated');
        }
        
        const headers = options.headers || {};
        headers['Authorization'] = `Bearer ${this.accessToken}`;
        
        try {
            const response = await fetch(url, {
                ...options,
                headers
            });
            
            // Handle token expiration
            if (response.status === 401 && this.refreshToken) {
                const refreshed = await this.refreshAccessToken();
                if (refreshed) {
                    headers['Authorization'] = `Bearer ${this.accessToken}`;
                    return await fetch(url, {
                        ...options,
                        headers
                    });
                }
            }
            
            return response;
        } catch (error) {
            console.error('Fetch with auth failed:', error);
            throw error;
        }
    }
    
    logout() {
        this.accessToken = null;
        this.refreshToken = null;
        this.user = null;
        
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        sessionStorage.removeItem('access_token');
        sessionStorage.removeItem('refresh_token');
        
        // Redirect to login page
        window.location.href = '/web/login.html';
    }
    
    storeTokens(accessToken, refreshToken, rememberMe = false) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
        
        if (rememberMe) {
            localStorage.setItem('access_token', accessToken);
            localStorage.setItem('refresh_token', refreshToken);
        } else {
            sessionStorage.setItem('access_token', accessToken);
            sessionStorage.setItem('refresh_token', refreshToken);
        }
    }
}

// Create singleton instance
const authManager = new AuthManager();

/**
 * Route guard to protect pages that require authentication
 */
async function requireAuth() {
    if (!authManager.isLoggedIn()) {
        window.location.href = '/web/login.html';
        return false;
    }
    
    // Check if token is valid by loading user
    const user = await authManager.loadUser();
    if (!user) {
        authManager.logout();
        return false;
    }
    
    return true;
}

/**
 * Route guard to protect admin-only pages
 */
async function requireAdmin() {
    const user = await requireAuth();
    if (!user) return false;
    
    if (user.role !== 'admin') {
        // Redirect to forbidden page or home
        window.location.href = '/';
        return false;
    }
    
    return true;
}

/**
 * Intercept fetch requests to add auth headers automatically
 */
const originalFetch = window.fetch;
window.fetch = async function(url, options) {
    // Skip auth for public endpoints
    const publicEndpoints = [
        '/api/auth/register',
        '/api/auth/login',
        '/api/auth/health',
        '/web/',
        '/static/',
        '/docs/',
        '/redoc/'
    ];
    
    const isPublic = publicEndpoints.some(endpoint => 
        url.startsWith(endpoint) || url === endpoint
    );
    
    if (!isPublic && authManager.isLoggedIn()) {
        options = options || {};
        options.headers = options.headers || {};
        
        // Add auth header
        options.headers['Authorization'] = `Bearer ${authManager.accessToken}`;
        
        try {
            const response = await originalFetch(url, options);
            
            // Handle token expiration
            if (response.status === 401 && authManager.refreshToken) {
                const refreshed = await authManager.refreshAccessToken();
                if (refreshed) {
                    options.headers['Authorization'] = `Bearer ${authManager.accessToken}`;
                    return await originalFetch(url, options);
                }
            }
            
            return response;
        } catch (error) {
            console.error('Fetch failed:', error);
            throw error;
        }
    }
    
    return originalFetch(url, options);
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        authManager,
        requireAuth,
        requireAdmin
    };
}