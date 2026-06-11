/**
 * ComfyUI API Key Injection
 * This script automatically adds the stored API key to all HTTP requests
 */

(function() {
    'use strict';

    // Get the stored API key
    function getApiKey() {
        return localStorage.getItem('comfyui_api_key') || sessionStorage.getItem('comfyui_api_key');
    }

    // Check if user is authenticated
    function checkAuth() {
        const apiKey = getApiKey();
        if (!apiKey && window.location.pathname !== '/auth_login.html') {
            // Redirect to login page if no API key is found
            window.location.href = '/auth_login.html';
            return false;
        }
        return true;
    }

    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const apiKey = getApiKey();
        
        if (apiKey) {
            // Clone or create the options object
            let [url, options = {}] = args;
            
            // Initialize headers if not present
            if (!options.headers) {
                options.headers = {};
            }
            
            // Convert Headers object to plain object if needed
            if (options.headers instanceof Headers) {
                const headersObj = {};
                options.headers.forEach((value, key) => {
                    headersObj[key] = value;
                });
                options.headers = headersObj;
            }
            
            // Add Authorization header if not already present
            if (!options.headers['Authorization'] && !options.headers['authorization']) {
                options.headers['Authorization'] = `Bearer ${apiKey}`;
            }
            
            // Update args
            args = [url, options];
        }
        
        // Call original fetch and handle 401 errors
        return originalFetch.apply(this, args).then(response => {
            if (response.status === 401 && window.location.pathname !== '/auth_login.html') {
                // Clear stored API key and redirect to login
                localStorage.removeItem('comfyui_api_key');
                sessionStorage.removeItem('comfyui_api_key');
                window.location.href = '/auth_login.html';
            }
            return response;
        });
    };

    // Intercept XMLHttpRequest
    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        this._url = url;
        return originalOpen.apply(this, [method, url, ...rest]);
    };
    
    XMLHttpRequest.prototype.send = function(...args) {
        const apiKey = getApiKey();
        
        if (apiKey && !this.getRequestHeader('Authorization')) {
            this.setRequestHeader('Authorization', `Bearer ${apiKey}`);
        }
        
        // Handle 401 responses
        this.addEventListener('load', function() {
            if (this.status === 401 && window.location.pathname !== '/auth_login.html') {
                localStorage.removeItem('comfyui_api_key');
                sessionStorage.removeItem('comfyui_api_key');
                window.location.href = '/auth_login.html';
            }
        });
        
        return originalSend.apply(this, args);
    };

    // Add logout function to window
    window.comfyuiLogout = function() {
        localStorage.removeItem('comfyui_api_key');
        sessionStorage.removeItem('comfyui_api_key');
        window.location.href = '/auth_login.html';
    };

    // Check authentication on page load (except for login page)
    if (window.location.pathname !== '/auth_login.html') {
        checkAuth();
    }

    console.log('[ComfyUI Auth] API key injection enabled');
})();
