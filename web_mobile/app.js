/**
 * ComfyUI Mobile Interface - Main Application
 * Initializes and manages the mobile interface
 */

class MobileApp {
    constructor() {
        this.apiClient = null;
        this.mobileInterface = null;
        this.isInitialized = false;
        this.startTime = Date.now();
    }

    /**
     * Initialize the mobile application
     */
    async initialize() {
        if (this.isInitialized) return;

        console.log('Initializing ComfyUI Mobile Interface...');
        
        try {
            // Show loading state
            this.showLoadingState();

            // Initialize API client
            this.apiClient = new ComfyUIAPIClient();
            
            // Wait for initial connection
            await this.waitForConnection();

            // Initialize mobile interface
            this.mobileInterface = new MobileInterface(this.apiClient);

            // Setup global error handling
            this.setupGlobalErrorHandling();

            // Setup performance monitoring
            this.setupPerformanceMonitoring();

            // Setup viewport handling
            this.setupViewportHandling();

            // Initialize service worker if available
            this.initializeServiceWorker();

            this.isInitialized = true;
            console.log('ComfyUI Mobile Interface initialized successfully');
            
            // Hide loading state
            this.hideLoadingState();

            // Log initialization time
            const initTime = Date.now() - this.startTime;
            console.log(`Mobile interface initialized in ${initTime}ms`);

        } catch (error) {
            console.error('Failed to initialize mobile interface:', error);
            this.showErrorState(error);
        }
    }

    /**
     * Wait for API connection
     * @returns {Promise} Connection promise
     */
    waitForConnection() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Connection timeout'));
            }, 10000);

            const checkConnection = () => {
                if (this.apiClient.isWebSocketConnected()) {
                    clearTimeout(timeout);
                    resolve();
                } else {
                    // Listen for connection event
                    this.apiClient.on('connected', () => {
                        clearTimeout(timeout);
                        resolve();
                    });

                    this.apiClient.on('connection_error', (error) => {
                        clearTimeout(timeout);
                        reject(error);
                    });
                }
            };

            // Check immediately and then wait for events
            checkConnection();
        });
    }

    /**
     * Show loading state
     */
    showLoadingState() {
        document.body.classList.add('loading');
        
        // Add loading overlay if not exists
        if (!document.getElementById('loadingOverlay')) {
            const overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                    <h3>Loading ComfyUI Mobile</h3>
                    <p>Connecting to server...</p>
                </div>
            `;
            document.body.appendChild(overlay);

            // Add loading styles
            const style = document.createElement('style');
            style.textContent = `
                #loadingOverlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: var(--background-color);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                }
                .loading-content {
                    text-align: center;
                    padding: var(--spacing-xl);
                }
                .loading-spinner {
                    font-size: 2rem;
                    color: var(--primary-color);
                    margin-bottom: var(--spacing-lg);
                }
                .loading-content h3 {
                    margin-bottom: var(--spacing-sm);
                    color: var(--text-primary);
                }
                .loading-content p {
                    color: var(--text-secondary);
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        document.body.classList.remove('loading');
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.remove(), 300);
        }
    }

    /**
     * Show error state
     * @param {Error} error - Error object
     */
    showErrorState(error) {
        const errorOverlay = document.createElement('div');
        errorOverlay.id = 'errorOverlay';
        errorOverlay.innerHTML = `
            <div class="error-content">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h3>Connection Error</h3>
                <p>${error.message}</p>
                <button id="retryBtn" class="btn-primary">
                    <i class="fas fa-redo"></i>
                    Retry
                </button>
                <button id="desktopBtn" class="btn-secondary">
                    <i class="fas fa-desktop"></i>
                    Switch to Desktop
                </button>
            </div>
        `;

        // Add error styles
        const style = document.createElement('style');
        style.textContent = `
            #errorOverlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: var(--background-color);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
            }
            .error-content {
                text-align: center;
                padding: var(--spacing-xl);
                max-width: 400px;
            }
            .error-icon {
                font-size: 3rem;
                color: var(--error-color);
                margin-bottom: var(--spacing-lg);
            }
            .error-content h3 {
                margin-bottom: var(--spacing-sm);
                color: var(--text-primary);
            }
            .error-content p {
                color: var(--text-secondary);
                margin-bottom: var(--spacing-lg);
            }
            .error-content button {
                margin: var(--spacing-sm);
            }
        `;
        document.head.appendChild(style);

        // Replace loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }

        document.body.appendChild(errorOverlay);

        // Add event listeners
        document.getElementById('retryBtn').addEventListener('click', () => {
            errorOverlay.remove();
            this.initialize();
        });

        document.getElementById('desktopBtn').addEventListener('click', () => {
            window.location.href = '/';
        });
    }

    /**
     * Setup global error handling
     */
    setupGlobalErrorHandling() {
        // Handle uncaught errors
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            Utils.showToast('An error occurred: ' + event.error.message, 'error');
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            Utils.showToast('An error occurred: ' + event.reason, 'error');
        });

        // Handle API errors
        this.apiClient.on('connection_error', (error) => {
            console.error('API connection error:', error);
            Utils.showToast('Connection error: ' + error.message, 'error');
        });
    }

    /**
     * Setup performance monitoring
     */
    setupPerformanceMonitoring() {
        // Monitor memory usage
        if ('memory' in performance) {
            setInterval(() => {
                const memory = performance.memory;
                if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.9) {
                    console.warn('High memory usage detected');
                }
            }, 30000); // Check every 30 seconds
        }

        // Monitor long tasks
        if ('PerformanceObserver' in window) {
            try {
                const observer = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    entries.forEach((entry) => {
                        if (entry.duration > 50) { // Tasks longer than 50ms
                            console.warn('Long task detected:', entry.duration + 'ms');
                        }
                    });
                });
                observer.observe({ entryTypes: ['longtask'] });
            } catch (e) {
                console.log('Long task monitoring not supported');
            }
        }
    }

    /**
     * Setup viewport handling
     */
    setupViewportHandling() {
        // Handle orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                // Refresh layout after orientation change
                if (this.mobileInterface) {
                    this.mobileInterface.refreshWorkflow();
                }
            }, 100);
        });

        // Handle viewport changes (e.g., virtual keyboard)
        const viewportHandler = Utils.debounce(() => {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
        }, 100);

        window.addEventListener('resize', viewportHandler);
        window.addEventListener('orientationchange', viewportHandler);
        
        // Initial viewport setup
        viewportHandler();

        // Handle visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // App is hidden
                console.log('App hidden');
            } else {
                // App is visible
                console.log('App visible');
                if (this.mobileInterface) {
                    this.mobileInterface.refreshWorkflow();
                }
            }
        });
    }

    /**
     * Initialize service worker
     */
    async initializeServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/sw.js');
                console.log('Service worker registered:', registration);
            } catch (error) {
                console.log('Service worker registration failed:', error);
            }
        }
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S to save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                if (this.mobileInterface) {
                    this.mobileInterface.showSaveWorkflowDialog();
                }
            }

            // Ctrl/Cmd + O to open
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                if (this.mobileInterface) {
                    this.mobileInterface.showLoadWorkflowDialog();
                }
            }

            // Escape to cancel connection
            if (e.key === 'Escape') {
                if (this.mobileInterface && this.mobileInterface.connectionState.active) {
                    this.mobileInterface.cancelConnection();
                }
            }

            // Space to queue prompt
            if (e.key === ' ' && !e.target.matches('input, textarea')) {
                e.preventDefault();
                if (this.mobileInterface) {
                    this.mobileInterface.queueCurrentWorkflow();
                }
            }
        });
    }

    /**
     * Get app statistics
     * @returns {Object} App statistics
     */
    getStatistics() {
        return {
            initialized: this.isInitialized,
            initTime: Date.now() - this.startTime,
            connected: this.apiClient ? this.apiClient.isWebSocketConnected() : false,
            nodesCount: this.mobileInterface ? this.mobileInterface.linearizedNodes.length : 0,
            memory: performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
            } : null,
            viewport: Utils.getViewportSize(),
            touchDevice: Utils.isTouchDevice()
        };
    }

    /**
     * Cleanup application
     */
    cleanup() {
        if (this.apiClient) {
            this.apiClient.disconnect();
        }
        
        // Remove event listeners
        window.removeEventListener('error', this.handleError);
        window.removeEventListener('unhandledrejection', this.handleRejection);
        
        this.isInitialized = false;
        console.log('Mobile application cleaned up');
    }
}

// Initialize the mobile app when DOM is ready
let mobileApp;
let mobileInterface;

document.addEventListener('DOMContentLoaded', async () => {
    // Check if this is a mobile device or small screen
    const isMobile = Utils.isTouchDevice() || window.innerWidth < 768;
    
    if (isMobile) {
        console.log('Mobile device detected, initializing mobile interface...');
    } else {
        console.log('Desktop device detected, mobile interface available');
    }

    // Initialize the app
    mobileApp = new MobileApp();
    await mobileApp.initialize();
    
    // Make mobile interface globally available for debugging
    mobileInterface = mobileApp.mobileInterface;
    
    // Expose to global scope for debugging
    window.mobileApp = mobileApp;
    window.mobileInterface = mobileInterface;
    
    // Setup keyboard shortcuts
    mobileApp.setupKeyboardShortcuts();
    
    // Log app statistics
    console.log('App Statistics:', mobileApp.getStatistics());
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (mobileApp) {
        mobileApp.cleanup();
    }
});

// Handle app install prompt (PWA)
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    
    // Show install button/banner
    Utils.showToast('Install ComfyUI Mobile for offline access', 'info', 5000);
});

// Handle successful app installation
window.addEventListener('appinstalled', (evt) => {
    console.log('App installed successfully');
    Utils.showToast('ComfyUI Mobile installed successfully', 'success');
});

// Export for use in other contexts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MobileApp, Utils, GraphLinearization, ComfyUIAPIClient, MobileInterface };
}