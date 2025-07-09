/**
 * ComfyUI Mobile Interface - Utility Functions
 */

// Utility functions for mobile interface
class Utils {
    /**
     * Debounce function to limit rapid function calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function to limit function execution frequency
     * @param {Function} func - Function to throttle
     * @param {number} limit - Limit in milliseconds
     * @returns {Function} Throttled function
     */
    static throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Deep clone an object
     * @param {Object} obj - Object to clone
     * @returns {Object} Cloned object
     */
    static deepClone(obj) {
        if (obj === null || typeof obj !== "object") return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => Utils.deepClone(item));
        if (typeof obj === "object") {
            const clonedObj = {};
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    clonedObj[key] = Utils.deepClone(obj[key]);
                }
            }
            return clonedObj;
        }
    }

    /**
     * Generate a unique ID
     * @returns {string} Unique ID
     */
    static generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    /**
     * Format file size in human readable format
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted size
     */
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Format duration in human readable format
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    static formatDuration(seconds) {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    /**
     * Check if device has touch support
     * @returns {boolean} True if touch is supported
     */
    static isTouchDevice() {
        return ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
    }

    /**
     * Get viewport dimensions
     * @returns {Object} Viewport width and height
     */
    static getViewportSize() {
        return {
            width: Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0),
            height: Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)
        };
    }

    /**
     * Check if element is in viewport
     * @param {Element} element - Element to check
     * @returns {boolean} True if element is visible
     */
    static isElementInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    /**
     * Smooth scroll to element
     * @param {Element} element - Element to scroll to
     * @param {Object} options - Scroll options
     */
    static scrollToElement(element, options = {}) {
        const defaultOptions = {
            behavior: 'smooth',
            block: 'center',
            inline: 'nearest'
        };
        element.scrollIntoView({ ...defaultOptions, ...options });
    }

    /**
     * Add CSS class with animation support
     * @param {Element} element - Target element
     * @param {string} className - Class name to add
     * @param {number} duration - Animation duration
     */
    static addClassAnimated(element, className, duration = 300) {
        element.classList.add(className);
        if (duration > 0) {
            setTimeout(() => {
                element.classList.add('fade-in');
            }, 10);
        }
    }

    /**
     * Remove CSS class with animation support
     * @param {Element} element - Target element
     * @param {string} className - Class name to remove
     * @param {number} duration - Animation duration
     */
    static removeClassAnimated(element, className, duration = 300) {
        element.style.transition = `opacity ${duration}ms ease`;
        element.style.opacity = '0';
        setTimeout(() => {
            element.classList.remove(className);
            element.style.opacity = '';
            element.style.transition = '';
        }, duration);
    }

    /**
     * Show toast notification
     * @param {string} message - Message to display
     * @param {string} type - Toast type (success, error, warning, info)
     * @param {number} duration - Display duration in milliseconds
     */
    static showToast(message, type = 'info', duration = 3000) {
        // Remove existing toasts
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add toast styles if not already present
        if (!document.querySelector('#toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                .toast {
                    position: fixed;
                    top: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: var(--surface-color);
                    border: 1px solid var(--border-color);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-elevated);
                    padding: var(--spacing-md);
                    z-index: 9999;
                    animation: slideDown 0.3s ease;
                }
                .toast-content {
                    display: flex;
                    align-items: center;
                    gap: var(--spacing-sm);
                }
                .toast-success { border-left: 4px solid var(--success-color); }
                .toast-error { border-left: 4px solid var(--error-color); }
                .toast-warning { border-left: 4px solid var(--warning-color); }
                .toast-info { border-left: 4px solid var(--primary-color); }
                @keyframes slideDown {
                    from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
                    to { transform: translateX(-50%) translateY(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(toast);

        // Auto remove toast
        setTimeout(() => {
            toast.style.animation = 'slideUp 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    /**
     * Get appropriate icon for toast type
     * @param {string} type - Toast type
     * @returns {string} Icon name
     */
    static getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    /**
     * Local storage utility with error handling
     */
    static storage = {
        get(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.warn('LocalStorage get error:', error);
                return defaultValue;
            }
        },

        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.warn('LocalStorage set error:', error);
                return false;
            }
        },

        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.warn('LocalStorage remove error:', error);
                return false;
            }
        },

        clear() {
            try {
                localStorage.clear();
                return true;
            } catch (error) {
                console.warn('LocalStorage clear error:', error);
                return false;
            }
        }
    };

    /**
     * Event emitter for custom events
     */
    static events = {
        listeners: new Map(),

        on(event, callback) {
            if (!this.listeners.has(event)) {
                this.listeners.set(event, []);
            }
            this.listeners.get(event).push(callback);
        },

        off(event, callback) {
            if (!this.listeners.has(event)) return;
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        },

        emit(event, data) {
            if (!this.listeners.has(event)) return;
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('Event callback error:', error);
                }
            });
        }
    };

    /**
     * Touch gesture detection
     */
    static gesture = {
        /**
         * Add long press event listener
         * @param {Element} element - Target element
         * @param {Function} callback - Callback function
         * @param {number} duration - Long press duration in ms
         */
        longPress(element, callback, duration = 500) {
            let timer;
            let startTouch;

            const start = (e) => {
                startTouch = e.touches ? e.touches[0] : e;
                timer = setTimeout(() => {
                    callback(e);
                }, duration);
            };

            const end = () => {
                clearTimeout(timer);
            };

            const move = (e) => {
                const currentTouch = e.touches ? e.touches[0] : e;
                const deltaX = Math.abs(currentTouch.clientX - startTouch.clientX);
                const deltaY = Math.abs(currentTouch.clientY - startTouch.clientY);
                
                // Cancel long press if moved too much
                if (deltaX > 10 || deltaY > 10) {
                    clearTimeout(timer);
                }
            };

            element.addEventListener('touchstart', start);
            element.addEventListener('mousedown', start);
            element.addEventListener('touchend', end);
            element.addEventListener('mouseup', end);
            element.addEventListener('touchmove', move);
            element.addEventListener('mousemove', move);
            element.addEventListener('contextmenu', (e) => e.preventDefault());
        },

        /**
         * Add swipe event listener
         * @param {Element} element - Target element
         * @param {Function} callback - Callback function
         * @param {number} threshold - Swipe threshold in pixels
         */
        swipe(element, callback, threshold = 50) {
            let startTouch;
            let startTime;

            const start = (e) => {
                startTouch = e.touches ? e.touches[0] : e;
                startTime = Date.now();
            };

            const end = (e) => {
                if (!startTouch) return;

                const endTouch = e.changedTouches ? e.changedTouches[0] : e;
                const deltaX = endTouch.clientX - startTouch.clientX;
                const deltaY = endTouch.clientY - startTouch.clientY;
                const deltaTime = Date.now() - startTime;

                // Only consider fast swipes
                if (deltaTime > 300) return;

                const absDeltaX = Math.abs(deltaX);
                const absDeltaY = Math.abs(deltaY);

                if (Math.max(absDeltaX, absDeltaY) > threshold) {
                    let direction;
                    if (absDeltaX > absDeltaY) {
                        direction = deltaX > 0 ? 'right' : 'left';
                    } else {
                        direction = deltaY > 0 ? 'down' : 'up';
                    }
                    callback(direction, { deltaX, deltaY, deltaTime });
                }

                startTouch = null;
            };

            element.addEventListener('touchstart', start);
            element.addEventListener('mousedown', start);
            element.addEventListener('touchend', end);
            element.addEventListener('mouseup', end);
        }
    };
}

// Export for use in other files
window.Utils = Utils;