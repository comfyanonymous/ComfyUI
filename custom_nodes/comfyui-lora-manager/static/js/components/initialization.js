/**
 * Initialization Component
 * Manages the display of initialization progress and status
 */
import { appCore } from '../core.js';
import { getSessionItem, setSessionItem } from '../utils/storageHelpers.js';

class InitializationManager {
    constructor() {
        this.currentTipIndex = 0;
        this.tipInterval = null;
        this.websocket = null;
        this.progress = 0;
        this.processingStartTime = null;
        this.processedFilesCount = 0;
        this.totalFilesCount = 0;
        this.averageProcessingTime = null;
        this.pageType = null; // Added page type property
    }

    /**
     * Initialize the component
     */
    initialize() {
        // Initialize core application for theme and header functionality
        appCore.initialize().then(() => {
            console.log('Core application initialized for initialization component');
        });

        // Detect the current page type
        this.detectPageType();

        // Check session storage for saved progress
        this.restoreProgress();
        
        // Setup the tip carousel
        this.setupTipCarousel();
        
        // Connect to WebSocket for progress updates
        this.connectWebSocket();

        // Add event listeners for tip navigation
        this.setupTipNavigation();
        
        // Show first tip as active
        document.querySelector('.tip-item').classList.add('active');
    }

    /**
     * Detect the current page type
     */
    detectPageType() {
        // Get the current page type from URL or data attribute
        const path = window.location.pathname;
        if (path.includes('/checkpoints')) {
            this.pageType = 'checkpoints';
        } else if (path.includes('/loras')) {
            this.pageType = 'loras';
        } else {
            // Default to loras if can't determine
            this.pageType = 'loras';
        }
        console.log(`Initialization component detected page type: ${this.pageType}`);
    }

    /**
     * Get the storage key with page type prefix
     */
    getStorageKey(key) {
        return `${this.pageType}_${key}`;
    }

    /**
     * Restore progress from session storage if available
     */
    restoreProgress() {
        const savedProgress = getSessionItem(this.getStorageKey('initProgress'));
        if (savedProgress) {
            console.log(`Restoring ${this.pageType} progress from session storage:`, savedProgress);
            
            // Restore progress percentage
            if (savedProgress.progress !== undefined) {
                this.updateProgress(savedProgress.progress);
            }
            
            // Restore processed files count and total files
            if (savedProgress.processedFiles !== undefined) {
                this.processedFilesCount = savedProgress.processedFiles;
            }
            
            if (savedProgress.totalFiles !== undefined) {
                this.totalFilesCount = savedProgress.totalFiles;
            }
            
            // Restore processing time metrics if available
            if (savedProgress.averageProcessingTime !== undefined) {
                this.averageProcessingTime = savedProgress.averageProcessingTime;
                this.updateRemainingTime();
            }
            
            // Restore progress status message
            if (savedProgress.details) {
                this.updateStatusMessage(savedProgress.details);
            }
        }
    }

    /**
     * Connect to WebSocket for initialization progress updates
     */
    connectWebSocket() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            this.websocket = new WebSocket(`${wsProtocol}${window.location.host}/ws/init-progress`);
            
            this.websocket.onopen = () => {
                console.log('Connected to initialization progress WebSocket');
            };
            
            this.websocket.onmessage = (event) => {
                this.handleProgressUpdate(JSON.parse(event.data));
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                // Fall back to polling if WebSocket fails
                this.fallbackToPolling();
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket connection closed');
                // Check if we need to fall back to polling
                if (!this.pollingActive) {
                    this.fallbackToPolling();
                }
            };
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.fallbackToPolling();
        }
    }

    /**
     * Fall back to polling if WebSocket connection fails
     */
    fallbackToPolling() {
        this.pollingActive = true;
        this.pollProgress();
        
        // Set a simulated progress that moves forward slowly
        // This gives users feedback even if the backend isn't providing updates
        let simulatedProgress = this.progress || 0;
        const simulateInterval = setInterval(() => {
            simulatedProgress += 0.5;
            if (simulatedProgress > 95) {
                clearInterval(simulateInterval);
                return;
            }
            
            // Only use simulated progress if we haven't received a real update
            if (this.progress < simulatedProgress) {
                this.updateProgress(simulatedProgress);
            }
        }, 1000);
    }

    /**
     * Poll for progress updates from the server
     */
    pollProgress() {
        const checkProgress = () => {
            fetch('/api/init-status')
                .then(response => response.json())
                .then(data => {
                    this.handleProgressUpdate(data);
                    
                    // If initialization is complete, stop polling
                    if (data.status !== 'complete') {
                        setTimeout(checkProgress, 2000);
                    } else {
                        window.location.reload();
                    }
                })
                .catch(error => {
                    console.error('Error polling for progress:', error);
                    setTimeout(checkProgress, 3000); // Try again after a longer delay
                });
        };
        
        checkProgress();
    }

    /**
     * Handle progress updates from WebSocket or polling
     */
    handleProgressUpdate(data) {
        if (!data) return;
        
        // Check if this update is for our page type
        if (data.pageType && data.pageType !== this.pageType) {
            console.log(`Ignoring update for ${data.pageType}, we're on ${this.pageType}`);
            return;
        }

        // If no pageType is specified in the data but we have scanner_type, map it to pageType
        if (!data.pageType && data.scanner_type) {
            const scannerTypeToPageType = {
                'lora': 'loras',
                'checkpoint': 'checkpoints'
            };
            
            if (scannerTypeToPageType[data.scanner_type] !== this.pageType) {
                console.log(`Ignoring update for ${data.scanner_type}, we're on ${this.pageType}`);
                return;
            }
        }
        
        // Save progress data to session storage
        setSessionItem(this.getStorageKey('initProgress'), {
            ...data,
            averageProcessingTime: this.averageProcessingTime,
            processedFiles: this.processedFilesCount,
            totalFiles: this.totalFilesCount
        });
        
        // Update progress percentage
        if (data.progress !== undefined) {
            this.updateProgress(data.progress);
        }
        
        // Update stage-specific details
        if (data.details) {
            this.updateStatusMessage(data.details);
        }
        
        // Track files count for time estimation
        if (data.stage === 'count_models' && data.details) {
            const match = data.details.match(/Found (\d+)/);
            if (match && match[1]) {
                this.totalFilesCount = parseInt(match[1]);
            }
        }
        
        // Track processed files for time estimation
        if (data.stage === 'process_models' && data.details) {
            const match = data.details.match(/Processing .* files: (\d+)\/(\d+)/);
            if (match && match[1] && match[2]) {
                const currentCount = parseInt(match[1]);
                const totalCount = parseInt(match[2]);
                
                // Make sure we have valid total count
                if (totalCount > 0 && this.totalFilesCount === 0) {
                    this.totalFilesCount = totalCount;
                }
                
                // Start tracking processing time once we've processed some files
                if (currentCount > 0 && !this.processingStartTime && this.processedFilesCount === 0) {
                    this.processingStartTime = Date.now();
                }
                
                // Calculate average processing time based on elapsed time and files processed
                if (this.processingStartTime && currentCount > this.processedFilesCount) {
                    const newFiles = currentCount - this.processedFilesCount;
                    const elapsedTime = Date.now() - this.processingStartTime;
                    const timePerFile = elapsedTime / currentCount; // ms per file
                    
                    // Update moving average
                    if (!this.averageProcessingTime) {
                        this.averageProcessingTime = timePerFile;
                    } else {
                        // Simple exponential moving average
                        this.averageProcessingTime = this.averageProcessingTime * 0.7 + timePerFile * 0.3;
                    }
                    
                    // Update remaining time estimate
                    this.updateRemainingTime();
                }
                
                this.processedFilesCount = currentCount;
            }
        }
        
        // If initialization is complete, reload the page
        if (data.status === 'complete') {
            this.showCompletionMessage();
            
            // Remove session storage data since we're done
            setSessionItem(this.getStorageKey('initProgress'), null);
            
            // Give the user a moment to see the completion message
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        }
    }

    /**
     * Update the remaining time display based on current progress
     */
    updateRemainingTime() {
        if (!this.averageProcessingTime || !this.totalFilesCount || this.totalFilesCount <= 0) {
            document.getElementById('remainingTime').textContent = 'Estimating...';
            return;
        }
        
        const remainingFiles = this.totalFilesCount - this.processedFilesCount;
        const remainingTimeMs = remainingFiles * this.averageProcessingTime;
        
        if (remainingTimeMs <= 0) {
            document.getElementById('remainingTime').textContent = 'Almost done...';
            return;
        }
        
        // Format the time for display
        let formattedTime;
        if (remainingTimeMs < 60000) {
            // Less than a minute
            formattedTime = 'Less than a minute';
        } else if (remainingTimeMs < 3600000) {
            // Less than an hour
            const minutes = Math.round(remainingTimeMs / 60000);
            formattedTime = `~${minutes} minute${minutes !== 1 ? 's' : ''}`;
        } else {
            // Hours and minutes
            const hours = Math.floor(remainingTimeMs / 3600000);
            const minutes = Math.round((remainingTimeMs % 3600000) / 60000);
            formattedTime = `~${hours} hour${hours !== 1 ? 's' : ''} ${minutes} minute${minutes !== 1 ? 's' : ''}`;
        }
        
        document.getElementById('remainingTime').textContent = formattedTime + ' remaining';
    }

    /**
     * Update status message
     */
    updateStatusMessage(message) {
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.textContent = message;
        }
    }

    /**
     * Update the progress bar and percentage
     */
    updateProgress(progress) {
        this.progress = progress;
        const progressBar = document.getElementById('initProgressBar');
        const progressPercentage = document.getElementById('progressPercentage');
        
        if (progressBar && progressPercentage) {
            progressBar.style.width = `${progress}%`;
            progressPercentage.textContent = `${Math.round(progress)}%`;
        }
    }

    /**
     * Setup the tip carousel to rotate through tips
     */
    setupTipCarousel() {
        const tipItems = document.querySelectorAll('.tip-item');
        if (tipItems.length === 0) return;
        
        // Show the first tip
        tipItems[0].classList.add('active');
        document.querySelector('.tip-dot').classList.add('active');
        
        // Set up automatic rotation
        this.tipInterval = setInterval(() => {
            this.showNextTip();
        }, 8000); // Change tip every 8 seconds
    }

    /**
     * Setup tip navigation dots
     */
    setupTipNavigation() {
        const tipDots = document.querySelectorAll('.tip-dot');
        
        tipDots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                this.showTipByIndex(index);
            });
        });
    }

    /**
     * Show the next tip in the carousel
     */
    showNextTip() {
        const tipItems = document.querySelectorAll('.tip-item');
        const tipDots = document.querySelectorAll('.tip-dot');
        
        if (tipItems.length === 0) return;
        
        // Hide current tip
        tipItems[this.currentTipIndex].classList.remove('active');
        tipDots[this.currentTipIndex].classList.remove('active');
        
        // Calculate next index
        this.currentTipIndex = (this.currentTipIndex + 1) % tipItems.length;
        
        // Show next tip
        tipItems[this.currentTipIndex].classList.add('active');
        tipDots[this.currentTipIndex].classList.add('active');
    }

    /**
     * Show a specific tip by index
     */
    showTipByIndex(index) {
        const tipItems = document.querySelectorAll('.tip-item');
        const tipDots = document.querySelectorAll('.tip-dot');
        
        if (index >= tipItems.length || index < 0) return;
        
        // Hide current tip
        tipItems[this.currentTipIndex].classList.remove('active');
        tipDots[this.currentTipIndex].classList.remove('active');
        
        // Update index and show selected tip
        this.currentTipIndex = index;
        
        // Show selected tip
        tipItems[this.currentTipIndex].classList.add('active');
        tipDots[this.currentTipIndex].classList.add('active');
        
        // Reset interval to prevent quick tip change
        if (this.tipInterval) {
            clearInterval(this.tipInterval);
            this.tipInterval = setInterval(() => {
                this.showNextTip();
            }, 8000);
        }
    }

    /**
     * Show completion message
     */
    showCompletionMessage() {
        // Update progress to 100%
        this.updateProgress(100);
        
        // Update status message
        this.updateStatusMessage('Initialization complete!');
        
        // Update title and subtitle
        const initTitle = document.getElementById('initTitle');
        const initSubtitle = document.getElementById('initSubtitle');
        const remainingTime = document.getElementById('remainingTime');
        
        if (initTitle) {
            initTitle.textContent = 'Initialization Complete';
        }
        
        if (initSubtitle) {
            initSubtitle.textContent = 'Reloading page...';
        }
        
        if (remainingTime) {
            remainingTime.textContent = 'Done!';
        }
    }

    /**
     * Clean up resources when the component is destroyed
     */
    cleanup() {
        if (this.tipInterval) {
            clearInterval(this.tipInterval);
        }
        
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.close();
        }
    }
}

// Create and export the initialization manager
export const initManager = new InitializationManager();

// Initialize the component when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if we're in initialization mode
    const initContainer = document.getElementById('initializationContainer');
    if (initContainer) {
        initManager.initialize();
    }
});

// Clean up when the page is unloaded
window.addEventListener('beforeunload', () => {
    initManager.cleanup();
});