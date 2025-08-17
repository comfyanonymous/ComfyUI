import { getStorageItem, setStorageItem } from '../utils/storageHelpers.js';

/**
 * Manages help modal functionality and tutorial update notifications
 */
export class HelpManager {
    constructor() {
        this.lastViewedTimestamp = getStorageItem('help_last_viewed', 0);
        this.latestContentTimestamp = new Date('2025-07-09').getTime(); // Will be updated from server or config
        this.isInitialized = false;
        
        // Default latest content data - could be fetched from server
        this.latestVideoData = {
            timestamp: new Date('2024-06-09').getTime(), // Default timestamp
            walkthrough: {
                id: 'hvKw31YpE-U',
                title: 'Getting Started with LoRA Manager'
            },
            playlistUpdated: true
        };
    }

    /**
     * Initialize the help manager
     */
    initialize() {
        if (this.isInitialized) return;
        
        console.log('HelpManager: Initializing...');
        
        // Set up event handlers
        this.setupEventListeners();
        
        // Check if we need to show the badge
        this.updateHelpBadge();
        
        // Fetch latest video data (could be implemented to fetch from remote source)
        this.fetchLatestVideoData();
        
        this.isInitialized = true;
        return this;
    }
    
    /**
     * Set up event listeners for help modal
     */
    setupEventListeners() {
        // Help toggle button
        const helpToggleBtn = document.getElementById('helpToggleBtn');
        if (helpToggleBtn) {
            helpToggleBtn.addEventListener('click', () => this.openHelpModal());
        }
        
        // Help modal tab functionality
        const tabButtons = document.querySelectorAll('.help-tabs .tab-btn');
        tabButtons.forEach(button => {
            button.addEventListener('click', (event) => {
                // Remove active class from all buttons and panes
                document.querySelectorAll('.help-tabs .tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelectorAll('.help-content .tab-pane').forEach(pane => {
                    pane.classList.remove('active');
                });
                
                // Add active class to clicked button
                event.currentTarget.classList.add('active');
                
                // Show corresponding tab content
                const tabId = event.currentTarget.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    /**
     * Open the help modal
     */
    openHelpModal() {
        // Use modalManager to open the help modal
        if (window.modalManager) {
            window.modalManager.toggleModal('helpModal');
            
            // Add visual indicator to Documentation tab if there's new content
            this.updateDocumentationTabIndicator();
            
            // Update the last viewed timestamp
            this.markContentAsViewed();
            
            // Hide the badge
            this.hideHelpBadge();
        }
    }

    /**
     * Add visual indicator to Documentation tab for new content
     */
    updateDocumentationTabIndicator() {
        const docTab = document.querySelector('.tab-btn[data-tab="documentation"]');
        if (docTab && this.hasNewContent()) {
            docTab.classList.add('has-new-content');
        }
    }
    
    /**
     * Mark content as viewed by saving current timestamp
     */
    markContentAsViewed() {
        this.lastViewedTimestamp = Date.now();
        setStorageItem('help_last_viewed', this.lastViewedTimestamp);
    }
    
    /**
     * Fetch latest video data (could be implemented to actually fetch from a remote source)
     */
    fetchLatestVideoData() {
        // In a real implementation, you'd fetch this from your server
        // For now, we'll just use the hardcoded data from constructor
        
        // Update the timestamp with the latest data
        this.latestContentTimestamp = Math.max(this.latestContentTimestamp, this.latestVideoData.timestamp);
        
        // Check again if we need to show the badge with this new data
        this.updateHelpBadge();
    }
    
    /**
     * Update help badge visibility based on timestamps
     */
    updateHelpBadge() {
        if (this.hasNewContent()) {
            this.showHelpBadge();
        } else {
            this.hideHelpBadge();
        }
    }
    
    /**
     * Check if there's new content the user hasn't seen
     */
    hasNewContent() {
        // If user has never viewed the help, or the content is newer than last viewed
        return this.lastViewedTimestamp === 0 || this.latestContentTimestamp > this.lastViewedTimestamp;
    }
    
    /**
     * Show the help badge
     */
    showHelpBadge() {
        const helpBadge = document.querySelector('#helpToggleBtn .update-badge');
        if (helpBadge) {
            helpBadge.classList.add('visible');
        }
    }
    
    /**
     * Hide the help badge
     */
    hideHelpBadge() {
        const helpBadge = document.querySelector('#helpToggleBtn .update-badge');
        if (helpBadge) {
            helpBadge.classList.remove('visible');
        }
    }
}

// Create singleton instance
export const helpManager = new HelpManager();