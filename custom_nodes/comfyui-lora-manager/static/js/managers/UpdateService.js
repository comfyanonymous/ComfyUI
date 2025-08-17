import { modalManager } from './ModalManager.js';
import { 
    getStorageItem, 
    setStorageItem, 
    getStoredVersionInfo, 
    setStoredVersionInfo,
    isVersionMatch,
    resetDismissedBanner
} from '../utils/storageHelpers.js';
import { bannerService } from './BannerService.js';

export class UpdateService {
    constructor() {
        this.updateCheckInterval = 60 * 60 * 1000; // 1 hour
        this.currentVersion = "v0.0.0";  // Initialize with default values
        this.latestVersion = "v0.0.0";   // Initialize with default values
        this.updateInfo = null;
        this.updateAvailable = false;
        this.gitInfo = {
            short_hash: "unknown",
            branch: "unknown",
            commit_date: "unknown"
        };
        this.updateNotificationsEnabled = getStorageItem('show_update_notifications', true);
        this.lastCheckTime = parseInt(getStorageItem('last_update_check') || '0');
        this.isUpdating = false;
        this.nightlyMode = getStorageItem('nightly_updates', false);
        this.currentVersionInfo = null;
        this.versionMismatch = false;
    }

    initialize() {
        // Register event listener for update notification toggle
        const updateCheckbox = document.getElementById('updateNotifications');
        if (updateCheckbox) {
            updateCheckbox.checked = this.updateNotificationsEnabled;
            updateCheckbox.addEventListener('change', (e) => {
                this.updateNotificationsEnabled = e.target.checked;
                setStorageItem('show_update_notifications', e.target.checked);
                this.updateBadgeVisibility();
            });
        }

        const updateBtn = document.getElementById('updateBtn');
        if (updateBtn) {
            updateBtn.addEventListener('click', () => this.performUpdate());
        }
        
        // Register event listener for nightly update toggle
        const nightlyCheckbox = document.getElementById('nightlyUpdateToggle');
        if (nightlyCheckbox) {
            nightlyCheckbox.checked = this.nightlyMode;
            nightlyCheckbox.addEventListener('change', (e) => {
                this.nightlyMode = e.target.checked;
                setStorageItem('nightly_updates', e.target.checked);
                this.updateNightlyWarning();
                this.updateModalContent();
                // Re-check for updates when switching channels
                this.manualCheckForUpdates();
            });
            this.updateNightlyWarning();
        }
        
        // Perform update check if needed
        this.checkForUpdates().then(() => {
            // Ensure badges are updated after checking
            this.updateBadgeVisibility();
        });

        // Immediately update modal content with current values (even if from default)
        this.updateModalContent();
        
        // Check version info for mismatch after loading basic info
        this.checkVersionInfo();
    }
    
    updateNightlyWarning() {
        const warning = document.getElementById('nightlyWarning');
        if (warning) {
            warning.style.display = this.nightlyMode ? 'flex' : 'none';
        }
    }
    
    async checkForUpdates() {
        // Check if we should perform an update check
        const now = Date.now();
        const forceCheck = this.lastCheckTime === 0;
        
        if (!forceCheck && now - this.lastCheckTime < this.updateCheckInterval) {
            // If we already have update info, just update the UI
            if (this.updateAvailable) {
                this.updateBadgeVisibility();
            }
            return;
        }
        
        try {
            // Call backend API to check for updates with nightly flag
            const response = await fetch(`/api/check-updates?nightly=${this.nightlyMode}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentVersion = data.current_version || "v0.0.0";
                this.latestVersion = data.latest_version || "v0.0.0";
                this.updateInfo = data;
                this.gitInfo = data.git_info || this.gitInfo;
                
                // Explicitly set update availability based on version comparison
                this.updateAvailable = this.isNewerVersion(this.latestVersion, this.currentVersion);
                
                // Update last check time
                this.lastCheckTime = now;
                setStorageItem('last_update_check', now.toString());
                
                // Update UI
                this.updateBadgeVisibility();
                this.updateModalContent();

                console.log("Update check complete:", {
                    currentVersion: this.currentVersion,
                    latestVersion: this.latestVersion,
                    updateAvailable: this.updateAvailable,
                    gitInfo: this.gitInfo
                });
            }
        } catch (error) {
            console.error('Failed to check for updates:', error);
        }
    }
    
    // Helper method to compare version strings
    isNewerVersion(latestVersion, currentVersion) {
        if (!latestVersion || !currentVersion) return false;
        
        // Remove 'v' prefix if present
        const latest = latestVersion.replace(/^v/, '');
        const current = currentVersion.replace(/^v/, '');
        
        // Split version strings into components
        const latestParts = latest.split(/[-\.]/);
        const currentParts = current.split(/[-\.]/);
        
        // Compare major, minor, patch versions
        for (let i = 0; i < 3; i++) {
            const latestNum = parseInt(latestParts[i] || '0', 10);
            const currentNum = parseInt(currentParts[i] || '0', 10);
            
            if (latestNum > currentNum) return true;
            if (latestNum < currentNum) return false;
        }
        
        // If numeric versions are the same, check for beta/alpha status
        const latestIsBeta = latest.includes('beta') || latest.includes('alpha');
        const currentIsBeta = current.includes('beta') || current.includes('alpha');
        
        // Release version is newer than beta/alpha
        if (!latestIsBeta && currentIsBeta) return true;
        
        return false;
    }
    
    updateBadgeVisibility() {
        const updateToggle = document.querySelector('.update-toggle');
        const updateBadge = document.querySelector('.update-toggle .update-badge');
        
        if (updateToggle) {
            updateToggle.title = this.updateNotificationsEnabled && this.updateAvailable 
                ? "Update Available" 
                : "Check Updates";
        }
        
        // Force updating badges visibility based on current state
        const shouldShow = this.updateNotificationsEnabled && this.updateAvailable;
        
        if (updateBadge) {
            updateBadge.classList.toggle('visible', shouldShow);
            console.log("Update badge visibility:", shouldShow ? "visible" : "hidden");
        }
    }
    
    updateModalContent() {
        const modal = document.getElementById('updateModal');
        if (!modal) return;
        
        // Update title based on update availability
        const headerTitle = modal.querySelector('.update-header h2');
        if (headerTitle) {
            headerTitle.textContent = this.updateAvailable ? "Update Available" : "Check for Updates";
        }
        
        // Always update version information, even if updateInfo is null
        const currentVersionEl = modal.querySelector('.current-version .version-number');
        const newVersionEl = modal.querySelector('.new-version .version-number');
        
        if (currentVersionEl) currentVersionEl.textContent = this.currentVersion;
        
        if (newVersionEl) {
            newVersionEl.textContent = this.latestVersion;
        }
        
        // Update update button state
        const updateBtn = modal.querySelector('#updateBtn');
        if (updateBtn) {
            updateBtn.classList.toggle('disabled', !this.updateAvailable || this.isUpdating);
            updateBtn.disabled = !this.updateAvailable || this.isUpdating;
        }
        
        // Update git info
        const gitInfoEl = modal.querySelector('.git-info');
        if (gitInfoEl && this.gitInfo) {
            if (this.gitInfo.short_hash !== 'unknown') {
                let gitText = `Commit: ${this.gitInfo.short_hash}`;
                if (this.gitInfo.commit_date !== 'unknown') {
                    gitText += ` - Date: ${this.gitInfo.commit_date}`;
                }
                gitInfoEl.textContent = gitText;
                gitInfoEl.style.display = 'block';
            } else {
                gitInfoEl.style.display = 'none';
            }
        }
        
        // Update changelog content if available
        if (this.updateInfo && this.updateInfo.changelog) {
            const changelogContent = modal.querySelector('.changelog-content');
            if (changelogContent) {
                changelogContent.innerHTML = ''; // Clear existing content
                
                // Create changelog item
                const changelogItem = document.createElement('div');
                changelogItem.className = 'changelog-item';
                
                const versionHeader = document.createElement('h4');
                versionHeader.textContent = `Version ${this.latestVersion}`;
                changelogItem.appendChild(versionHeader);
                
                // Create changelog list
                const changelogList = document.createElement('ul');
                
                if (this.updateInfo.changelog && this.updateInfo.changelog.length > 0) {
                    this.updateInfo.changelog.forEach(item => {
                        const listItem = document.createElement('li');
                        // Parse markdown in changelog items
                        listItem.innerHTML = this.parseMarkdown(item);
                        changelogList.appendChild(listItem);
                    });
                } else {
                    // If no changelog items available
                    const listItem = document.createElement('li');
                    listItem.textContent = "No detailed changelog available. Check GitHub for more information.";
                    changelogList.appendChild(listItem);
                }
                
                changelogItem.appendChild(changelogList);
                changelogContent.appendChild(changelogItem);
            }
        }
        
        // Update GitHub link to point to the specific release if available
        const githubLink = modal.querySelector('.update-link');
        if (githubLink && this.latestVersion) {
            const versionTag = this.latestVersion.replace(/^v/, '');
            githubLink.href = `https://github.com/willmiao/ComfyUI-Lora-Manager/releases/tag/v${versionTag}`;
        }
    }
    
    async performUpdate() {
        if (!this.updateAvailable || this.isUpdating) {
            return;
        }
        
        try {
            this.isUpdating = true;
            this.updateUpdateUI('updating', 'Updating...');
            this.showUpdateProgress(true);
            
            // Update progress
            this.updateProgress(10, 'Preparing update...');
            
            const response = await fetch('/api/perform-update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    nightly: this.nightlyMode
                })
            });
            
            this.updateProgress(50, 'Installing update...');
            
            const data = await response.json();
            
            if (data.success) {
                this.updateProgress(100, 'Update completed successfully!');
                this.updateUpdateUI('success', 'Updated!');
                
                // Show success message and suggest restart
                setTimeout(() => {
                    this.showUpdateCompleteMessage(data.new_version);
                }, 1000);
                
            } else {
                throw new Error(data.error || 'Update failed');
            }
            
        } catch (error) {
            console.error('Update failed:', error);
            this.updateUpdateUI('error', 'Update Failed');
            this.updateProgress(0, `Update failed: ${error.message}`);
            
            // Hide progress after error
            setTimeout(() => {
                this.showUpdateProgress(false);
            }, 3000);
        } finally {
            this.isUpdating = false;
        }
    }
    
    updateUpdateUI(state, text) {
        const updateBtn = document.getElementById('updateBtn');
        const updateBtnText = document.getElementById('updateBtnText');
        
        if (updateBtn && updateBtnText) {
            // Remove existing state classes
            updateBtn.classList.remove('updating', 'success', 'error', 'disabled');
            
            // Add new state class
            if (state !== 'normal') {
                updateBtn.classList.add(state);
            }
            
            // Update button text
            updateBtnText.textContent = text;
            
            // Update disabled state
            updateBtn.disabled = (state === 'updating' || state === 'disabled');
        }
    }
    
    showUpdateProgress(show) {
        const progressContainer = document.getElementById('updateProgress');
        if (progressContainer) {
            progressContainer.style.display = show ? 'block' : 'none';
        }
    }
    
    updateProgress(percentage, text) {
        const progressFill = document.getElementById('updateProgressFill');
        const progressText = document.getElementById('updateProgressText');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = text;
        }
    }
    
    showUpdateCompleteMessage(newVersion) {
        const modal = document.getElementById('updateModal');
        if (!modal) return;
        
        // Update the modal content to show completion
        const progressText = document.getElementById('updateProgressText');
        if (progressText) {
            progressText.innerHTML = `
                <div style="text-align: center; color: var(--lora-success);">
                    <i class="fas fa-check-circle" style="margin-right: 8px;"></i>
                    Successfully updated to ${newVersion}!
                    <br><br>
                    <div style="opacity: 0.95; color: var(--lora-error); font-size: 1em;">
                        Please restart ComfyUI or LoRA Manager to apply update.<br>
                        Make sure to reload your browser for both LoRA Manager and ComfyUI.
                    </div>
                </div>
            `;
        }
        
        // Update current version display
        this.currentVersion = newVersion;
        this.updateAvailable = false;
        
        // Refresh the modal content
        // setTimeout(() => {
        //     this.updateModalContent();
        //     this.showUpdateProgress(false);
        // }, 2000);
    }
    
    // Simple markdown parser for changelog items
    parseMarkdown(text) {
        if (!text) return '';
        
        // Handle bold text (**text**)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Handle italic text (*text*)
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Handle inline code (`code`)
        text = text.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Handle links [text](url)
        text = text.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        return text;
    }
    
    toggleUpdateModal() {
        const updateModal = modalManager.getModal('updateModal');
        
        // If modal is already open, just close it
        if (updateModal && updateModal.isOpen) {
            modalManager.closeModal('updateModal');
            return;
        }
        
        // Update the modal content immediately with current data
        this.updateModalContent();
        
        // Show the modal with current data
        modalManager.showModal('updateModal');
        
        // Then check for updates in the background
        this.manualCheckForUpdates().then(() => {
            // Update the modal content again after the check completes
            this.updateModalContent();
        });
    }
    
    async manualCheckForUpdates() {
        this.lastCheckTime = 0; // Reset last check time to force check
        await this.checkForUpdates();
        // Ensure badge visibility is updated after manual check
        this.updateBadgeVisibility();
    }
    
    async checkVersionInfo() {
        try {
            // Call API to get current version info
            const response = await fetch('/api/version-info');
            const data = await response.json();
            
            if (data.success) {
                this.currentVersionInfo = data.version;
                
                // Check if version matches stored version
                this.versionMismatch = !isVersionMatch(this.currentVersionInfo);
                
                if (this.versionMismatch) {
                    console.log('Version mismatch detected:', {
                        current: this.currentVersionInfo,
                        stored: getStoredVersionInfo()
                    });
                    
                    // Reset dismissed status for version mismatch banner
                    resetDismissedBanner('version-mismatch');
                    
                    // Register and show the version mismatch banner
                    this.registerVersionMismatchBanner();
                }
            }
        } catch (error) {
            console.error('Failed to check version info:', error);
        }
    }
    
    registerVersionMismatchBanner() {
        // Get stored and current version for display
        const storedVersion = getStoredVersionInfo() || 'unknown';
        const currentVersion = this.currentVersionInfo || 'unknown';
        
        bannerService.registerBanner('version-mismatch', {
            id: 'version-mismatch',
            title: 'Application Update Detected',
            content: `Your browser is running an outdated version of LoRA Manager (${storedVersion}). The server has been updated to version ${currentVersion}. Please refresh to ensure proper functionality.`,
            actions: [
                {
                    text: 'Refresh Now',
                    icon: 'fas fa-sync',
                    action: 'hardRefresh',
                    type: 'primary'
                }
            ],
            dismissible: false,
            priority: 10,
            countdown: 15,
            onRegister: (bannerElement) => {
                // Add countdown element
                const countdownEl = document.createElement('div');
                countdownEl.className = 'banner-countdown';
                countdownEl.innerHTML = `<span>Refreshing in <strong>15</strong> seconds...</span>`;
                bannerElement.querySelector('.banner-content').appendChild(countdownEl);
                
                // Start countdown
                let seconds = 15;
                const countdownInterval = setInterval(() => {
                    seconds--;
                    const strongEl = countdownEl.querySelector('strong');
                    if (strongEl) strongEl.textContent = seconds;
                    
                    if (seconds <= 0) {
                        clearInterval(countdownInterval);
                        this.performHardRefresh();
                    }
                }, 1000);
                
                // Store interval ID for cleanup
                bannerElement.dataset.countdownInterval = countdownInterval;
                
                // Add action button event handler
                const actionBtn = bannerElement.querySelector('.banner-action[data-action="hardRefresh"]');
                if (actionBtn) {
                    actionBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        clearInterval(countdownInterval);
                        this.performHardRefresh();
                    });
                }
            },
            onRemove: (bannerElement) => {
                // Clear any existing interval
                const intervalId = bannerElement.dataset.countdownInterval;
                if (intervalId) {
                    clearInterval(parseInt(intervalId));
                }
            }
        });
    }
    
    performHardRefresh() {
        // Update stored version info before refreshing
        setStoredVersionInfo(this.currentVersionInfo);
        
        // Force a hard refresh by adding cache-busting parameter
        const cacheBuster = new Date().getTime();
        window.location.href = window.location.pathname + 
            (window.location.search ? window.location.search + '&' : '?') + 
            `cache=${cacheBuster}`;
    }
}

// Create and export singleton instance
export const updateService = new UpdateService();
