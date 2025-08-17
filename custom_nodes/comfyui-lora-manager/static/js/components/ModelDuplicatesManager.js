// Model Duplicates Manager Component for LoRAs and Checkpoints
import { showToast } from '../utils/uiHelpers.js';
import { state, getCurrentPageState } from '../state/index.js';
import { formatDate } from '../utils/formatters.js';
import { resetAndReload} from '../api/modelApiFactory.js';
import { LoadingManager } from '../managers/LoadingManager.js';

export class ModelDuplicatesManager {
    constructor(pageManager, modelType = 'loras') {
        this.pageManager = pageManager;
        this.duplicateGroups = [];
        this.inDuplicateMode = false;
        this.selectedForDeletion = new Set();
        this.modelType = modelType; // Use the provided modelType or default to 'loras'
        
        // Verification tracking
        this.verifiedGroups = new Set(); // Track which groups have been verified
        this.mismatchedFiles = new Map(); // Map file paths to actual hashes for mismatched files
        
        // Loading manager for verification process
        this.loadingManager = new LoadingManager();
        
        // Bind methods
        this.renderModelCard = this.renderModelCard.bind(this);
        this.renderTooltip = this.renderTooltip.bind(this);
        this.checkDuplicatesCount = this.checkDuplicatesCount.bind(this);
        this.handleVerifyHashes = this.handleVerifyHashes.bind(this);
        
        // Keep track of which controls need to be re-enabled
        this.disabledControls = [];
        
        // Check for duplicates on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', this.checkDuplicatesCount);
        } else {
            this.checkDuplicatesCount();
        }
    }
    
    // Method to check for duplicates count using existing endpoint
    async checkDuplicatesCount() {
        try {
            const endpoint = `/api/${this.modelType}/find-duplicates`;
            const response = await fetch(endpoint);
            
            if (!response.ok) {
                throw new Error(`Failed to get duplicates count: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                const duplicatesCount = (data.duplicates || []).length;
                this.updateDuplicatesBadge(duplicatesCount);
            } else {
                this.updateDuplicatesBadge(0);
            }
        } catch (error) {
            console.error('Error checking duplicates count:', error);
            this.updateDuplicatesBadge(0);
        }
    }
    
    // Method to update the badge
    updateDuplicatesBadge(count) {
        const badge = document.getElementById('duplicatesBadge');
        if (!badge) return;
        
        if (count > 0) {
            badge.textContent = count;
            badge.classList.add('pulse');
        } else {
            badge.textContent = '';
            badge.classList.remove('pulse');
        }
    }
    
    // Toggle method to enter/exit duplicates mode
    toggleDuplicateMode() {
        if (this.inDuplicateMode) {
            this.exitDuplicateMode();
        } else {
            this.findDuplicates();
        }
    }
    
    async findDuplicates() {
        try {
            // Determine API endpoint based on model type
            const endpoint = `/api/${this.modelType}/find-duplicates`;
            
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`Failed to find duplicates: ${response.statusText}`);
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Unknown error finding duplicates');
            }
            
            this.duplicateGroups = data.duplicates || [];
            
            // Update the badge with the current count
            this.updateDuplicatesBadge(this.duplicateGroups.length);
            
            if (this.duplicateGroups.length === 0) {
                showToast('No duplicate models found', 'info');
                return false;
            }
            
            this.enterDuplicateMode();
            return true;
        } catch (error) {
            console.error('Error finding duplicates:', error);
            showToast('Failed to find duplicates: ' + error.message, 'error');
            return false;
        }
    }
    
    enterDuplicateMode() {
        this.inDuplicateMode = true;
        this.selectedForDeletion.clear();
        
        // Update state
        const pageState = getCurrentPageState();
        pageState.duplicatesMode = true;
        
        // Show duplicates banner
        const banner = document.getElementById('duplicatesBanner');
        const countSpan = document.getElementById('duplicatesCount');
        
        if (banner && countSpan) {
            countSpan.textContent = `Found ${this.duplicateGroups.length} duplicate group${this.duplicateGroups.length !== 1 ? 's' : ''}`;
            banner.style.display = 'block';
            
            // Setup help tooltip behavior
            this.setupHelpTooltip();
        }
        
        // Disable virtual scrolling if active
        if (state.virtualScroller) {
            state.virtualScroller.disable();
        }
        
        // Add duplicate-mode class to the body
        document.body.classList.add('duplicate-mode');
        
        // Render duplicate groups
        this.renderDuplicateGroups();
        
        // Update selected count
        this.updateSelectedCount();
        
        // Update Duplicates button to show active state
        const duplicatesBtn = document.getElementById('findDuplicatesBtn');
        if (duplicatesBtn) {
            duplicatesBtn.classList.add('active');
            duplicatesBtn.title = 'Exit Duplicates Mode';
            // Change icon and text to indicate it's now an exit button
            duplicatesBtn.innerHTML = '<i class="fas fa-times"></i> Exit Duplicates';
        }
        
        // Disable all control buttons except the duplicates button
        this.disableControlButtons();
    }
    
    exitDuplicateMode() {
        this.inDuplicateMode = false;
        this.selectedForDeletion.clear();
        
        // Update state
        const pageState = getCurrentPageState();
        pageState.duplicatesMode = false;
        
        // Hide duplicates banner
        const banner = document.getElementById('duplicatesBanner');
        if (banner) {
            banner.style.display = 'none';
        }
        
        // Remove duplicate-mode class from the body
        document.body.classList.remove('duplicate-mode');
        
        // Clear the model grid first
        const modelGrid = document.getElementById('modelGrid');
        if (modelGrid) {
            modelGrid.innerHTML = '';
        }
        
        // Re-enable virtual scrolling
        state.virtualScroller.enable();
        
        // Restore Duplicates button to its original state
        const duplicatesBtn = document.getElementById('findDuplicatesBtn');
        if (duplicatesBtn) {
            duplicatesBtn.classList.remove('active');
            duplicatesBtn.title = 'Find duplicate models';
            duplicatesBtn.innerHTML = '<i class="fas fa-clone"></i> Duplicates <span id="duplicatesBadge" class="badge"></span>';
            
            // Restore badge
            const newBadge = duplicatesBtn.querySelector('#duplicatesBadge');
            const oldBadge = document.getElementById('duplicatesBadge');
            if (oldBadge && oldBadge.textContent) {
                newBadge.textContent = oldBadge.textContent;
                newBadge.classList.add('pulse');
            }
        }
        
        // Re-enable all control buttons
        this.enableControlButtons();

        this.checkDuplicatesCount();
    }
    
    // Disable all control buttons except the duplicates button
    disableControlButtons() {
        this.disabledControls = [];
        
        // Select all control buttons except the duplicates button
        const controlButtons = document.querySelectorAll('.control-group button:not(#findDuplicatesBtn), .dropdown-group, .toggle-folders-btn, #favoriteFilterBtn');
        
        controlButtons.forEach(button => {
            // Only disable enabled buttons (don't disable already disabled buttons)
            if (!button.disabled && !button.classList.contains('disabled')) {
                this.disabledControls.push(button);
                button.disabled = true;
                button.classList.add('disabled-during-duplicates');
            }
        });
    }
    
    // Re-enable all previously disabled control buttons
    enableControlButtons() {
        this.disabledControls.forEach(button => {
            button.disabled = false;
            button.classList.remove('disabled-during-duplicates');
        });
        this.disabledControls = [];
    }
    
    renderDuplicateGroups() {
        const modelGrid = document.getElementById('modelGrid');
        if (!modelGrid) return;
        
        // Clear existing content
        modelGrid.innerHTML = '';
        
        // Render each duplicate group
        this.duplicateGroups.forEach((group, groupIndex) => {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'duplicate-group';
            groupDiv.dataset.hash = group.hash;
            
            // Create group header
            const header = document.createElement('div');
            header.className = 'duplicate-group-header';
            
            // Create verification status badge
            const verificationBadge = document.createElement('span');
            verificationBadge.className = 'verification-badge';
            if (this.verifiedGroups.has(group.hash)) {
                verificationBadge.classList.add('verified');
                verificationBadge.innerHTML = '<i class="fas fa-check-circle"></i> Verified';
            } else {
                verificationBadge.classList.add('metadata');
                verificationBadge.innerHTML = '<i class="fas fa-tag"></i> Metadata Hash';
            }
            
            header.innerHTML = `
                <span>Duplicate Group #${groupIndex + 1} (${group.models.length} models with same hash: ${group.hash})</span>
                <span>
                    <button class="btn-verify-hashes" data-hash="${group.hash}" title="Recalculate SHA256 hashes to verify if these are true duplicates">
                        <i class="fas fa-fingerprint"></i> Verify Hashes
                    </button>
                    <button class="btn-select-all" onclick="modelDuplicatesManager.toggleSelectAllInGroup('${group.hash}')">
                        Select All
                    </button>
                </span>
            `;
            
            // Insert verification badge after the group title
            const headerFirstSpan = header.querySelector('span:first-child');
            headerFirstSpan.appendChild(verificationBadge);
            
            groupDiv.appendChild(header);
            
            // Create cards container
            const cardsDiv = document.createElement('div');
            cardsDiv.className = 'card-group-container';
            
            // Add scrollable class if there are many models in the group
            if (group.models.length > 6) {
                cardsDiv.classList.add('scrollable');
                
                // Add expand/collapse toggle button
                const toggleBtn = document.createElement('button');
                toggleBtn.className = 'group-toggle-btn';
                toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
                toggleBtn.title = "Expand/Collapse";
                toggleBtn.onclick = function() {
                    cardsDiv.classList.toggle('scrollable');
                    this.innerHTML = cardsDiv.classList.contains('scrollable') ? 
                        '<i class="fas fa-chevron-down"></i>' : 
                        '<i class="fas fa-chevron-up"></i>';
                };
                groupDiv.appendChild(toggleBtn);
            }
            
            // Add all model cards in this group
            group.models.forEach(model => {
                const card = this.renderModelCard(model, group.hash);
                cardsDiv.appendChild(card);
            });
            
            groupDiv.appendChild(cardsDiv);
            modelGrid.appendChild(groupDiv);
            
            // Add event listener to the verify hashes button
            const verifyButton = header.querySelector('.btn-verify-hashes');
            if (verifyButton) {
                verifyButton.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.handleVerifyHashes(group);
                });
            }
        });
    }
    
    renderModelCard(model, groupHash) {
        // Create basic card structure
        const card = document.createElement('div');
        card.className = 'model-card duplicate';
        card.dataset.hash = model.sha256;
        card.dataset.filePath = model.file_path;
        
        // Check if this model is a mismatched file
        const isMismatched = this.mismatchedFiles.has(model.file_path);
        
        // Add mismatched class if needed
        if (isMismatched) {
            card.classList.add('hash-mismatch');
        }
        
        // Create card content using structure similar to createLoraCard in LoraCard.js
        const previewContainer = document.createElement('div');
        previewContainer.className = 'card-preview';
        
        // Determine if preview is a video
        const isVideo = model.preview_url && model.preview_url.endsWith('.mp4');
        let preview;
        
        if (isVideo) {
            // Create video element for MP4 previews
            preview = document.createElement('video');
            preview.loading = 'lazy';
            preview.controls = true;
            preview.muted = true;
            preview.loop = true;
            
            const source = document.createElement('source');
            source.src = model.preview_url;
            source.type = 'video/mp4';
            preview.appendChild(source);
        } else {
            // Create image element for standard previews
            preview = document.createElement('img');
            preview.loading = 'lazy';
            preview.alt = model.model_name;
            
            if (model.preview_url) {
                preview.src = model.preview_url;
            } else {
                // Use placeholder
                preview.src = '/loras_static/images/no-preview.png';
            }
        }
        
        // Add NSFW blur if needed
        if (model.preview_nsfw_level > 0) {
            preview.classList.add('nsfw');
        }
        
        previewContainer.appendChild(preview);
        
        // Add hash mismatch badge if needed
        if (isMismatched) {
            const mismatchBadge = document.createElement('div');
            mismatchBadge.className = 'mismatch-badge';
            mismatchBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Different Hash';
            previewContainer.appendChild(mismatchBadge);
        }
        
        // Mark as latest if applicable
        if (model.is_latest) {
            card.classList.add('latest');
        }
        
        // Move tooltip listeners to the preview container for consistent behavior
        // regardless of whether the preview is an image or video
        previewContainer.addEventListener('mouseover', () => this.renderTooltip(card, model));
        previewContainer.addEventListener('mouseout', () => {
            const tooltip = document.querySelector('.model-tooltip');
            if (tooltip) tooltip.remove();
        });
        
        // Add card footer with just model name
        const footer = document.createElement('div');
        footer.className = 'card-footer';
        
        const modelInfo = document.createElement('div');
        modelInfo.className = 'model-info';
        
        const modelName = document.createElement('span');
        modelName.className = 'model-name';
        modelName.textContent = model.model_name;
        modelInfo.appendChild(modelName);
        
        footer.appendChild(modelInfo);
        previewContainer.appendChild(footer);
        card.appendChild(previewContainer);
        
        // Add selection checkbox
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'selector-checkbox';
        checkbox.dataset.filePath = model.file_path;
        checkbox.dataset.groupHash = groupHash;
        
        // Check if already selected
        if (this.selectedForDeletion.has(model.file_path)) {
            checkbox.checked = true;
            card.classList.add('duplicate-selected');
        }
        
        // Disable checkbox for mismatched files
        if (isMismatched) {
            checkbox.disabled = true;
            checkbox.title = "This file has a different actual hash and can't be selected";
        }
        
        // Add change event to checkbox
        checkbox.addEventListener('change', (e) => {
            e.stopPropagation();
            this.toggleCardSelection(model.file_path, card, checkbox);
        });
        
        // Make the entire card clickable for selection
        card.addEventListener('click', (e) => {
            // Don't toggle if clicking on the checkbox directly or card actions
            if (e.target === checkbox || e.target.closest('.card-actions')) {
                return;
            }
            
            // Don't toggle if it's a mismatched file
            if (isMismatched) {
                return;
            }
            
            // Toggle checkbox state
            checkbox.checked = !checkbox.checked;
            this.toggleCardSelection(model.file_path, card, checkbox);
        });
        
        card.appendChild(checkbox);
        return card;
    }
    
    renderTooltip(card, model) {
        // Remove any existing tooltips
        const existingTooltip = document.querySelector('.model-tooltip');
        if (existingTooltip) existingTooltip.remove();
        
        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'model-tooltip';
        
        // Check if this model is a mismatched file and get the actual hash
        const isMismatched = this.mismatchedFiles.has(model.file_path);
        const actualHash = isMismatched ? this.mismatchedFiles.get(model.file_path) : null;
        
        // Add model information to tooltip
        let tooltipContent = `
            <div class="tooltip-header">${model.model_name}</div>
            <div class="tooltip-info">
                <div><strong>Version:</strong> ${model.civitai?.name || 'Unknown'}</div>
                <div><strong>Filename:</strong> ${model.file_name}</div>
                <div><strong>Path:</strong> ${model.file_path}</div>
                <div><strong>Base Model:</strong> ${model.base_model || 'Unknown'}</div>
                <div><strong>Modified:</strong> ${formatDate(model.modified)}</div>
                <div><strong>Metadata Hash:</strong> <span class="hash-value">${model.sha256}</span></div>
        `;
        
        // Add actual hash information if available
        if (isMismatched && actualHash) {
            tooltipContent += `<div class="hash-mismatch-info"><strong>Actual Hash:</strong> <span class="hash-value">${actualHash}</span></div>`;
        }
        
        tooltipContent += `</div>`;
        tooltip.innerHTML = tooltipContent;
        
        // Position tooltip relative to card
        const cardRect = card.getBoundingClientRect();
        tooltip.style.top = `${cardRect.top + window.scrollY - 10}px`;
        tooltip.style.left = `${cardRect.left + window.scrollX + cardRect.width + 10}px`;
        
        // Add tooltip to document
        document.body.appendChild(tooltip);
        
        // Check if tooltip is outside viewport and adjust if needed
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = `${cardRect.left + window.scrollX - tooltipRect.width - 10}px`;
        }
    }
    
    // Helper method to toggle card selection state
    toggleCardSelection(filePath, card, checkbox) {
        if (checkbox.checked) {
            this.selectedForDeletion.add(filePath);
            card.classList.add('duplicate-selected');
        } else {
            this.selectedForDeletion.delete(filePath);
            card.classList.remove('duplicate-selected');
        }
        
        this.updateSelectedCount();
    }
    
    updateSelectedCount() {
        const selectedCountEl = document.getElementById('duplicatesSelectedCount');
        if (selectedCountEl) {
            selectedCountEl.textContent = this.selectedForDeletion.size;
        }
        
        // Update delete button state
        const deleteBtn = document.querySelector('.btn-delete-selected');
        if (deleteBtn) {
            deleteBtn.disabled = this.selectedForDeletion.size === 0;
            deleteBtn.classList.toggle('disabled', this.selectedForDeletion.size === 0);
        }
    }
    
    toggleSelectAllInGroup(hash) {
        const checkboxes = document.querySelectorAll(`.selector-checkbox[data-group-hash="${hash}"]`);
        const allSelected = Array.from(checkboxes).every(checkbox => checkbox.checked);
        
        // If all are selected, deselect all; otherwise select all
        checkboxes.forEach(checkbox => {
            checkbox.checked = !allSelected;
            const filePath = checkbox.dataset.filePath;
            const card = checkbox.closest('.model-card');
            
            if (!allSelected) {
                this.selectedForDeletion.add(filePath);
                card.classList.add('duplicate-selected');
            } else {
                this.selectedForDeletion.delete(filePath);
                card.classList.remove('duplicate-selected');
            }
        });
        
        // Update the button text
        const button = document.querySelector(`.duplicate-group[data-hash="${hash}"] .btn-select-all`);
        if (button) {
            button.textContent = !allSelected ? "Deselect All" : "Select All";
        }
        
        this.updateSelectedCount();
    }
    
    async deleteSelectedDuplicates() {
        if (this.selectedForDeletion.size === 0) {
            showToast('No models selected for deletion', 'info');
            return;
        }
        
        try {
            // Show the delete confirmation modal instead of a simple confirm
            const modelDuplicateDeleteCount = document.getElementById('modelDuplicateDeleteCount');
            if (modelDuplicateDeleteCount) {
              modelDuplicateDeleteCount.textContent = this.selectedForDeletion.size;
            }
            
            // Use the modal manager to show the confirmation modal
            modalManager.showModal('modelDuplicateDeleteModal');
        } catch (error) {
            console.error('Error preparing delete:', error);
            showToast('Error: ' + error.message, 'error');
        }
    }
    
    // Execute deletion after confirmation
    async confirmDeleteDuplicates() {
        try {           
            // Close the modal
            modalManager.closeModal('modelDuplicateDeleteModal');
            
            // Prepare file paths for deletion
            const filePaths = Array.from(this.selectedForDeletion);
            
            // Call API to bulk delete
            const response = await fetch(`/api/${this.modelType}/bulk-delete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_paths: filePaths })
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete selected models');
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Unknown error deleting models');
            }
            
            showToast(`Successfully deleted ${data.total_deleted} models`, 'success');
            
            // If models were successfully deleted
            if (data.total_deleted > 0) {
                await resetAndReload(true);
                
                // Check if there are still duplicates
                try {
                    const endpoint = `/api/${this.modelType}/find-duplicates`;
                    const dupResponse = await fetch(endpoint);
                    
                    if (!dupResponse.ok) {
                        throw new Error(`Failed to get duplicates: ${dupResponse.statusText}`);
                    }
                    
                    const dupData = await dupResponse.json();
                    const remainingDuplicatesCount = (dupData.duplicates || []).length;
                    
                    // Update badge count
                    this.updateDuplicatesBadge(remainingDuplicatesCount);
                    
                    // If no more duplicates, exit duplicate mode
                    if (remainingDuplicatesCount === 0) {
                        this.exitDuplicateMode();
                    } else {
                        // If duplicates remain, refresh duplicate groups display
                        this.duplicateGroups = dupData.duplicates || [];
                        this.selectedForDeletion.clear();
                        this.renderDuplicateGroups();
                        this.updateSelectedCount();
                    }
                } catch (error) {
                    console.error('Error checking remaining duplicates:', error);
                }
            }
            
        } catch (error) {
            console.error('Error deleting models:', error);
            showToast('Failed to delete models: ' + error.message, 'error');
        }
    }
    
    // Public method to update the badge after refresh
    updateDuplicatesBadgeAfterRefresh() {
        // Use this method after refresh operations
        this.checkDuplicatesCount();
    }

    // Add this new method for tooltip behavior
    setupHelpTooltip() {
      const helpIcon = document.getElementById('duplicatesHelp');
      const helpTooltip = document.getElementById('duplicatesHelpTooltip');
      
      if (!helpIcon || !helpTooltip) return;
      
      helpIcon.addEventListener('mouseenter', (e) => {
          // Get the container's positioning context
          const bannerContent = helpIcon.closest('.banner-content');
          
          // Get positions relative to the viewport
          const iconRect = helpIcon.getBoundingClientRect();
          const bannerRect = bannerContent.getBoundingClientRect();
          
          // Set initial position relative to the banner content
          helpTooltip.style.display = 'block';
          helpTooltip.style.top = `${iconRect.bottom - bannerRect.top + 10}px`;
          helpTooltip.style.left = `${iconRect.left - bannerRect.left - 10}px`;
          
          // Check if the tooltip is going off-screen to the right
          const tooltipRect = helpTooltip.getBoundingClientRect();
          const viewportWidth = window.innerWidth;
          
          if (tooltipRect.right > viewportWidth - 20) {
              // Reposition relative to container if too close to right edge
              helpTooltip.style.left = `${bannerContent.offsetWidth - tooltipRect.width - 20}px`;
          }
      });
      
      // Rest of the event listeners remain unchanged
      helpIcon.addEventListener('mouseleave', () => {
          helpTooltip.style.display = 'none';
      });
      
      document.addEventListener('click', (e) => {
          if (!helpIcon.contains(e.target)) {
              helpTooltip.style.display = 'none';
          }
      });
    }

    // Handle verify hashes button click
    async handleVerifyHashes(group) {
        try {
            const groupHash = group.hash;
            
            // Check if already verified
            if (this.verifiedGroups.has(groupHash)) {
                showToast('This group has already been verified', 'info');
                return;
            }
            
            // Show loading state
            this.loadingManager.showSimpleLoading('Verifying hashes...');
            
            // Get file paths for all models in the group
            const filePaths = group.models.map(model => model.file_path);
            
            // Make API request to verify hashes
            const response = await fetch(`/api/${this.modelType}/verify-duplicates`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_paths: filePaths })
            });
            
            if (!response.ok) {
                throw new Error(`Verification failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Unknown error during verification');
            }
            
            // Process verification results
            const verifiedAsDuplicates = data.verified_as_duplicates;
            const mismatchedFiles = data.mismatched_files || [];
            
            // Update mismatchedFiles map
            if (data.new_hash_map) {
                Object.entries(data.new_hash_map).forEach(([path, hash]) => {
                    this.mismatchedFiles.set(path, hash);
                });
            }
            
            // Mark this group as verified
            this.verifiedGroups.add(groupHash);
            
            // Re-render the duplicate groups to show verification status
            this.renderDuplicateGroups();
            
            // Show appropriate toast message
            if (mismatchedFiles.length > 0) {
                showToast(`Verification complete. ${mismatchedFiles.length} file(s) have different actual hashes.`, 'warning');
            } else {
                showToast('Verification complete. All files are confirmed duplicates.', 'success');
            }
            
        } catch (error) {
            console.error('Error verifying hashes:', error);
            showToast('Failed to verify hashes: ' + error.message, 'error');
        } finally {
            // Hide loading state
            this.loadingManager.hide();
        }
    }
}
