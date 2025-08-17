// Duplicates Manager Component
import { showToast } from '../utils/uiHelpers.js';
import { RecipeCard } from './RecipeCard.js';
import { state, getCurrentPageState } from '../state/index.js';

export class DuplicatesManager {
    constructor(recipeManager) {
        this.recipeManager = recipeManager;
        this.duplicateGroups = [];
        this.inDuplicateMode = false;
        this.selectedForDeletion = new Set();
    }
    
    async findDuplicates() {
        try {
            const response = await fetch('/api/recipes/find-duplicates');
            if (!response.ok) {
                throw new Error('Failed to find duplicates');
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Unknown error finding duplicates');
            }
            
            this.duplicateGroups = data.duplicate_groups || [];
            
            if (this.duplicateGroups.length === 0) {
                showToast('No duplicate recipes found', 'info');
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
        
        // Clear the recipe grid first
        const recipeGrid = document.getElementById('recipeGrid');
        if (recipeGrid) {
            recipeGrid.innerHTML = '';
        }
        
        // Re-enable virtual scrolling
        state.virtualScroller.enable();
    }
    
    renderDuplicateGroups() {
        const recipeGrid = document.getElementById('recipeGrid');
        if (!recipeGrid) return;
        
        // Clear existing content
        recipeGrid.innerHTML = '';
        
        // Render each duplicate group
        this.duplicateGroups.forEach((group, groupIndex) => {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'duplicate-group';
            groupDiv.dataset.fingerprint = group.fingerprint;
            
            // Create group header
            const header = document.createElement('div');
            header.className = 'duplicate-group-header';
            header.innerHTML = `
                <span>Duplicate Group #${groupIndex + 1} (${group.recipes.length} recipes)</span>
                <span>
                    <button class="btn-select-all" onclick="recipeManager.duplicatesManager.toggleSelectAllInGroup('${group.fingerprint}')">
                        Select All
                    </button>
                    <button class="btn-select-latest" onclick="recipeManager.duplicatesManager.selectLatestInGroup('${group.fingerprint}')">
                        Keep Latest
                    </button>
                </span>
            `;
            groupDiv.appendChild(header);
            
            // Create cards container
            const cardsDiv = document.createElement('div');
            cardsDiv.className = 'card-group-container';
            
            // Add scrollable class if there are many recipes in the group
            if (group.recipes.length > 6) {
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
            
            // Sort recipes by date (newest first)
            const sortedRecipes = [...group.recipes].sort((a, b) => b.modified - a.modified);
            
            // Add all recipe cards in this group
            sortedRecipes.forEach((recipe, index) => {
                // Create recipe card
                const recipeCard = new RecipeCard(recipe, (recipe) => {
                    this.recipeManager.showRecipeDetails(recipe);
                });
                const card = recipeCard.element;
                
                // Add duplicate class
                card.classList.add('duplicate');
                
                // Mark the latest one
                if (index === 0) {
                    card.classList.add('latest');
                }
                
                // Add selection checkbox
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'selector-checkbox';
                checkbox.dataset.recipeId = recipe.id;
                checkbox.dataset.groupFingerprint = group.fingerprint;
                
                // Check if already selected
                if (this.selectedForDeletion.has(recipe.id)) {
                    checkbox.checked = true;
                    card.classList.add('duplicate-selected');
                }
                
                // Add change event to checkbox
                checkbox.addEventListener('change', (e) => {
                    e.stopPropagation();
                    this.toggleCardSelection(recipe.id, card, checkbox);
                });
                
                // Make the entire card clickable for selection
                card.addEventListener('click', (e) => {
                    // Don't toggle if clicking on the checkbox directly or card actions
                    if (e.target === checkbox || e.target.closest('.card-actions')) {
                        return;
                    }
                    
                    // Toggle checkbox state
                    checkbox.checked = !checkbox.checked;
                    this.toggleCardSelection(recipe.id, card, checkbox);
                });
                
                card.appendChild(checkbox);
                cardsDiv.appendChild(card);
            });
            
            groupDiv.appendChild(cardsDiv);
            recipeGrid.appendChild(groupDiv);
        });
    }
    
    // Helper method to toggle card selection state
    toggleCardSelection(recipeId, card, checkbox) {
        if (checkbox.checked) {
            this.selectedForDeletion.add(recipeId);
            card.classList.add('duplicate-selected');
        } else {
            this.selectedForDeletion.delete(recipeId);
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
    
    toggleSelectAllInGroup(fingerprint) {
        const checkboxes = document.querySelectorAll(`.selector-checkbox[data-group-fingerprint="${fingerprint}"]`);
        const allSelected = Array.from(checkboxes).every(checkbox => checkbox.checked);
        
        // If all are selected, deselect all; otherwise select all
        checkboxes.forEach(checkbox => {
            checkbox.checked = !allSelected;
            const recipeId = checkbox.dataset.recipeId;
            const card = checkbox.closest('.model-card');
            
            if (!allSelected) {
                this.selectedForDeletion.add(recipeId);
                card.classList.add('duplicate-selected');
            } else {
                this.selectedForDeletion.delete(recipeId);
                card.classList.remove('duplicate-selected');
            }
        });
        
        // Update the button text
        const button = document.querySelector(`.duplicate-group[data-fingerprint="${fingerprint}"] .btn-select-all`);
        if (button) {
            button.textContent = !allSelected ? "Deselect All" : "Select All";
        }
        
        this.updateSelectedCount();
    }
    
    selectAllInGroup(fingerprint) {
        const checkboxes = document.querySelectorAll(`.selector-checkbox[data-group-fingerprint="${fingerprint}"]`);
        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
            this.selectedForDeletion.add(checkbox.dataset.recipeId);
            checkbox.closest('.model-card').classList.add('duplicate-selected');
        });
        
        // Update the button text
        const button = document.querySelector(`.duplicate-group[data-fingerprint="${fingerprint}"] .btn-select-all`);
        if (button) {
            button.textContent = "Deselect All";
        }
        
        this.updateSelectedCount();
    }
    
    selectLatestInGroup(fingerprint) {
        // Find all checkboxes in this group
        const checkboxes = document.querySelectorAll(`.selector-checkbox[data-group-fingerprint="${fingerprint}"]`);
        
        // Get all the recipes in this group
        const group = this.duplicateGroups.find(g => g.fingerprint === fingerprint);
        if (!group) return;
        
        // Sort recipes by date (newest first)
        const sortedRecipes = [...group.recipes].sort((a, b) => b.modified - a.modified);
        
        // Skip the first (latest) one and select the rest for deletion
        for (let i = 1; i < sortedRecipes.length; i++) {
            const recipeId = sortedRecipes[i].id;
            const checkbox = document.querySelector(`.selector-checkbox[data-recipe-id="${recipeId}"]`);
            
            if (checkbox) {
                checkbox.checked = true;
                this.selectedForDeletion.add(recipeId);
                checkbox.closest('.model-card').classList.add('duplicate-selected');
            }
        }
        
        // Make sure the latest one is not selected
        const latestId = sortedRecipes[0].id;
        const latestCheckbox = document.querySelector(`.selector-checkbox[data-recipe-id="${latestId}"]`);
        
        if (latestCheckbox) {
            latestCheckbox.checked = false;
            this.selectedForDeletion.delete(latestId);
            latestCheckbox.closest('.model-card').classList.remove('duplicate-selected');
        }
        
        this.updateSelectedCount();
    }
    
    selectLatestDuplicates() {
        // For each duplicate group, select all but the latest recipe
        this.duplicateGroups.forEach(group => {
            this.selectLatestInGroup(group.fingerprint);
        });
    }
    
    async deleteSelectedDuplicates() {
        if (this.selectedForDeletion.size === 0) {
            showToast('No recipes selected for deletion', 'info');
            return;
        }
        
        try {
            // Show the delete confirmation modal instead of a simple confirm
            const duplicateDeleteCount = document.getElementById('duplicateDeleteCount');
            if (duplicateDeleteCount) {
                duplicateDeleteCount.textContent = this.selectedForDeletion.size;
            }
            
            // Use the modal manager to show the confirmation modal
            modalManager.showModal('duplicateDeleteModal');
        } catch (error) {
            console.error('Error preparing delete:', error);
            showToast('Error: ' + error.message, 'error');
        }
    }
    
    // Add new method to execute deletion after confirmation
    async confirmDeleteDuplicates() {
        try {           
            // Close the modal
            modalManager.closeModal('duplicateDeleteModal');
            
            // Prepare recipe IDs for deletion
            const recipeIds = Array.from(this.selectedForDeletion);
            
            // Call API to bulk delete
            const response = await fetch('/api/recipes/bulk-delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ recipe_ids: recipeIds })
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete selected recipes');
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Unknown error deleting recipes');
            }
            
            showToast(`Successfully deleted ${data.total_deleted} recipes`, 'success');
            
            // Exit duplicate mode if deletions were successful
            if (data.total_deleted > 0) {
                this.exitDuplicateMode();
            }
            
        } catch (error) {
            console.error('Error deleting recipes:', error);
            showToast('Failed to delete recipes: ' + error.message, 'error');
        }
    }
}
