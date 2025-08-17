/**
 * FolderTreeManager - Manages folder tree UI for download modal
 */
export class FolderTreeManager {
    constructor() {
        this.treeData = {};
        this.selectedPath = '';
        this.expandedNodes = new Set();
        this.pathSuggestions = [];
        this.onPathChangeCallback = null;
        this.activeSuggestionIndex = -1;
        this.elementsPrefix = '';
        
        // Bind methods
        this.handleTreeClick = this.handleTreeClick.bind(this);
        this.handlePathInput = this.handlePathInput.bind(this);
        this.handlePathSuggestionClick = this.handlePathSuggestionClick.bind(this);
        this.handleCreateFolder = this.handleCreateFolder.bind(this);
        this.handleBreadcrumbClick = this.handleBreadcrumbClick.bind(this);
        this.handlePathKeyDown = this.handlePathKeyDown.bind(this);
    }

    /**
     * Initialize the folder tree manager
     * @param {Object} config - Configuration object
     * @param {Function} config.onPathChange - Callback when path changes
     * @param {string} config.elementsPrefix - Prefix for element IDs (e.g., 'move' for move modal)
     */
    init(config = {}) {
        this.onPathChangeCallback = config.onPathChange;
        this.elementsPrefix = config.elementsPrefix || '';
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        const pathInput = document.getElementById(this.getElementId('folderPath'));
        const createFolderBtn = document.getElementById(this.getElementId('createFolderBtn'));
        const folderTree = document.getElementById(this.getElementId('folderTree'));
        const breadcrumbNav = document.getElementById(this.getElementId('breadcrumbNav'));
        const pathSuggestions = document.getElementById(this.getElementId('pathSuggestions'));

        if (pathInput) {
            pathInput.addEventListener('input', this.handlePathInput);
            pathInput.addEventListener('keydown', this.handlePathKeyDown);
        }

        if (createFolderBtn) {
            createFolderBtn.addEventListener('click', this.handleCreateFolder);
        }

        if (folderTree) {
            folderTree.addEventListener('click', this.handleTreeClick);
        }

        if (breadcrumbNav) {
            breadcrumbNav.addEventListener('click', this.handleBreadcrumbClick);
        }

        if (pathSuggestions) {
            pathSuggestions.addEventListener('click', this.handlePathSuggestionClick);
        }

        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            const pathInput = document.getElementById(this.getElementId('folderPath'));
            const suggestions = document.getElementById(this.getElementId('pathSuggestions'));
            
            if (pathInput && suggestions && 
                !pathInput.contains(e.target) && 
                !suggestions.contains(e.target)) {
                suggestions.style.display = 'none';
                this.activeSuggestionIndex = -1;
            }
        });
    }

    /**
     * Get element ID with prefix
     */
    getElementId(elementName) {
        return this.elementsPrefix ? `${this.elementsPrefix}${elementName.charAt(0).toUpperCase()}${elementName.slice(1)}` : elementName;
    }

    /**
     * Handle path input key events with enhanced keyboard navigation
     */
    handlePathKeyDown(event) {
        const suggestions = document.getElementById(this.getElementId('pathSuggestions'));
        const isVisible = suggestions && suggestions.style.display !== 'none';
        
        if (isVisible) {
            const suggestionItems = suggestions.querySelectorAll('.path-suggestion');
            const maxIndex = suggestionItems.length - 1;
            
            switch (event.key) {
                case 'Escape':
                    event.preventDefault();
                    event.stopPropagation();
                    this.hideSuggestions();
                    this.activeSuggestionIndex = -1;
                    break;
                    
                case 'ArrowDown':
                    event.preventDefault();
                    this.activeSuggestionIndex = Math.min(this.activeSuggestionIndex + 1, maxIndex);
                    this.updateActiveSuggestion(suggestionItems);
                    break;
                    
                case 'ArrowUp':
                    event.preventDefault();
                    this.activeSuggestionIndex = Math.max(this.activeSuggestionIndex - 1, -1);
                    this.updateActiveSuggestion(suggestionItems);
                    break;
                    
                case 'Enter':
                    event.preventDefault();
                    if (this.activeSuggestionIndex >= 0 && suggestionItems[this.activeSuggestionIndex]) {
                        const path = suggestionItems[this.activeSuggestionIndex].dataset.path;
                        this.selectPath(path);
                        this.hideSuggestions();
                    } else {
                        this.selectCurrentInput();
                    }
                    break;
            }
        } else if (event.key === 'Enter') {
            event.preventDefault();
            this.selectCurrentInput();
        }
    }

    /**
     * Update active suggestion highlighting
     */
    updateActiveSuggestion(suggestionItems) {
        suggestionItems.forEach((item, index) => {
            item.classList.toggle('active', index === this.activeSuggestionIndex);
            if (index === this.activeSuggestionIndex) {
                item.scrollIntoView({ block: 'nearest' });
            }
        });
    }

    /**
     * Load and render folder tree data
     * @param {Object} treeData - Hierarchical tree data
     */
    async loadTree(treeData) {
        this.treeData = treeData;
        this.pathSuggestions = this.extractAllPaths(treeData);
        this.renderTree();
    }

    /**
     * Extract all paths from tree data for autocomplete
     */
    extractAllPaths(treeData, currentPath = '') {
        const paths = [];
        
        for (const [folderName, children] of Object.entries(treeData)) {
            const newPath = currentPath ? `${currentPath}/${folderName}` : folderName;
            paths.push(newPath);
            
            if (Object.keys(children).length > 0) {
                paths.push(...this.extractAllPaths(children, newPath));
            }
        }
        
        return paths.sort();
    }

    /**
     * Render the complete folder tree
     */
    renderTree() {
        const folderTree = document.getElementById(this.getElementId('folderTree'));
        if (!folderTree) return;

        // Show placeholder if treeData is empty
        if (!this.treeData || Object.keys(this.treeData).length === 0) {
            folderTree.innerHTML = `
                <div class="folder-tree-placeholder" style="padding:24px;text-align:center;color:var(--text-color);opacity:0.7;">
                    <i class="fas fa-folder-open" style="font-size:2em;opacity:0.5;"></i>
                    <div>No folders found.<br/>You can create a new folder using the button above.</div>
                </div>
            `;
            return;
        }

        folderTree.innerHTML = this.renderTreeNode(this.treeData, '');
    }

    /**
     * Render a single tree node
     */
    renderTreeNode(nodeData, basePath) {
        const entries = Object.entries(nodeData);
        if (entries.length === 0) return '';
        
        return entries.map(([folderName, children]) => {
            const currentPath = basePath ? `${basePath}/${folderName}` : folderName;
            const hasChildren = Object.keys(children).length > 0;
            const isExpanded = this.expandedNodes.has(currentPath);
            const isSelected = this.selectedPath === currentPath;
            
            return `
                <div class="tree-node ${hasChildren ? 'has-children' : ''}" data-path="${currentPath}">
                    <div class="tree-node-content ${isSelected ? 'selected' : ''}">
                        <div class="tree-expand-icon ${isExpanded ? 'expanded' : ''}" 
                             style="${hasChildren ? '' : 'opacity: 0; pointer-events: none;'}">
                            <i class="fas fa-chevron-right"></i>
                        </div>
                        <div class="tree-folder-icon">
                            <i class="fas fa-folder"></i>
                        </div>
                        <div class="tree-folder-name">${folderName}</div>
                    </div>
                    ${hasChildren ? `
                        <div class="tree-children ${isExpanded ? 'expanded' : ''}">
                            ${this.renderTreeNode(children, currentPath)}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
    }

    /**
     * Handle tree node clicks
     */
    handleTreeClick(event) {
        const expandIcon = event.target.closest('.tree-expand-icon');
        const nodeContent = event.target.closest('.tree-node-content');
        
        if (expandIcon) {
            // Toggle expand/collapse
            const treeNode = expandIcon.closest('.tree-node');
            const path = treeNode.dataset.path;
            const children = treeNode.querySelector('.tree-children');
            
            if (this.expandedNodes.has(path)) {
                this.expandedNodes.delete(path);
                expandIcon.classList.remove('expanded');
                if (children) children.classList.remove('expanded');
            } else {
                this.expandedNodes.add(path);
                expandIcon.classList.add('expanded');
                if (children) children.classList.add('expanded');
            }
        } else if (nodeContent) {
            // Select folder
            const treeNode = nodeContent.closest('.tree-node');
            const path = treeNode.dataset.path;
            this.selectPath(path);
        }
    }

    /**
     * Handle path input changes
     */
    handlePathInput(event) {
        const input = event.target;
        const query = input.value.toLowerCase();
        
        this.activeSuggestionIndex = -1; // Reset active suggestion
        
        if (query.length === 0) {
            this.hideSuggestions();
            return;
        }
        
        const matches = this.pathSuggestions.filter(path => 
            path.toLowerCase().includes(query)
        ).slice(0, 10); // Limit to 10 suggestions
        
        this.showSuggestions(matches, query);
    }

    /**
     * Show path suggestions
     */
    showSuggestions(suggestions, query) {
        const suggestionsEl = document.getElementById(this.getElementId('pathSuggestions'));
        if (!suggestionsEl) return;
        
        if (suggestions.length === 0) {
            this.hideSuggestions();
            return;
        }
        
        suggestionsEl.innerHTML = suggestions.map(path => {
            const highlighted = this.highlightMatch(path, query);
            return `<div class="path-suggestion" data-path="${path}">${highlighted}</div>`;
        }).join('');
        
        suggestionsEl.style.display = 'block';
        this.activeSuggestionIndex = -1; // Reset active index
    }

    /**
     * Hide path suggestions
     */
    hideSuggestions() {
        const suggestionsEl = document.getElementById(this.getElementId('pathSuggestions'));
        if (suggestionsEl) {
            suggestionsEl.style.display = 'none';
        }
    }

    /**
     * Highlight matching text in suggestions
     */
    highlightMatch(text, query) {
        const index = text.toLowerCase().indexOf(query.toLowerCase());
        if (index === -1) return text;
        
        return text.substring(0, index) + 
               `<strong>${text.substring(index, index + query.length)}</strong>` + 
               text.substring(index + query.length);
    }

    /**
     * Handle suggestion clicks
     */
    handlePathSuggestionClick(event) {
        const suggestion = event.target.closest('.path-suggestion');
        if (suggestion) {
            const path = suggestion.dataset.path;
            this.selectPath(path);
            this.hideSuggestions();
        }
    }

    /**
     * Handle create folder button click
     */
    handleCreateFolder() {
        const currentPath = this.selectedPath;
        this.showCreateFolderForm(currentPath);
    }

    /**
     * Show inline create folder form
     */
    showCreateFolderForm(parentPath) {
        // Find the parent node in the tree
        const parentNode = parentPath ? 
            document.querySelector(`[data-path="${parentPath}"]`) : 
            document.getElementById(this.getElementId('folderTree'));
        
        if (!parentNode) return;
        
        // Check if form already exists
        if (parentNode.querySelector('.create-folder-form')) return;
        
        const form = document.createElement('div');
        form.className = 'create-folder-form';
        form.innerHTML = `
            <input type="text" placeholder="New folder name" class="new-folder-input" />
            <button type="button" class="confirm">✓</button>
            <button type="button" class="cancel">✗</button>
        `;
        
        const input = form.querySelector('.new-folder-input');
        const confirmBtn = form.querySelector('.confirm');
        const cancelBtn = form.querySelector('.cancel');
        
        confirmBtn.addEventListener('click', () => {
            const folderName = input.value.trim();
            if (folderName) {
                this.createFolder(parentPath, folderName);
            }
            form.remove();
        });
        
        cancelBtn.addEventListener('click', () => {
            form.remove();
        });
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                confirmBtn.click();
            } else if (e.key === 'Escape') {
                cancelBtn.click();
            }
        });
        
        if (parentPath) {
            // Add to children area
            const childrenEl = parentNode.querySelector('.tree-children');
            if (childrenEl) {
                childrenEl.appendChild(form);
            } else {
                parentNode.appendChild(form);
            }
        } else {
            // Add to root
            parentNode.appendChild(form);
        }
        
        input.focus();
    }

    /**
     * Create new folder
     */
    createFolder(parentPath, folderName) {
        const newPath = parentPath ? `${parentPath}/${folderName}` : folderName;
        
        // Add to tree data
        const pathParts = newPath.split('/');
        let current = this.treeData;
        
        for (const part of pathParts) {
            if (!current[part]) {
                current[part] = {};
            }
            current = current[part];
        }
        
        // Update suggestions
        this.pathSuggestions = this.extractAllPaths(this.treeData);
        
        // Expand parent if needed
        if (parentPath) {
            this.expandedNodes.add(parentPath);
        }
        
        // Re-render tree
        this.renderTree();
        
        // Select the new folder
        this.selectPath(newPath);
    }

    /**
     * Handle breadcrumb navigation clicks
     */
    handleBreadcrumbClick(event) {
        const breadcrumbItem = event.target.closest('.breadcrumb-item');
        if (breadcrumbItem) {
            const path = breadcrumbItem.dataset.path;
            this.selectPath(path);
        }
    }

    /**
     * Select a path and update UI
     */
    selectPath(path) {
        this.selectedPath = path;
        
        // Update path input
        const pathInput = document.getElementById(this.getElementId('folderPath'));
        if (pathInput) {
            pathInput.value = path;
        }
        
        // Update tree selection
        const treeContainer = document.getElementById(this.getElementId('folderTree'));
        if (treeContainer) {
            treeContainer.querySelectorAll('.tree-node-content').forEach(node => {
                node.classList.remove('selected');
            });
            
            const selectedNode = treeContainer.querySelector(`[data-path="${path}"] .tree-node-content`);
            if (selectedNode) {
                selectedNode.classList.add('selected');
                
                // Expand parents to show selection
                this.expandPathParents(path);
            }
        }
        
        // Update breadcrumbs
        this.updateBreadcrumbs(path);
        
        // Trigger callback
        if (this.onPathChangeCallback) {
            this.onPathChangeCallback(path);
        }
    }

    /**
     * Expand all parent nodes of a given path
     */
    expandPathParents(path) {
        const parts = path.split('/');
        let currentPath = '';
        
        for (let i = 0; i < parts.length - 1; i++) {
            currentPath = currentPath ? `${currentPath}/${parts[i]}` : parts[i];
            this.expandedNodes.add(currentPath);
        }
        
        this.renderTree();
    }

    /**
     * Update breadcrumb navigation
     */
    updateBreadcrumbs(path) {
        const breadcrumbNav = document.getElementById(this.getElementId('breadcrumbNav'));
        if (!breadcrumbNav) return;
        
        const parts = path ? path.split('/') : [];
        let currentPath = '';
        
        const breadcrumbs = [`
            <span class="breadcrumb-item ${!path ? 'active' : ''}" data-path="">
                <i class="fas fa-home"></i> Root
            </span>
        `];
        
        parts.forEach((part, index) => {
            currentPath = currentPath ? `${currentPath}/${part}` : part;
            const isLast = index === parts.length - 1;
            
            if (index > 0) {
                breadcrumbs.push(`<span class="breadcrumb-separator">/</span>`);
            }
            
            breadcrumbs.push(`
                <span class="breadcrumb-item ${isLast ? 'active' : ''}" data-path="${currentPath}">
                    ${part}
                </span>
            `);
        });
        
        breadcrumbNav.innerHTML = breadcrumbs.join('');
    }

    /**
     * Select current input value as path
     */
    selectCurrentInput() {
        const pathInput = document.getElementById(this.getElementId('folderPath'));
        if (pathInput) {
            const path = pathInput.value.trim();
            this.selectPath(path);
        }
    }

    /**
     * Get the currently selected path
     */
    getSelectedPath() {
        return this.selectedPath;
    }

    /**
     * Clear selection
     */
    clearSelection() {
        this.selectPath('');
    }

    /**
     * Clean up event handlers
     */
    destroy() {
        const pathInput = document.getElementById(this.getElementId('folderPath'));
        const createFolderBtn = document.getElementById(this.getElementId('createFolderBtn'));
        const folderTree = document.getElementById(this.getElementId('folderTree'));
        const breadcrumbNav = document.getElementById(this.getElementId('breadcrumbNav'));
        const pathSuggestions = document.getElementById(this.getElementId('pathSuggestions'));

        if (pathInput) {
            pathInput.removeEventListener('input', this.handlePathInput);
            pathInput.removeEventListener('keydown', this.handlePathKeyDown);
        }
        if (createFolderBtn) {
            createFolderBtn.removeEventListener('click', this.handleCreateFolder);
        }
        if (folderTree) {
            folderTree.removeEventListener('click', this.handleTreeClick);
        }
        if (breadcrumbNav) {
            breadcrumbNav.removeEventListener('click', this.handleBreadcrumbClick);
        }
        if (pathSuggestions) {
            pathSuggestions.removeEventListener('click', this.handlePathSuggestionClick);
        }
    }
}
