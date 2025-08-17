import { state, getCurrentPageState } from '../state/index.js';
import { showToast } from './uiHelpers.js';

export class VirtualScroller {
    constructor(options) {
        // Configuration
        this.gridElement = options.gridElement;
        this.createItemFn = options.createItemFn;
        this.fetchItemsFn = options.fetchItemsFn;
        this.overscan = options.overscan || 5; // Extra items to render above/below viewport
        this.containerElement = options.containerElement || this.gridElement.parentElement;
        this.scrollContainer = options.scrollContainer || this.containerElement;
        this.batchSize = options.batchSize || 50;
        this.pageSize = options.pageSize || 100;
        this.itemAspectRatio = 896/1152; // Aspect ratio of cards
        this.rowGap = options.rowGap || 20; // Add vertical gap between rows (default 20px)
        
        // Add container padding properties
        this.containerPaddingTop = options.containerPaddingTop || 4; // Default top padding from CSS
        this.containerPaddingBottom = options.containerPaddingBottom || 4; // Default bottom padding from CSS
        
        // Add data windowing enable/disable flag
        this.enableDataWindowing = options.enableDataWindowing !== undefined ? options.enableDataWindowing : false;

        // State
        this.items = []; // All items metadata
        this.renderedItems = new Map(); // Map of rendered DOM elements by index
        this.totalItems = 0;
        this.isLoading = false;
        this.hasMore = true;
        this.lastScrollTop = 0;
        this.scrollDirection = 'down';
        this.lastRenderRange = { start: 0, end: 0 };
        this.pendingScroll = null;
        this.resizeObserver = null;

        // Data windowing parameters
        this.windowSize = options.windowSize || 2000; // ±1000 items from current view
        this.windowPadding = options.windowPadding || 500; // Buffer before loading more
        this.dataWindow = { start: 0, end: 0 }; // Current data window indices
        this.absoluteWindowStart = 0; // Start index in absolute terms
        this.fetchingWindow = false; // Flag to track window fetching state

        // Responsive layout state
        this.itemWidth = 0;
        this.itemHeight = 0;
        this.columnsCount = 0;
        this.gridPadding = 12; // Gap between cards
        this.columnGap = 12; // Horizontal gap

        // Add loading timeout state
        this.loadingTimeout = null;
        this.loadingTimeoutDuration = options.loadingTimeoutDuration || 15000; // 15 seconds default

        // Initialize
        this.initializeContainer();
        this.setupEventListeners();
        this.calculateLayout();
    }

    initializeContainer() {
        // Add virtual scroll class to grid
        this.gridElement.classList.add('virtual-scroll');

        // Set the container to have relative positioning
        if (getComputedStyle(this.containerElement).position === 'static') {
            this.containerElement.style.position = 'relative';
        }

        // Create a spacer element with the total height
        this.spacerElement = document.createElement('div');
        this.spacerElement.className = 'virtual-scroll-spacer';
        this.spacerElement.style.width = '100%';
        this.spacerElement.style.height = '0px'; // Will be updated as items are loaded
        this.spacerElement.style.pointerEvents = 'none';
        
        // The grid will be used for the actual visible items
        this.gridElement.style.position = 'relative';
        this.gridElement.style.minHeight = '0';
        
        // Apply padding directly to ensure consistency
        this.gridElement.style.paddingTop = `${this.containerPaddingTop}px`;
        this.gridElement.style.paddingBottom = `${this.containerPaddingBottom}px`;
        
        // Place the spacer inside the grid container
        this.gridElement.appendChild(this.spacerElement);
    }

    calculateLayout() {
        const pageState = getCurrentPageState();
        if (pageState.duplicatesMode) {
            return false
        }

        // Get container width and style information
        const containerWidth = this.containerElement.clientWidth;
        const containerStyle = getComputedStyle(this.containerElement);
        const paddingLeft = parseInt(containerStyle.paddingLeft, 10) || 0;
        const paddingRight = parseInt(containerStyle.paddingRight, 10) || 0;
        
        // Calculate available content width (excluding padding)
        const availableContentWidth = containerWidth - paddingLeft - paddingRight;
        
        // Get display density setting
        const displayDensity = state.global.settings?.displayDensity || 'default';
        
        // Set exact column counts and grid widths to match CSS container widths
        let maxColumns, maxGridWidth;
        
        // Match exact column counts and CSS container width values based on density
        if (window.innerWidth >= 3000) { // 4K
            if (displayDensity === 'default') {
                maxColumns = 8;
            } else if (displayDensity === 'medium') {
                maxColumns = 9;
            } else { // compact
                maxColumns = 10;
            }
            maxGridWidth = 2400; // Match exact CSS container width for 4K
        } else if (window.innerWidth >= 2000) { // 2K/1440p
            if (displayDensity === 'default') {
                maxColumns = 6;
            } else if (displayDensity === 'medium') {
                maxColumns = 7;
            } else { // compact
                maxColumns = 8;
            }
            maxGridWidth = 1800; // Match exact CSS container width for 2K
        } else {
            // 1080p
            if (displayDensity === 'default') {
                maxColumns = 5;
            } else if (displayDensity === 'medium') {
                maxColumns = 6;
            } else { // compact
                maxColumns = 7;
            }
            maxGridWidth = 1400; // Match exact CSS container width for 1080p
        }
        
        // Calculate baseCardWidth based on desired column count and available space
        // Formula: (maxGridWidth - (columns-1)*gap) / columns
        const baseCardWidth = (maxGridWidth - ((maxColumns - 1) * this.columnGap)) / maxColumns;
        
        // Use the smaller of available content width or max grid width
        const actualGridWidth = Math.min(availableContentWidth, maxGridWidth);
        
        // Set exact column count based on screen size and mode
        this.columnsCount = maxColumns;
        
        // When available width is smaller than maxGridWidth, recalculate columns
        if (availableContentWidth < maxGridWidth) {
            // Calculate how many columns can fit in the available space
            this.columnsCount = Math.max(1, Math.floor(
                (availableContentWidth + this.columnGap) / (baseCardWidth + this.columnGap)
            ));
        }
        
        // Calculate actual item width
        this.itemWidth = (actualGridWidth - (this.columnsCount - 1) * this.columnGap) / this.columnsCount;
        
        // Calculate height based on aspect ratio
        this.itemHeight = this.itemWidth / this.itemAspectRatio;
        
        // Calculate the left offset to center the grid within the content area
        this.leftOffset = Math.max(0, (availableContentWidth - actualGridWidth) / 2);

        // Update grid element max-width to match available width
        this.gridElement.style.maxWidth = `${actualGridWidth}px`;
        
        // Add or remove density classes for style adjustments
        this.gridElement.classList.remove('default-density', 'medium-density', 'compact-density');
        this.gridElement.classList.add(`${displayDensity}-density`);
        
        // Update spacer height
        this.updateSpacerHeight();
        
        // Re-render with new layout
        this.clearRenderedItems();
        this.scheduleRender();
        
        return true;
    }

    setupEventListeners() {
        // Debounced scroll handler
        this.scrollHandler = this.debounce(() => this.handleScroll(), 10);
        this.scrollContainer.addEventListener('scroll', this.scrollHandler);
        
        // Window resize handler for layout recalculation
        this.resizeHandler = this.debounce(() => {
            this.calculateLayout();
        }, 150);
        
        window.addEventListener('resize', this.resizeHandler);
        
        // Use ResizeObserver for more accurate container size detection
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(this.debounce(() => {
                this.calculateLayout();
            }, 150));
            
            this.resizeObserver.observe(this.containerElement);
        }
    }

    async initialize() {
        try {
            await this.loadInitialBatch();
            this.scheduleRender();
        } catch (err) {
            console.error('Failed to initialize virtual scroller:', err);
            showToast('Failed to load items', 'error');
        }
    }

    async loadInitialBatch() {
        const pageState = getCurrentPageState();
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.setLoadingTimeout(); // Add loading timeout safety
        
        try {
            const { items, totalItems, hasMore } = await this.fetchItemsFn(1, this.pageSize);
            
            // Initialize the data window with the first batch of items
            this.items = items || [];
            this.totalItems = totalItems || 0;
            this.hasMore = hasMore;
            this.dataWindow = { start: 0, end: this.items.length };
            this.absoluteWindowStart = 0;
            
            // Update the spacer height based on the total number of items
            this.updateSpacerHeight();
            
            // Check if there are no items and show placeholder if needed
            if (this.items.length === 0) {
                this.showNoItemsPlaceholder();
            } else {
                this.removeNoItemsPlaceholder();
            }
            
            // Reset page state to sync with our virtual scroller
            pageState.currentPage = 2; // Next page to load would be 2
            pageState.hasMore = this.hasMore;
            pageState.isLoading = false;
            
            return { items, totalItems, hasMore };
        } catch (err) {
            console.error('Failed to load initial batch:', err);
            this.showNoItemsPlaceholder('Failed to load items. Please try refreshing the page.');
            throw err;
        } finally {
            this.isLoading = false;
            this.clearLoadingTimeout(); // Clear the timeout
        }
    }

    async loadMoreItems() {
        const pageState = getCurrentPageState();
        if (this.isLoading || !this.hasMore) return;
        
        this.isLoading = true;
        pageState.isLoading = true;
        this.setLoadingTimeout(); // Add loading timeout safety
        
        try {
            console.log('Loading more items, page:', pageState.currentPage);
            const { items, hasMore } = await this.fetchItemsFn(pageState.currentPage, this.pageSize);
            
            if (items && items.length > 0) {
                this.items = [...this.items, ...items];
                this.hasMore = hasMore;
                pageState.hasMore = hasMore;
                
                // Update page for next request
                pageState.currentPage++;
                
                // Update the spacer height
                this.updateSpacerHeight();
                
                // Render the newly loaded items if they're in view
                this.scheduleRender();
                
                console.log(`Loaded ${items.length} more items, total now: ${this.items.length}`);
            } else {
                this.hasMore = false;
                pageState.hasMore = false;
                console.log('No more items to load');
            }
            
            return items;
        } catch (err) {
            console.error('Failed to load more items:', err);
            showToast('Failed to load more items', 'error');
        } finally {
            this.isLoading = false;
            pageState.isLoading = false;
            this.clearLoadingTimeout(); // Clear the timeout
        }
    }

    // Add new methods for loading timeout
    setLoadingTimeout() {
        // Clear any existing timeout first
        this.clearLoadingTimeout();
        
        // Set a new timeout to prevent loading state from getting stuck
        this.loadingTimeout = setTimeout(() => {
            if (this.isLoading) {
                console.warn('Loading timeout occurred. Resetting loading state.');
                this.isLoading = false;
                const pageState = getCurrentPageState();
                pageState.isLoading = false;
            }
        }, this.loadingTimeoutDuration);
    }

    clearLoadingTimeout() {
        if (this.loadingTimeout) {
            clearTimeout(this.loadingTimeout);
            this.loadingTimeout = null;
        }
    }

    updateSpacerHeight() {
        if (this.columnsCount === 0) return;
        
        // Calculate total rows needed based on total items and columns
        const totalRows = Math.ceil(this.totalItems / this.columnsCount);
        // Add row gaps to the total height calculation
        const totalHeight = totalRows * this.itemHeight + (totalRows - 1) * this.rowGap;
        
        // Include container padding in the total height
        const spacerHeight = totalHeight + this.containerPaddingTop + this.containerPaddingBottom;
        
        // Update spacer height to represent all items
        this.spacerElement.style.height = `${spacerHeight}px`;
    }

    getVisibleRange() {
        const scrollTop = this.scrollContainer.scrollTop;
        const viewportHeight = this.scrollContainer.clientHeight;
        
        // Calculate the visible row range, accounting for row gaps
        const rowHeight = this.itemHeight + this.rowGap;
        const startRow = Math.floor(scrollTop / rowHeight);
        const endRow = Math.ceil((scrollTop + viewportHeight) / rowHeight);
        
        // Add overscan for smoother scrolling
        const overscanRows = this.overscan;
        const firstRow = Math.max(0, startRow - overscanRows);
        const lastRow = Math.min(Math.ceil(this.totalItems / this.columnsCount), endRow + overscanRows);
        
        // Calculate item indices
        const firstIndex = firstRow * this.columnsCount;
        const lastIndex = Math.min(this.totalItems, lastRow * this.columnsCount);
        
        return { start: firstIndex, end: lastIndex };
    }

    // Update the scheduleRender method to check for disabled state
    scheduleRender() {
        if (this.disabled || this.renderScheduled) return;
        
        this.renderScheduled = true;
        requestAnimationFrame(() => {
            this.renderItems();
            this.renderScheduled = false;
        });
    }

    // Update the renderItems method to check for disabled state
    renderItems() {
        if (this.disabled || this.items.length === 0 || this.columnsCount === 0) return;
        
        const { start, end } = this.getVisibleRange();
        
        // Check if render range has significantly changed
        const isSameRange = 
            start >= this.lastRenderRange.start && 
            end <= this.lastRenderRange.end &&
            Math.abs(start - this.lastRenderRange.start) < 10;
            
        if (isSameRange) return;
        
        this.lastRenderRange = { start, end };
        
        // Determine which items need to be added and removed
        const currentIndices = new Set();
        for (let i = start; i < end && i < this.items.length; i++) {
            currentIndices.add(i);
        }
        
        // Remove items that are no longer visible
        for (const [index, element] of this.renderedItems.entries()) {
            if (!currentIndices.has(index)) {
                element.remove();
                this.renderedItems.delete(index);
            }
        }
        
        // Use DocumentFragment for batch DOM operations
        const fragment = document.createDocumentFragment();
        
        // Add new visible items to the fragment
        for (let i = start; i < end && i < this.items.length; i++) {
            if (!this.renderedItems.has(i)) {
                const item = this.items[i];
                const element = this.createItemElement(item, i);
                fragment.appendChild(element);
                this.renderedItems.set(i, element);
            }
        }
        
        // Add the fragment to the grid (single DOM operation)
        if (fragment.childNodes.length > 0) {
            this.gridElement.appendChild(fragment);
        }
        
        // If we're close to the end and have more items to load, fetch them
        if (end > this.items.length - (this.columnsCount * 2) && this.hasMore && !this.isLoading) {
            this.loadMoreItems();
        }
        
        // Check if we need to slide the data window
        this.slideDataWindow();
    }

    clearRenderedItems() {
        this.renderedItems.forEach(element => element.remove());
        this.renderedItems.clear();
        this.lastRenderRange = { start: 0, end: 0 };
    }

    refreshWithData(items, totalItems, hasMore) {
        this.items = items || [];
        this.totalItems = totalItems || 0;
        this.hasMore = hasMore;
        this.updateSpacerHeight();
        
        // Check if there are no items and show placeholder if needed
        if (this.items.length === 0) {
            this.showNoItemsPlaceholder();
        } else {
            this.removeNoItemsPlaceholder();
        }
        
        // Clear all rendered items and redraw
        this.clearRenderedItems();
        this.scheduleRender();
    }

    createItemElement(item, index) {
        // Create the DOM element
        const element = this.createItemFn(item);
        
        // Add virtual scroll item class
        element.classList.add('virtual-scroll-item');
        
        // Calculate the position
        const row = Math.floor(index / this.columnsCount);
        const col = index % this.columnsCount;
        
        // Calculate precise positions with row gap included
        // Add the top padding to account for container padding
        const topPos = this.containerPaddingTop + (row * (this.itemHeight + this.rowGap));
        
        // Position correctly with leftOffset (no need to add padding as absolute
        // positioning is already relative to the padding edge of the container)
        const leftPos = this.leftOffset + (col * (this.itemWidth + this.columnGap));
        
        // Position the element with absolute positioning
        element.style.position = 'absolute';
        element.style.left = `${leftPos}px`;
        element.style.top = `${topPos}px`;
        element.style.width = `${this.itemWidth}px`;
        element.style.height = `${this.itemHeight}px`;
        
        return element;
    }

    handleScroll() {
        // Determine scroll direction
        const scrollTop = this.scrollContainer.scrollTop;
        this.scrollDirection = scrollTop > this.lastScrollTop ? 'down' : 'up';
        this.lastScrollTop = scrollTop;
        
        // Handle large jumps in scroll position - check if we need to fetch a new window
        const { scrollHeight } = this.scrollContainer;
        const scrollRatio = scrollTop / scrollHeight;
        
        // Only perform data windowing if the feature is enabled
        if (this.enableDataWindowing && this.totalItems > this.windowSize) {
            const estimatedIndex = Math.floor(scrollRatio * this.totalItems);
            const currentWindowStart = this.absoluteWindowStart;
            const currentWindowEnd = currentWindowStart + this.items.length;
            
            // If the estimated position is outside our current window by a significant amount
            if (estimatedIndex < currentWindowStart || estimatedIndex > currentWindowEnd) {
                // Fetch a new data window centered on the estimated position
                this.fetchDataWindow(Math.max(0, estimatedIndex - Math.floor(this.windowSize / 2)));
                return; // Skip normal rendering until new data is loaded
            }
        }
        
        // Render visible items
        this.scheduleRender();
        
        // If we're near the bottom and have more items, load them
        const { clientHeight } = this.scrollContainer;
        const scrollBottom = scrollTop + clientHeight;
        
        // Fix the threshold calculation - use percentage of remaining height instead
        // We'll trigger loading when within 20% of the bottom of rendered content
        const remainingScroll = scrollHeight - scrollBottom;
        const scrollThreshold = Math.min(
            // Either trigger when within 20% of the total height from bottom
            scrollHeight * 0.2,
            // Or when within 2 rows of content from the bottom, whichever is larger
            (this.itemHeight + this.rowGap) * 2
        );
        
        const shouldLoadMore = remainingScroll <= scrollThreshold;
        
        if (shouldLoadMore && this.hasMore && !this.isLoading) {
            this.loadMoreItems();
        }
    }

    // Method to fetch data for a specific window position
    async fetchDataWindow(targetIndex) {
        // Skip if data windowing is disabled or already fetching
        if (!this.enableDataWindowing || this.fetchingWindow) return;
        
        this.fetchingWindow = true;
        
        try {
            // Calculate which page we need to fetch based on target index
            const targetPage = Math.floor(targetIndex / this.pageSize) + 1;
            console.log(`Fetching data window for index ${targetIndex}, page ${targetPage}`);
            
            const { items, totalItems, hasMore } = await this.fetchItemsFn(targetPage, this.pageSize);
            
            if (items && items.length > 0) {
                // Calculate new absolute window start
                this.absoluteWindowStart = (targetPage - 1) * this.pageSize;
                
                // Replace the entire data window with new items
                this.items = items;
                this.dataWindow = { 
                    start: 0,
                    end: items.length
                };
                
                this.totalItems = totalItems || 0;
                this.hasMore = hasMore;
                
                // Update the current page for future fetches
                const pageState = getCurrentPageState();
                pageState.currentPage = targetPage + 1;
                pageState.hasMore = hasMore;
                
                // Update the spacer height and clear current rendered items
                this.updateSpacerHeight();
                this.clearRenderedItems();
                this.scheduleRender();
                
                console.log(`Loaded ${items.length} items for window at absolute index ${this.absoluteWindowStart}`);
            }
        } catch (err) {
            console.error('Failed to fetch data window:', err);
            showToast('Failed to load items at this position', 'error');
        } finally {
            this.fetchingWindow = false;
        }
    }

    // Method to slide the data window if we're approaching its edges
    async slideDataWindow() {
        // Skip if data windowing is disabled
        if (!this.enableDataWindowing) return;
        
        const { start, end } = this.getVisibleRange();
        const windowStart = this.dataWindow.start;
        const windowEnd = this.dataWindow.end;
        const absoluteIndex = this.absoluteWindowStart + windowStart;
        
        // Calculate the midpoint of the visible range
        const visibleMidpoint = Math.floor((start + end) / 2);
        const absoluteMidpoint = this.absoluteWindowStart + visibleMidpoint;
        
        // Check if we're too close to the window edges
        const closeToStart = start - windowStart < this.windowPadding;
        const closeToEnd = windowEnd - end < this.windowPadding;
        
        // If we're close to either edge and have total items > window size
        if ((closeToStart || closeToEnd) && this.totalItems > this.windowSize) {
            // Calculate a new target index centered around the current viewport
            const halfWindow = Math.floor(this.windowSize / 2);
            const targetIndex = Math.max(0, absoluteMidpoint - halfWindow);
            
            // Don't fetch a new window if we're already showing items near the beginning
            if (targetIndex === 0 && this.absoluteWindowStart === 0) {
                return;
            }
            
            // Don't fetch if we're showing the end of the list and are near the end
            if (this.absoluteWindowStart + this.items.length >= this.totalItems && 
                this.totalItems - end < halfWindow) {
                return;
            }
            
            // Fetch the new data window
            await this.fetchDataWindow(targetIndex);
        }
    }

    reset() {
        // Remove all rendered items
        this.clearRenderedItems();
        
        // Reset state
        this.items = [];
        this.totalItems = 0;
        this.hasMore = true;
        
        // Reset spacer height
        this.spacerElement.style.height = '0px';
        
        // Remove any placeholder
        this.removeNoItemsPlaceholder();
        
        // Schedule a re-render
        this.scheduleRender();
    }

    dispose() {
        // Remove event listeners
        this.scrollContainer.removeEventListener('scroll', this.scrollHandler);
        window.removeEventListener('resize', this.resizeHandler);
        
        // Clean up the resize observer if present
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        // Remove rendered elements
        this.clearRenderedItems();
        
        // Remove spacer
        this.spacerElement.remove();
        
        // Remove virtual scroll class
        this.gridElement.classList.remove('virtual-scroll');
        
        // Clear any pending timeout
        this.clearLoadingTimeout();
    }

    // Add methods to handle placeholder display
    showNoItemsPlaceholder(message) {
        // Remove any existing placeholder first
        this.removeNoItemsPlaceholder();
        
        // Create placeholder message
        const placeholder = document.createElement('div');
        placeholder.className = 'placeholder-message';
        
        // Determine appropriate message based on page type
        let placeholderText = '';
        
        if (message) {
            placeholderText = message;
        } else {
            const pageType = state.currentPageType;
            
            if (pageType === 'recipes') {
                placeholderText = `
                    <p>No recipes found</p>
                    <p>Add recipe images to your recipes folder to see them here.</p>
                `;
            } else if (pageType === 'loras') {
                placeholderText = `
                    <p>No LoRAs found</p>
                    <p>Add LoRAs to your models folder to see them here.</p>
                `;
            } else if (pageType === 'checkpoints') {
                placeholderText = `
                    <p>No checkpoints found</p>
                    <p>Add checkpoints to your models folder to see them here.</p>
                `;
            } else {
                placeholderText = `
                    <p>No items found</p>
                    <p>Try adjusting your search filters or add more content.</p>
                `;
            }
        }
        
        placeholder.innerHTML = placeholderText;
        placeholder.id = 'virtualScrollPlaceholder';
        
        // Append placeholder to the grid
        this.gridElement.appendChild(placeholder);
    }

    removeNoItemsPlaceholder() {
        const placeholder = document.getElementById('virtualScrollPlaceholder');
        if (placeholder) {
            placeholder.remove();
        }
    }

    // Utility method for debouncing
    debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }

    // Add disable method to stop rendering and events
    disable() {
        // Detach scroll event listener
        this.scrollContainer.removeEventListener('scroll', this.scrollHandler);
        
        // Clear all rendered items from the DOM
        this.clearRenderedItems();
        
        // Hide the spacer element
        if (this.spacerElement) {
            this.spacerElement.style.display = 'none';
        }
        
        // Flag as disabled
        this.disabled = true;
        
        console.log('Virtual scroller disabled');
    }

    // Add enable method to resume rendering and events
    enable() {
        if (!this.disabled) return;
        
        // Reattach scroll event listener
        this.scrollContainer.addEventListener('scroll', this.scrollHandler);
        
        // Check if spacer element exists in the DOM, if not, recreate it
        if (!this.spacerElement || !this.gridElement.contains(this.spacerElement)) {
            console.log('Spacer element not found in DOM, recreating it');
            
            // Create a new spacer element
            this.spacerElement = document.createElement('div');
            this.spacerElement.className = 'virtual-scroll-spacer';
            this.spacerElement.style.width = '100%';
            this.spacerElement.style.height = '0px';
            this.spacerElement.style.pointerEvents = 'none';
            
            // Append it to the grid
            this.gridElement.appendChild(this.spacerElement);
            
            // Update the spacer height
            this.updateSpacerHeight();
        } else {
            // Show the spacer element if it exists
            this.spacerElement.style.display = 'block';
        }
        
        // Flag as enabled
        this.disabled = false;
        
        // Re-render items
        this.scheduleRender();
        
        console.log('Virtual scroller enabled');
    }

    // Helper function for deep merging objects
    deepMerge(target, source) {
        if (!source) return target;
        
        const result = { ...target };
        
        Object.keys(source).forEach(key => {
            if (source[key] !== null && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                // If property exists in target and is an object, recursively merge
                if (target[key] && typeof target[key] === 'object' && !Array.isArray(target[key])) {
                    result[key] = this.deepMerge(target[key], source[key]);
                } else {
                    // Otherwise just assign the source value
                    result[key] = source[key];
                }
            } else {
                // For non-objects (including arrays), just assign the value
                result[key] = source[key];
            }
        });
        
        return result;
    }

    updateSingleItem(filePath, updatedItem) {
        if (!filePath || !updatedItem) {
            console.error('Invalid parameters for updateSingleItem');
            return false;
        }

        // Find the index of the item with the matching file_path
        const index = this.items.findIndex(item => item.file_path === filePath);
        if (index === -1) {
            console.warn(`Item with file path ${filePath} not found in virtual scroller data`);
            return false;
        }

        // Update the item data using deep merge
        this.items[index] = this.deepMerge(this.items[index], updatedItem);
        
        // If the item is currently rendered, update its DOM representation
        if (this.renderedItems.has(index)) {
            const element = this.renderedItems.get(index);
            
            // Remove the old element
            element.remove();
            this.renderedItems.delete(index);
            
            // Create and render the updated element
            const updatedElement = this.createItemElement(this.items[index], index);
            
            // Add update indicator visual effects
            updatedElement.classList.add('updated');
            
            // Add temporary update tag
            const updateIndicator = document.createElement('div');
            updateIndicator.className = 'update-indicator';
            updateIndicator.textContent = 'Updated';
            updatedElement.querySelector('.card-preview').appendChild(updateIndicator);
            
            // Automatically remove the updated class after animation completes
            setTimeout(() => {
                updatedElement.classList.remove('updated');
            }, 1500);
            
            // Automatically remove the indicator after animation completes
            setTimeout(() => {
                if (updateIndicator && updateIndicator.parentNode) {
                    updateIndicator.remove();
                }
            }, 2000);
            
            this.renderedItems.set(index, updatedElement);
            this.gridElement.appendChild(updatedElement);
        }
        
        return true;
    }

    // New method to remove an item by file path
    removeItemByFilePath(filePath) {
        if (!filePath || this.disabled || this.items.length === 0) return false;

        // Find the index of the item with the matching file path
        const index = this.items.findIndex(item => item.file_path === filePath);

        if (index === -1) {
            console.warn(`Item with file path ${filePath} not found in virtual scroller data`);
            return false;
        }

        // Remove the item from the data array
        this.items.splice(index, 1);
        
        // Decrement total count
        this.totalItems = Math.max(0, this.totalItems - 1);
        
        // Remove the item from rendered items if it exists
        if (this.renderedItems.has(index)) {
            this.renderedItems.get(index).remove();
            this.renderedItems.delete(index);
        }
        
        // Shift all rendered items with higher indices down by 1
        const indicesToUpdate = [];
        
        // Collect all indices that need to be updated
        for (const [idx, element] of this.renderedItems.entries()) {
            if (idx > index) {
                indicesToUpdate.push(idx);
            }
        }
        
        // Update the elements and map entries
        for (const idx of indicesToUpdate) {
            const element = this.renderedItems.get(idx);
            this.renderedItems.delete(idx);
            // The item is now at the previous index
            this.renderedItems.set(idx - 1, element);
        }
        
        // Update the spacer height to reflect the new total
        this.updateSpacerHeight();
        
        // Re-render to ensure proper layout
        this.clearRenderedItems();
        this.scheduleRender();
        
        console.log(`Removed item with file path ${filePath} from virtual scroller data`);
        return true;
    }

    // Add keyboard navigation methods
    handlePageUpDown(direction) {
        // Prevent duplicate animations by checking last trigger time
        const now = Date.now();
        if (this.lastPageNavTime && now - this.lastPageNavTime < 300) {
            return; // Ignore rapid repeated triggers
        }
        this.lastPageNavTime = now;
        
        const scrollContainer = this.scrollContainer;
        const viewportHeight = scrollContainer.clientHeight;
        
        // Calculate scroll distance (one viewport minus 10% overlap for context)
        const scrollDistance = viewportHeight * 0.9;
        
        // Determine the new scroll position
        const newScrollTop = scrollContainer.scrollTop + (direction === 'down' ? scrollDistance : -scrollDistance);
        
        // Remove any existing transition indicators
        this.removeExistingTransitionIndicator();
        
        // Scroll to the new position with smooth animation
        scrollContainer.scrollTo({
            top: newScrollTop,
            behavior: 'smooth'
        });
        
        // Page transition indicator removed
        // this.showTransitionIndicator();
        
        // Force render after scrolling
        setTimeout(() => this.renderItems(), 100);
        setTimeout(() => this.renderItems(), 300);
    }

    // Helper to remove existing indicators
    removeExistingTransitionIndicator() {
        const existingIndicator = document.querySelector('.page-transition-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
    }

    scrollToTop() {
        this.removeExistingTransitionIndicator();
        
        // Page transition indicator removed
        // this.showTransitionIndicator();
        
        this.scrollContainer.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
        
        // Force render after scrolling
        setTimeout(() => this.renderItems(), 100);
    }

    scrollToBottom() {
        this.removeExistingTransitionIndicator();
        
        // Page transition indicator removed
        // this.showTransitionIndicator();
        
        // Start loading all remaining pages to ensure content is available
        this.loadRemainingPages().then(() => {
            // After loading all content, scroll to the very bottom
            const maxScroll = this.scrollContainer.scrollHeight - this.scrollContainer.clientHeight;
            this.scrollContainer.scrollTo({
                top: maxScroll,
                behavior: 'smooth'
            });
        });
    }
    
    // New method to load all remaining pages
    async loadRemainingPages() {
        // If we're already at the end or loading, don't proceed
        if (!this.hasMore || this.isLoading) return;
        
        console.log('Loading all remaining pages for End key navigation...');
        
        // Keep loading pages until we reach the end
        while (this.hasMore && !this.isLoading) {
            await this.loadMoreItems();
            
            // Force render after each page load
            this.renderItems();
            
            // Small delay to prevent overwhelming the browser
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        console.log('Finished loading all pages');
        
        // Final render to ensure all content is displayed
        this.renderItems();
    }
}
