/**
 * ComfyUI Mobile Interface - Mobile UI Components
 * Handles mobile-specific UI interactions and components
 */

class MobileInterface {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.linearization = new GraphLinearization();
        this.currentWorkflow = null;
        this.linearizedNodes = [];
        this.selectedNode = null;
        this.connectionState = {
            active: false,
            sourceNode: null,
            sourceSocket: null
        };
        
        this.initialize();
    }

    /**
     * Initialize mobile interface
     */
    initialize() {
        this.setupEventListeners();
        this.setupAPIEventListeners();
        this.loadInitialWorkflow();
    }

    /**
     * Setup UI event listeners
     */
    setupEventListeners() {
        // Header actions
        document.getElementById('toggleView').addEventListener('click', () => {
            this.toggleToDesktopView();
        });

        document.getElementById('queuePrompt').addEventListener('click', () => {
            this.queueCurrentWorkflow();
        });

        // Bottom action bar
        document.getElementById('loadWorkflow').addEventListener('click', () => {
            this.showLoadWorkflowDialog();
        });

        document.getElementById('saveWorkflow').addEventListener('click', () => {
            this.showSaveWorkflowDialog();
        });

        document.getElementById('clearWorkflow').addEventListener('click', () => {
            this.clearWorkflow();
        });

        document.getElementById('addNode').addEventListener('click', () => {
            this.showAddNodeDialog();
        });

        // Modal controls
        document.getElementById('closeModal').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('cancelEdit').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('saveEdit').addEventListener('click', () => {
            this.saveNodeEdit();
        });

        // Bottom sheet controls
        document.getElementById('closeBottomSheet').addEventListener('click', () => {
            this.closeBottomSheet();
        });

        // Connection overlay
        document.getElementById('cancelConnection').addEventListener('click', () => {
            this.cancelConnection();
        });

        // Global gestures
        this.setupGlobalGestures();
    }

    /**
     * Setup API event listeners
     */
    setupAPIEventListeners() {
        this.apiClient.on('connected', () => {
            this.updateConnectionStatus('Connected');
            this.enableQueueButton();
        });

        this.apiClient.on('disconnected', () => {
            this.updateConnectionStatus('Disconnected');
            this.disableQueueButton();
        });

        this.apiClient.on('connection_error', (error) => {
            this.updateConnectionStatus('Connection Error');
            this.disableQueueButton();
            Utils.showToast('Connection error: ' + error.message, 'error');
        });

        this.apiClient.on('status_update', (data) => {
            this.updateQueueStatus(data);
        });

        this.apiClient.on('progress_update', (data) => {
            this.updateProgress(data);
        });

        this.apiClient.on('node_executing', (data) => {
            this.updateNodeStatus(data.node, 'executing');
        });

        this.apiClient.on('node_executed', (data) => {
            this.updateNodeStatus(data.node, 'executed');
        });

        this.apiClient.on('execution_start', (data) => {
            this.onExecutionStart(data);
        });

        this.apiClient.on('execution_success', (data) => {
            this.onExecutionSuccess(data);
        });

        this.apiClient.on('execution_error', (data) => {
            this.onExecutionError(data);
        });
    }

    /**
     * Setup global gestures
     */
    setupGlobalGestures() {
        const workflowContainer = document.querySelector('.workflow-container');
        
        // Pull to refresh
        let startY = 0;
        let currentY = 0;
        let pulling = false;

        workflowContainer.addEventListener('touchstart', (e) => {
            if (workflowContainer.scrollTop === 0) {
                startY = e.touches[0].clientY;
                pulling = true;
            }
        });

        workflowContainer.addEventListener('touchmove', (e) => {
            if (pulling) {
                currentY = e.touches[0].clientY;
                const pullDistance = currentY - startY;
                
                if (pullDistance > 50) {
                    // Visual feedback for pull to refresh
                    workflowContainer.style.transform = `translateY(${Math.min(pullDistance - 50, 30)}px)`;
                }
            }
        });

        workflowContainer.addEventListener('touchend', () => {
            if (pulling) {
                const pullDistance = currentY - startY;
                if (pullDistance > 80) {
                    this.refreshWorkflow();
                }
                workflowContainer.style.transform = '';
                pulling = false;
            }
        });
    }

    /**
     * Load initial workflow
     */
    async loadInitialWorkflow() {
        try {
            // Try to load a default workflow or show empty state
            const workflows = await this.apiClient.getWorkflows();
            if (workflows.length > 0) {
                const workflow = await this.apiClient.loadWorkflow(workflows[0]);
                this.setWorkflow(workflow);
            } else {
                this.showEmptyState();
            }
        } catch (error) {
            console.error('Failed to load initial workflow:', error);
            this.showEmptyState();
        }
    }

    /**
     * Set current workflow
     * @param {Object} workflow - Workflow object
     */
    setWorkflow(workflow) {
        this.currentWorkflow = workflow;
        this.linearizedNodes = this.linearization.linearizeWorkflow(workflow);
        this.renderWorkflowPipeline();
        this.updateQueueButton();
    }

    /**
     * Render workflow pipeline
     */
    renderWorkflowPipeline() {
        const pipelineContainer = document.getElementById('workflowPipeline');
        pipelineContainer.innerHTML = '';

        if (this.linearizedNodes.length === 0) {
            this.showEmptyState();
            return;
        }

        this.linearizedNodes.forEach((node, index) => {
            const nodeCard = this.createNodeCard(node, index);
            pipelineContainer.appendChild(nodeCard);
        });
    }

    /**
     * Create node card element
     * @param {Object} node - Node data
     * @param {number} index - Node index
     * @returns {HTMLElement} Node card element
     */
    createNodeCard(node, index) {
        const card = document.createElement('div');
        card.className = 'node-card';
        card.dataset.nodeId = node.id;
        card.dataset.nodeIndex = index;

        card.innerHTML = `
            <div class="node-card-header">
                <div class="node-info">
                    <div class="node-title">${node.title}</div>
                    <div class="node-subtitle">#${node.id} • ${node.type}</div>
                </div>
                <div class="node-actions">
                    <span class="node-status ${node.status}">${node.status}</span>
                    <button class="btn-icon node-menu-btn" data-node-id="${node.id}">
                        <i class="fas fa-ellipsis-v"></i>
                    </button>
                </div>
            </div>
            <div class="node-card-body">
                ${this.renderNodeSockets(node)}
                ${this.renderNodeWidgets(node)}
            </div>
        `;

        // Add event listeners
        this.setupNodeCardListeners(card, node);

        return card;
    }

    /**
     * Render node sockets
     * @param {Object} node - Node data
     * @returns {string} HTML string
     */
    renderNodeSockets(node) {
        if (node.inputs.length === 0 && node.outputs.length === 0) {
            return '';
        }

        return `
            <div class="node-sockets">
                <div class="socket-group">
                    <h4>Inputs</h4>
                    <div class="socket-list">
                        ${node.inputs.map(input => this.renderSocket(input, 'input', node.id)).join('')}
                    </div>
                </div>
                <div class="socket-group">
                    <h4>Outputs</h4>
                    <div class="socket-list">
                        ${node.outputs.map(output => this.renderSocket(output, 'output', node.id)).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render socket
     * @param {Object} socket - Socket data
     * @param {string} type - Socket type ('input' or 'output')
     * @param {string} nodeId - Node ID
     * @returns {string} HTML string
     */
    renderSocket(socket, type, nodeId) {
        const isInput = type === 'input';
        const isConnected = isInput ? socket.connected : false;
        const connectionInfo = isInput && isConnected ? socket.connection : null;

        if (isInput && isConnected) {
            return `
                <div class="connection-reference" 
                     data-source-node="${connectionInfo.sourceNodeId}"
                     data-socket-name="${socket.name}">
                    <div class="socket-name">${socket.name}</div>
                    <div class="socket-connection">
                        → ${connectionInfo.sourceNodeTitle}
                    </div>
                </div>
            `;
        } else {
            return `
                <div class="socket ${isConnected ? 'connected' : ''}" 
                     data-socket-name="${socket.name}"
                     data-socket-type="${socket.type}"
                     data-socket-direction="${type}"
                     data-node-id="${nodeId}">
                    <span class="socket-name">${socket.name}</span>
                    <span class="socket-type">${socket.type}</span>
                </div>
            `;
        }
    }

    /**
     * Render node widgets
     * @param {Object} node - Node data
     * @returns {string} HTML string
     */
    renderNodeWidgets(node) {
        if (node.widgets.length === 0) {
            return '';
        }

        return `
            <div class="node-widgets">
                ${node.widgets.map(widget => `
                    <div class="widget" data-widget-name="${widget.name}">
                        <span class="widget-label">${widget.name}</span>
                        <span class="widget-value">${this.formatWidgetValue(widget)}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Format widget value for display
     * @param {Object} widget - Widget data
     * @returns {string} Formatted value
     */
    formatWidgetValue(widget) {
        if (widget.type === 'number') {
            return Number(widget.value).toFixed(2);
        } else if (widget.type === 'string' && widget.value.length > 20) {
            return widget.value.substring(0, 20) + '...';
        }
        return String(widget.value);
    }

    /**
     * Setup node card event listeners
     * @param {HTMLElement} card - Card element
     * @param {Object} node - Node data
     */
    setupNodeCardListeners(card, node) {
        const header = card.querySelector('.node-card-header');
        const menuBtn = card.querySelector('.node-menu-btn');
        const sockets = card.querySelectorAll('.socket');
        const connectionRefs = card.querySelectorAll('.connection-reference');

        // Toggle expand/collapse
        header.addEventListener('click', (e) => {
            if (e.target.closest('.node-menu-btn')) return;
            this.toggleNodeCard(card);
        });

        // Context menu
        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.showNodeContextMenu(node);
        });

        // Long press for context menu
        Utils.gesture.longPress(header, () => {
            this.showNodeContextMenu(node);
        });

        // Socket interactions
        sockets.forEach(socket => {
            socket.addEventListener('click', (e) => {
                this.handleSocketClick(socket, node);
            });
        });

        // Connection reference navigation
        connectionRefs.forEach(ref => {
            ref.addEventListener('click', (e) => {
                const sourceNodeId = ref.dataset.sourceNode;
                this.scrollToNode(sourceNodeId);
            });
        });

        // Double tap to edit
        let lastTap = 0;
        header.addEventListener('click', (e) => {
            const now = Date.now();
            if (now - lastTap < 300) {
                this.showNodeEditModal(node);
            }
            lastTap = now;
        });
    }

    /**
     * Toggle node card expanded state
     * @param {HTMLElement} card - Card element
     */
    toggleNodeCard(card) {
        card.classList.toggle('expanded');
        
        // Smooth animation
        const body = card.querySelector('.node-card-body');
        if (card.classList.contains('expanded')) {
            body.style.display = 'block';
            body.style.maxHeight = body.scrollHeight + 'px';
        } else {
            body.style.maxHeight = '0px';
            setTimeout(() => {
                body.style.display = 'none';
            }, 300);
        }
    }

    /**
     * Handle socket click
     * @param {HTMLElement} socket - Socket element
     * @param {Object} node - Node data
     */
    handleSocketClick(socket, node) {
        const socketName = socket.dataset.socketName;
        const socketType = socket.dataset.socketType;
        const socketDirection = socket.dataset.socketDirection;

        if (this.connectionState.active) {
            // Complete connection
            this.completeConnection(socket, node);
        } else {
            // Start connection
            this.startConnection(socket, node, socketName, socketType, socketDirection);
        }
    }

    /**
     * Start connection process
     * @param {HTMLElement} socket - Socket element
     * @param {Object} node - Node data
     * @param {string} socketName - Socket name
     * @param {string} socketType - Socket type
     * @param {string} direction - Socket direction
     */
    startConnection(socket, node, socketName, socketType, direction) {
        if (direction !== 'output') return; // Only start from outputs

        this.connectionState = {
            active: true,
            sourceNode: node,
            sourceSocket: {
                name: socketName,
                type: socketType,
                element: socket
            }
        };

        // Visual feedback
        socket.classList.add('connecting');
        document.getElementById('connectionOverlay').classList.add('active');
        
        // Update all compatible sockets
        this.updateSocketCompatibility();

        Utils.showToast('Tap a compatible input to connect', 'info');
    }

    /**
     * Complete connection process
     * @param {HTMLElement} socket - Target socket element
     * @param {Object} node - Target node data
     */
    completeConnection(socket, node) {
        const socketDirection = socket.dataset.socketDirection;
        const socketType = socket.dataset.socketType;

        if (socketDirection !== 'input') {
            Utils.showToast('Can only connect to inputs', 'error');
            return;
        }

        // Check compatibility
        if (!this.linearization.areSocketsCompatible(
            this.connectionState.sourceSocket.type, 
            socketType
        )) {
            Utils.showToast('Incompatible socket types', 'error');
            return;
        }

        // Create connection in workflow
        this.createConnection(
            this.connectionState.sourceNode,
            this.connectionState.sourceSocket,
            node,
            socket.dataset.socketName
        );

        this.cancelConnection();
        Utils.showToast('Connection created', 'success');
    }

    /**
     * Cancel connection process
     */
    cancelConnection() {
        if (this.connectionState.sourceSocket) {
            this.connectionState.sourceSocket.element.classList.remove('connecting');
        }

        this.connectionState = {
            active: false,
            sourceNode: null,
            sourceSocket: null
        };

        document.getElementById('connectionOverlay').classList.remove('active');
        this.clearSocketCompatibility();
    }

    /**
     * Update socket compatibility visual feedback
     */
    updateSocketCompatibility() {
        const allSockets = document.querySelectorAll('.socket');
        const sourceType = this.connectionState.sourceSocket.type;

        allSockets.forEach(socket => {
            const socketType = socket.dataset.socketType;
            const socketDirection = socket.dataset.socketDirection;
            const nodeId = socket.dataset.nodeId;

            if (socketDirection === 'input' && nodeId !== this.connectionState.sourceNode.id) {
                const isCompatible = this.linearization.areSocketsCompatible(sourceType, socketType);
                socket.classList.toggle('compatible', isCompatible);
                socket.classList.toggle('incompatible', !isCompatible);
            }
        });
    }

    /**
     * Clear socket compatibility visual feedback
     */
    clearSocketCompatibility() {
        const allSockets = document.querySelectorAll('.socket');
        allSockets.forEach(socket => {
            socket.classList.remove('compatible', 'incompatible');
        });
    }

    /**
     * Create connection in workflow
     * @param {Object} sourceNode - Source node
     * @param {Object} sourceSocket - Source socket
     * @param {Object} targetNode - Target node
     * @param {string} targetSocketName - Target socket name
     */
    createConnection(sourceNode, sourceSocket, targetNode, targetSocketName) {
        // Update workflow data structure
        if (!this.currentWorkflow.nodes[targetNode.id].inputs) {
            this.currentWorkflow.nodes[targetNode.id].inputs = {};
        }

        this.currentWorkflow.nodes[targetNode.id].inputs[targetSocketName] = [
            sourceNode.id,
            sourceSocket.name
        ];

        // Re-linearize and re-render
        this.linearizedNodes = this.linearization.linearizeWorkflow(this.currentWorkflow);
        this.renderWorkflowPipeline();
    }

    /**
     * Show node context menu
     * @param {Object} node - Node data
     */
    showNodeContextMenu(node) {
        const bottomSheet = document.getElementById('bottomSheet');
        const title = document.getElementById('bottomSheetTitle');
        const body = document.getElementById('bottomSheetBody');

        title.textContent = `${node.title} Actions`;
        body.innerHTML = `
            <button class="bottom-sheet-action" data-action="edit">
                <i class="fas fa-edit"></i>
                Edit Parameters
            </button>
            <button class="bottom-sheet-action" data-action="bypass">
                <i class="fas fa-forward"></i>
                Bypass Node
            </button>
            <button class="bottom-sheet-action" data-action="mute">
                <i class="fas fa-volume-mute"></i>
                Mute Node
            </button>
            <button class="bottom-sheet-action" data-action="clone">
                <i class="fas fa-clone"></i>
                Clone Node
            </button>
            <button class="bottom-sheet-action" data-action="delete">
                <i class="fas fa-trash"></i>
                Delete Node
            </button>
        `;

        // Add action listeners
        body.querySelectorAll('.bottom-sheet-action').forEach(btn => {
            btn.addEventListener('click', () => {
                this.handleNodeAction(btn.dataset.action, node);
                this.closeBottomSheet();
            });
        });

        bottomSheet.classList.add('active');
    }

    /**
     * Handle node action
     * @param {string} action - Action type
     * @param {Object} node - Node data
     */
    handleNodeAction(action, node) {
        switch (action) {
            case 'edit':
                this.showNodeEditModal(node);
                break;
            case 'bypass':
                this.bypassNode(node);
                break;
            case 'mute':
                this.muteNode(node);
                break;
            case 'clone':
                this.cloneNode(node);
                break;
            case 'delete':
                this.deleteNode(node);
                break;
        }
    }

    /**
     * Show node edit modal
     * @param {Object} node - Node data
     */
    showNodeEditModal(node) {
        const modal = document.getElementById('nodeDetailModal');
        const title = document.getElementById('modalTitle');
        const body = document.getElementById('modalBody');

        title.textContent = `Edit ${node.title}`;
        body.innerHTML = this.renderNodeEditForm(node);

        modal.classList.add('active');
        this.selectedNode = node;
    }

    /**
     * Render node edit form
     * @param {Object} node - Node data
     * @returns {string} HTML string
     */
    renderNodeEditForm(node) {
        let html = '';

        node.widgets.forEach(widget => {
            html += `
                <div class="form-group">
                    <label class="form-label">${widget.name}</label>
                    ${this.renderWidgetInput(widget)}
                </div>
            `;
        });

        return html || '<p>No editable parameters</p>';
    }

    /**
     * Render widget input
     * @param {Object} widget - Widget data
     * @returns {string} HTML string
     */
    renderWidgetInput(widget) {
        switch (widget.type) {
            case 'number':
                return `<input type="number" class="form-input" 
                               data-widget-name="${widget.name}"
                               value="${widget.value}"
                               step="0.01">`;
            case 'slider':
                return `<input type="range" class="form-range"
                               data-widget-name="${widget.name}"
                               value="${widget.value}"
                               min="${widget.options.min || 0}"
                               max="${widget.options.max || 100}"
                               step="${widget.options.step || 1}">`;
            case 'text':
                return `<input type="text" class="form-input"
                               data-widget-name="${widget.name}"
                               value="${widget.value}">`;
            case 'textarea':
                return `<textarea class="form-textarea"
                                  data-widget-name="${widget.name}"
                                  rows="4">${widget.value}</textarea>`;
            case 'select':
                const options = widget.options.values || [];
                return `<select class="form-select" data-widget-name="${widget.name}">
                    ${options.map(opt => 
                        `<option value="${opt}" ${opt === widget.value ? 'selected' : ''}>${opt}</option>`
                    ).join('')}
                </select>`;
            default:
                return `<input type="text" class="form-input"
                               data-widget-name="${widget.name}"
                               value="${widget.value}">`;
        }
    }

    /**
     * Save node edit
     */
    saveNodeEdit() {
        if (!this.selectedNode) return;

        const form = document.getElementById('modalBody');
        const inputs = form.querySelectorAll('[data-widget-name]');

        inputs.forEach(input => {
            const widgetName = input.dataset.widgetName;
            const widget = this.selectedNode.widgets.find(w => w.name === widgetName);
            
            if (widget) {
                widget.value = input.value;
                // Update workflow data
                if (this.currentWorkflow.nodes[this.selectedNode.id].widgets_values) {
                    this.currentWorkflow.nodes[this.selectedNode.id].widgets_values[widget.index] = input.value;
                }
            }
        });

        // Re-render the node card
        this.renderWorkflowPipeline();
        this.closeModal();
        
        Utils.showToast('Node updated', 'success');
    }

    /**
     * Close modal
     */
    closeModal() {
        document.getElementById('nodeDetailModal').classList.remove('active');
        this.selectedNode = null;
    }

    /**
     * Close bottom sheet
     */
    closeBottomSheet() {
        document.getElementById('bottomSheet').classList.remove('active');
    }

    /**
     * Scroll to node
     * @param {string} nodeId - Node ID
     */
    scrollToNode(nodeId) {
        const nodeCard = document.querySelector(`[data-node-id="${nodeId}"]`);
        if (nodeCard) {
            Utils.scrollToElement(nodeCard);
            nodeCard.classList.add('fade-in');
            setTimeout(() => nodeCard.classList.remove('fade-in'), 1000);
        }
    }

    /**
     * Queue current workflow
     */
    async queueCurrentWorkflow() {
        if (!this.currentWorkflow) {
            Utils.showToast('No workflow to queue', 'error');
            return;
        }

        try {
            const response = await this.apiClient.queuePrompt(this.currentWorkflow);
            Utils.showToast('Workflow queued successfully', 'success');
            this.updateQueueButton();
        } catch (error) {
            Utils.showToast('Failed to queue workflow: ' + error.message, 'error');
        }
    }

    /**
     * Update connection status
     * @param {string} status - Connection status
     */
    updateConnectionStatus(status) {
        document.getElementById('connectionStatus').textContent = status;
    }

    /**
     * Update queue status
     * @param {Object} data - Queue data
     */
    updateQueueStatus(data) {
        const queueSize = data.exec_info?.queue_remaining || 0;
        document.getElementById('queueSize').textContent = queueSize;
    }

    /**
     * Update queue button state
     */
    updateQueueButton() {
        const button = document.getElementById('queuePrompt');
        const hasWorkflow = this.currentWorkflow && this.linearizedNodes.length > 0;
        const isConnected = this.apiClient.isWebSocketConnected();
        
        button.disabled = !hasWorkflow || !isConnected;
    }

    /**
     * Enable queue button
     */
    enableQueueButton() {
        this.updateQueueButton();
    }

    /**
     * Disable queue button
     */
    disableQueueButton() {
        document.getElementById('queuePrompt').disabled = true;
    }

    /**
     * Show empty state
     */
    showEmptyState() {
        document.getElementById('workflowPipeline').innerHTML = `
            <div class="loading-message">
                <i class="fas fa-plus-circle"></i>
                <p>No workflow loaded</p>
                <button class="btn-primary" onclick="mobileInterface.showLoadWorkflowDialog()">
                    Load Workflow
                </button>
            </div>
        `;
    }

    /**
     * Toggle to desktop view
     */
    toggleToDesktopView() {
        window.location.href = '/';
    }

    /**
     * Show load workflow dialog
     */
    showLoadWorkflowDialog() {
        Utils.showToast('Load workflow feature coming soon', 'info');
    }

    /**
     * Show save workflow dialog
     */
    showSaveWorkflowDialog() {
        Utils.showToast('Save workflow feature coming soon', 'info');
    }

    /**
     * Show add node dialog
     */
    showAddNodeDialog() {
        Utils.showToast('Add node feature coming soon', 'info');
    }

    /**
     * Clear workflow
     */
    clearWorkflow() {
        if (confirm('Are you sure you want to clear the workflow?')) {
            this.currentWorkflow = null;
            this.linearizedNodes = [];
            this.showEmptyState();
            this.updateQueueButton();
            Utils.showToast('Workflow cleared', 'success');
        }
    }

    /**
     * Refresh workflow
     */
    refreshWorkflow() {
        if (this.currentWorkflow) {
            this.linearizedNodes = this.linearization.linearizeWorkflow(this.currentWorkflow);
            this.renderWorkflowPipeline();
            Utils.showToast('Workflow refreshed', 'success');
        }
    }

    /**
     * Update node status
     * @param {string} nodeId - Node ID
     * @param {string} status - New status
     */
    updateNodeStatus(nodeId, status) {
        const nodeCard = document.querySelector(`[data-node-id="${nodeId}"]`);
        if (nodeCard) {
            const statusElement = nodeCard.querySelector('.node-status');
            statusElement.textContent = status;
            statusElement.className = `node-status ${status}`;
        }
    }

    /**
     * Update progress
     * @param {Object} data - Progress data
     */
    updateProgress(data) {
        // Update progress UI if needed
        console.log('Progress update:', data);
    }

    /**
     * Handle execution start
     * @param {Object} data - Execution data
     */
    onExecutionStart(data) {
        Utils.showToast('Execution started', 'info');
    }

    /**
     * Handle execution success
     * @param {Object} data - Execution data
     */
    onExecutionSuccess(data) {
        Utils.showToast('Execution completed successfully', 'success');
        // Reset all node statuses
        this.linearizedNodes.forEach(node => {
            this.updateNodeStatus(node.id, 'idle');
        });
    }

    /**
     * Handle execution error
     * @param {Object} data - Error data
     */
    onExecutionError(data) {
        Utils.showToast('Execution failed: ' + data.exception_message, 'error');
        // Reset all node statuses
        this.linearizedNodes.forEach(node => {
            this.updateNodeStatus(node.id, 'idle');
        });
    }

    // Node manipulation methods (placeholders)
    bypassNode(node) {
        Utils.showToast('Bypass node feature coming soon', 'info');
    }

    muteNode(node) {
        Utils.showToast('Mute node feature coming soon', 'info');
    }

    cloneNode(node) {
        Utils.showToast('Clone node feature coming soon', 'info');
    }

    deleteNode(node) {
        if (confirm(`Are you sure you want to delete ${node.title}?`)) {
            Utils.showToast('Delete node feature coming soon', 'info');
        }
    }
}

// Export for use in other files
window.MobileInterface = MobileInterface;