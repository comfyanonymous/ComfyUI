/**
 * ComfyUI Mobile Interface - API Client
 * Handles communication with ComfyUI backend
 */

class ComfyUIAPIClient {
    constructor() {
        this.baseURL = window.location.origin;
        this.websocket = null;
        this.clientId = this.generateClientId();
        this.isConnected = false;
        this.eventListeners = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // Initialize WebSocket connection
        this.initializeWebSocket();
    }

    /**
     * Generate unique client ID
     * @returns {string} Client ID
     */
    generateClientId() {
        return 'mobile_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Initialize WebSocket connection
     */
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsURL = `${protocol}//${window.location.host}/ws?clientId=${this.clientId}`;
        
        try {
            this.websocket = new WebSocket(wsURL);
            this.setupWebSocketHandlers();
        } catch (error) {
            console.error('WebSocket initialization error:', error);
            this.emit('connection_error', error);
        }
    }

    /**
     * Setup WebSocket event handlers
     */
    setupWebSocketHandlers() {
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.emit('connected');
        };

        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('WebSocket message parse error:', error);
            }
        };

        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.emit('disconnected');
            
            // Attempt to reconnect if not a clean close
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => this.reconnect(), this.reconnectDelay);
            }
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('connection_error', error);
        };
    }

    /**
     * Handle incoming WebSocket messages
     * @param {Object} data - Message data
     */
    handleWebSocketMessage(data) {
        const { type, data: messageData } = data;
        
        switch (type) {
            case 'status':
                this.emit('status_update', messageData);
                break;
            case 'progress':
                this.emit('progress_update', messageData);
                break;
            case 'executing':
                this.emit('node_executing', messageData);
                break;
            case 'executed':
                this.emit('node_executed', messageData);
                break;
            case 'execution_start':
                this.emit('execution_start', messageData);
                break;
            case 'execution_success':
                this.emit('execution_success', messageData);
                break;
            case 'execution_error':
                this.emit('execution_error', messageData);
                break;
            case 'execution_cached':
                this.emit('execution_cached', messageData);
                break;
            default:
                console.log('Unknown WebSocket message type:', type, messageData);
        }
    }

    /**
     * Reconnect to WebSocket
     */
    reconnect() {
        this.reconnectAttempts++;
        console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        this.initializeWebSocket();
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Exponential backoff
    }

    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }

    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    off(event, callback) {
        if (!this.eventListeners.has(event)) return;
        
        const callbacks = this.eventListeners.get(event);
        const index = callbacks.indexOf(callback);
        if (index > -1) {
            callbacks.splice(index, 1);
        }
    }

    /**
     * Emit event to listeners
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    emit(event, data) {
        if (!this.eventListeners.has(event)) return;
        
        this.eventListeners.get(event).forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Event callback error:', error);
            }
        });
    }

    /**
     * Make HTTP API request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise} Response promise
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API request error:', error);
            throw error;
        }
    }

    /**
     * Get system information
     * @returns {Promise<Object>} System info
     */
    async getSystemInfo() {
        return await this.request('/system_stats');
    }

    /**
     * Get available node types
     * @returns {Promise<Object>} Node types
     */
    async getNodeTypes() {
        return await this.request('/object_info');
    }

    /**
     * Get queue status
     * @returns {Promise<Object>} Queue status
     */
    async getQueueStatus() {
        return await this.request('/queue');
    }

    /**
     * Get execution history
     * @returns {Promise<Object>} Execution history
     */
    async getHistory() {
        return await this.request('/history');
    }

    /**
     * Queue a workflow prompt
     * @param {Object} workflow - Workflow object
     * @param {Object} options - Queue options
     * @returns {Promise<Object>} Queue response
     */
    async queuePrompt(workflow, options = {}) {
        const payload = {
            prompt: workflow,
            client_id: this.clientId,
            extra_data: options.extra_data || {}
        };

        if (options.number !== undefined) {
            payload.number = options.number;
        }

        if (options.front !== undefined) {
            payload.front = options.front;
        }

        return await this.request('/prompt', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * Cancel all queued prompts
     * @returns {Promise<Object>} Cancel response
     */
    async cancelQueue() {
        return await this.request('/queue', {
            method: 'POST',
            body: JSON.stringify({ delete: ['*'] })
        });
    }

    /**
     * Cancel specific prompt
     * @param {string} promptId - Prompt ID to cancel
     * @returns {Promise<Object>} Cancel response
     */
    async cancelPrompt(promptId) {
        return await this.request('/queue', {
            method: 'POST',
            body: JSON.stringify({ delete: [promptId] })
        });
    }

    /**
     * Get available models
     * @returns {Promise<Object>} Available models
     */
    async getModels() {
        return await this.request('/api/models');
    }

    /**
     * Upload file
     * @param {File} file - File to upload
     * @param {string} type - File type (input, temp, etc.)
     * @returns {Promise<Object>} Upload response
     */
    async uploadFile(file, type = 'input') {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('type', type);

        return await this.request('/upload/image', {
            method: 'POST',
            body: formData,
            headers: {} // Let browser set Content-Type for FormData
        });
    }

    /**
     * Get file URL
     * @param {string} filename - File name
     * @param {string} type - File type
     * @returns {string} File URL
     */
    getFileURL(filename, type = 'input') {
        return `${this.baseURL}/view?filename=${encodeURIComponent(filename)}&type=${type}`;
    }

    /**
     * Save workflow to server
     * @param {Object} workflow - Workflow object
     * @param {string} filename - File name
     * @returns {Promise<Object>} Save response
     */
    async saveWorkflow(workflow, filename) {
        return await this.request('/api/workflows', {
            method: 'POST',
            body: JSON.stringify({
                filename: filename,
                workflow: workflow
            })
        });
    }

    /**
     * Load workflow from server
     * @param {string} filename - File name
     * @returns {Promise<Object>} Workflow object
     */
    async loadWorkflow(filename) {
        return await this.request(`/api/workflows/${encodeURIComponent(filename)}`);
    }

    /**
     * Get available workflows
     * @returns {Promise<Array>} Available workflows
     */
    async getWorkflows() {
        return await this.request('/api/workflows');
    }

    /**
     * Delete workflow
     * @param {string} filename - File name
     * @returns {Promise<Object>} Delete response
     */
    async deleteWorkflow(filename) {
        return await this.request(`/api/workflows/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
    }

    /**
     * Get embeddings
     * @returns {Promise<Object>} Available embeddings
     */
    async getEmbeddings() {
        return await this.request('/embeddings');
    }

    /**
     * Get extension list
     * @returns {Promise<Object>} Available extensions
     */
    async getExtensions() {
        return await this.request('/extensions');
    }

    /**
     * Interrupt current execution
     * @returns {Promise<Object>} Interrupt response
     */
    async interrupt() {
        return await this.request('/interrupt', {
            method: 'POST'
        });
    }

    /**
     * Free memory
     * @returns {Promise<Object>} Free memory response
     */
    async freeMemory() {
        return await this.request('/free', {
            method: 'POST'
        });
    }

    /**
     * Get device stats
     * @returns {Promise<Object>} Device statistics
     */
    async getDeviceStats() {
        return await this.request('/api/device_stats');
    }

    /**
     * Validate workflow
     * @param {Object} workflow - Workflow to validate
     * @returns {Promise<Object>} Validation result
     */
    async validateWorkflow(workflow) {
        return await this.request('/api/validate', {
            method: 'POST',
            body: JSON.stringify({ workflow })
        });
    }

    /**
     * Get custom node info
     * @returns {Promise<Object>} Custom node information
     */
    async getCustomNodeInfo() {
        return await this.request('/api/nodes');
    }

    /**
     * Check if connected
     * @returns {boolean} Connection status
     */
    isWebSocketConnected() {
        return this.isConnected && this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }

    /**
     * Close connections
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close(1000, 'User disconnected');
        }
        this.isConnected = false;
    }

    /**
     * Ping server to check connectivity
     * @returns {Promise<boolean>} True if server is responsive
     */
    async ping() {
        try {
            const response = await fetch(`${this.baseURL}/health`, {
                method: 'GET',
                timeout: 5000
            });
            return response.ok;
        } catch (error) {
            return false;
        }
    }
}

// Export for use in other files
window.ComfyUIAPIClient = ComfyUIAPIClient;