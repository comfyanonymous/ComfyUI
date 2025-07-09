/**
 * ComfyUI Mobile Interface - Graph Linearization
 * Converts the 2D workflow graph into a linear execution sequence
 */

class GraphLinearization {
    constructor() {
        this.nodeCache = new Map();
        this.dependencyCache = new Map();
    }

    /**
     * Linearize a workflow graph using topological sort
     * @param {Object} workflow - The workflow object containing nodes
     * @returns {Array} Array of nodes in execution order
     */
    linearizeWorkflow(workflow) {
        if (!workflow || !workflow.nodes || Object.keys(workflow.nodes).length === 0) {
            return [];
        }

        try {
            // Clear caches
            this.nodeCache.clear();
            this.dependencyCache.clear();

            // Build dependency graph
            const dependencyGraph = this.buildDependencyGraph(workflow);
            
            // Perform topological sort
            const sortedNodes = this.topologicalSort(dependencyGraph);
            
            // Convert to linear array with additional metadata
            return this.enrichLinearizedNodes(sortedNodes, workflow);
        } catch (error) {
            console.error('Graph linearization error:', error);
            // Fallback to simple node order
            return this.fallbackLinearization(workflow);
        }
    }

    /**
     * Build dependency graph from workflow
     * @param {Object} workflow - The workflow object
     * @returns {Map} Dependency graph
     */
    buildDependencyGraph(workflow) {
        const graph = new Map();
        const nodes = workflow.nodes;

        // Initialize all nodes in the graph
        for (const nodeId in nodes) {
            graph.set(nodeId, {
                node: nodes[nodeId],
                dependencies: new Set(),
                dependents: new Set()
            });
        }

        // Build dependencies based on connections
        for (const nodeId in nodes) {
            const node = nodes[nodeId];
            
            if (node.inputs) {
                for (const inputName in node.inputs) {
                    const input = node.inputs[inputName];
                    
                    // Check if input is connected to another node
                    if (Array.isArray(input) && input.length >= 2) {
                        const sourceNodeId = input[0];
                        const sourceOutputIndex = input[1];
                        
                        if (graph.has(sourceNodeId)) {
                            // Add dependency
                            graph.get(nodeId).dependencies.add(sourceNodeId);
                            graph.get(sourceNodeId).dependents.add(nodeId);
                        }
                    }
                }
            }
        }

        return graph;
    }

    /**
     * Perform topological sort on dependency graph
     * @param {Map} graph - Dependency graph
     * @returns {Array} Sorted node IDs
     */
    topologicalSort(graph) {
        const result = [];
        const inDegree = new Map();
        const queue = [];

        // Calculate in-degrees
        for (const [nodeId, nodeData] of graph) {
            inDegree.set(nodeId, nodeData.dependencies.size);
            if (nodeData.dependencies.size === 0) {
                queue.push(nodeId);
            }
        }

        // Process nodes with no dependencies first
        while (queue.length > 0) {
            const currentNodeId = queue.shift();
            result.push(currentNodeId);

            // Reduce in-degree of dependent nodes
            const currentNode = graph.get(currentNodeId);
            for (const dependentId of currentNode.dependents) {
                const newInDegree = inDegree.get(dependentId) - 1;
                inDegree.set(dependentId, newInDegree);
                
                if (newInDegree === 0) {
                    queue.push(dependentId);
                }
            }
        }

        // Check for cycles
        if (result.length !== graph.size) {
            console.warn('Cycle detected in workflow graph, using partial sort');
            // Add remaining nodes
            for (const [nodeId] of graph) {
                if (!result.includes(nodeId)) {
                    result.push(nodeId);
                }
            }
        }

        return result;
    }

    /**
     * Enrich linearized nodes with additional metadata
     * @param {Array} sortedNodeIds - Sorted node IDs
     * @param {Object} workflow - Original workflow
     * @returns {Array} Enriched node array
     */
    enrichLinearizedNodes(sortedNodeIds, workflow) {
        const enrichedNodes = [];
        const nodeTypes = this.getNodeTypes();

        for (let i = 0; i < sortedNodeIds.length; i++) {
            const nodeId = sortedNodeIds[i];
            const originalNode = workflow.nodes[nodeId];
            
            if (!originalNode) continue;

            const enrichedNode = {
                id: nodeId,
                type: originalNode.type || 'unknown',
                title: this.getNodeTitle(originalNode),
                executionOrder: i + 1,
                inputs: this.processNodeInputs(originalNode, workflow.nodes),
                outputs: this.processNodeOutputs(originalNode),
                widgets: this.processNodeWidgets(originalNode),
                status: 'idle',
                category: this.getNodeCategory(originalNode.type, nodeTypes),
                description: this.getNodeDescription(originalNode.type, nodeTypes),
                position: originalNode.pos || [0, 0],
                size: originalNode.size || [200, 100],
                flags: originalNode.flags || {},
                mode: originalNode.mode || 0,
                originalNode: originalNode
            };

            enrichedNodes.push(enrichedNode);
        }

        return enrichedNodes;
    }

    /**
     * Process node inputs and resolve connections
     * @param {Object} node - Node object
     * @param {Object} allNodes - All nodes in the workflow
     * @returns {Array} Processed inputs
     */
    processNodeInputs(node, allNodes) {
        const inputs = [];
        
        if (node.inputs) {
            for (const inputName in node.inputs) {
                const input = node.inputs[inputName];
                const inputInfo = {
                    name: inputName,
                    type: this.getInputType(node.type, inputName),
                    connected: false,
                    connection: null,
                    value: null
                };

                // Check if input is connected
                if (Array.isArray(input) && input.length >= 2) {
                    const sourceNodeId = input[0];
                    const sourceOutputIndex = input[1];
                    const sourceNode = allNodes[sourceNodeId];
                    
                    if (sourceNode) {
                        inputInfo.connected = true;
                        inputInfo.connection = {
                            sourceNodeId: sourceNodeId,
                            sourceNodeTitle: this.getNodeTitle(sourceNode),
                            sourceOutputIndex: sourceOutputIndex,
                            sourceOutputName: this.getOutputName(sourceNode, sourceOutputIndex)
                        };
                    }
                } else {
                    // Static value
                    inputInfo.value = input;
                }

                inputs.push(inputInfo);
            }
        }

        return inputs;
    }

    /**
     * Process node outputs
     * @param {Object} node - Node object
     * @returns {Array} Processed outputs
     */
    processNodeOutputs(node) {
        const outputs = [];
        const nodeType = node.type;
        const outputInfo = this.getNodeOutputInfo(nodeType);

        if (outputInfo && outputInfo.length > 0) {
            outputInfo.forEach((output, index) => {
                outputs.push({
                    name: output.name || `Output ${index + 1}`,
                    type: output.type || 'unknown',
                    index: index
                });
            });
        } else {
            // Default output if no specific info available
            outputs.push({
                name: 'Output',
                type: 'unknown',
                index: 0
            });
        }

        return outputs;
    }

    /**
     * Process node widgets (editable parameters)
     * @param {Object} node - Node object
     * @returns {Array} Processed widgets
     */
    processNodeWidgets(node) {
        const widgets = [];
        
        if (node.widgets_values && node.widgets_values.length > 0) {
            const widgetInfo = this.getNodeWidgetInfo(node.type);
            
            node.widgets_values.forEach((value, index) => {
                const widget = {
                    name: widgetInfo[index]?.name || `Widget ${index + 1}`,
                    type: widgetInfo[index]?.type || 'text',
                    value: value,
                    index: index,
                    options: widgetInfo[index]?.options || {}
                };
                widgets.push(widget);
            });
        }

        return widgets;
    }

    /**
     * Get node title for display
     * @param {Object} node - Node object
     * @returns {string} Node title
     */
    getNodeTitle(node) {
        if (node.title) return node.title;
        if (node.type) return this.formatNodeType(node.type);
        return 'Unknown Node';
    }

    /**
     * Format node type for display
     * @param {string} type - Node type
     * @returns {string} Formatted type
     */
    formatNodeType(type) {
        return type.replace(/([A-Z])/g, ' $1').trim();
    }

    /**
     * Get node category based on type
     * @param {string} type - Node type
     * @param {Object} nodeTypes - Node types definition
     * @returns {string} Category
     */
    getNodeCategory(type, nodeTypes) {
        if (nodeTypes && nodeTypes[type] && nodeTypes[type].category) {
            return nodeTypes[type].category;
        }
        
        // Fallback categorization based on type name
        if (type.includes('Load')) return 'loaders';
        if (type.includes('Save')) return 'output';
        if (type.includes('Sample')) return 'sampling';
        if (type.includes('Encode') || type.includes('Decode')) return 'conditioning';
        if (type.includes('Image')) return 'image';
        if (type.includes('Model')) return 'models';
        
        return 'misc';
    }

    /**
     * Get node description
     * @param {string} type - Node type
     * @param {Object} nodeTypes - Node types definition
     * @returns {string} Description
     */
    getNodeDescription(type, nodeTypes) {
        if (nodeTypes && nodeTypes[type] && nodeTypes[type].description) {
            return nodeTypes[type].description;
        }
        return `${this.formatNodeType(type)} node`;
    }

    /**
     * Get input type for a node input
     * @param {string} nodeType - Node type
     * @param {string} inputName - Input name
     * @returns {string} Input type
     */
    getInputType(nodeType, inputName) {
        const nodeTypes = this.getNodeTypes();
        if (nodeTypes && nodeTypes[nodeType] && nodeTypes[nodeType].input) {
            const required = nodeTypes[nodeType].input.required || {};
            const optional = nodeTypes[nodeType].input.optional || {};
            
            if (required[inputName]) {
                return Array.isArray(required[inputName]) ? required[inputName][0] : required[inputName];
            }
            if (optional[inputName]) {
                return Array.isArray(optional[inputName]) ? optional[inputName][0] : optional[inputName];
            }
        }
        return 'unknown';
    }

    /**
     * Get output name for a node output
     * @param {Object} node - Node object
     * @param {number} outputIndex - Output index
     * @returns {string} Output name
     */
    getOutputName(node, outputIndex) {
        const outputs = this.getNodeOutputInfo(node.type);
        if (outputs && outputs[outputIndex]) {
            return outputs[outputIndex].name;
        }
        return `Output ${outputIndex + 1}`;
    }

    /**
     * Get node output information
     * @param {string} nodeType - Node type
     * @returns {Array} Output information
     */
    getNodeOutputInfo(nodeType) {
        const nodeTypes = this.getNodeTypes();
        if (nodeTypes && nodeTypes[nodeType] && nodeTypes[nodeType].output) {
            return nodeTypes[nodeType].output.map((type, index) => ({
                name: Array.isArray(type) ? type[0] : type,
                type: Array.isArray(type) ? type[0] : type
            }));
        }
        return [];
    }

    /**
     * Get node widget information
     * @param {string} nodeType - Node type
     * @returns {Array} Widget information
     */
    getNodeWidgetInfo(nodeType) {
        // This would be populated from the actual ComfyUI node definitions
        // For now, return empty array as placeholder
        return [];
    }

    /**
     * Get node types definition (placeholder)
     * @returns {Object} Node types
     */
    getNodeTypes() {
        // This would be fetched from the ComfyUI API
        // For now, return empty object as placeholder
        return {};
    }

    /**
     * Fallback linearization when topological sort fails
     * @param {Object} workflow - Workflow object
     * @returns {Array} Simple node array
     */
    fallbackLinearization(workflow) {
        const nodes = [];
        
        for (const nodeId in workflow.nodes) {
            const node = workflow.nodes[nodeId];
            nodes.push({
                id: nodeId,
                type: node.type || 'unknown',
                title: this.getNodeTitle(node),
                executionOrder: parseInt(nodeId),
                inputs: [],
                outputs: [],
                widgets: [],
                status: 'idle',
                category: 'misc',
                description: 'Node',
                position: node.pos || [0, 0],
                size: node.size || [200, 100],
                flags: node.flags || {},
                mode: node.mode || 0,
                originalNode: node
            });
        }

        // Sort by node ID as fallback
        nodes.sort((a, b) => parseInt(a.id) - parseInt(b.id));
        
        return nodes;
    }

    /**
     * Get connection compatibility between two sockets
     * @param {string} outputType - Output socket type
     * @param {string} inputType - Input socket type
     * @returns {boolean} True if compatible
     */
    areSocketsCompatible(outputType, inputType) {
        // Basic compatibility check - can be extended
        if (outputType === inputType) return true;
        
        // Special compatibility rules
        const compatibilityRules = {
            'MODEL': ['MODEL'],
            'CLIP': ['CLIP'],
            'VAE': ['VAE'],
            'CONDITIONING': ['CONDITIONING'],
            'LATENT': ['LATENT'],
            'IMAGE': ['IMAGE'],
            'MASK': ['MASK'],
            'INT': ['INT', 'FLOAT'],
            'FLOAT': ['FLOAT', 'INT'],
            'STRING': ['STRING'],
            'BOOLEAN': ['BOOLEAN'],
            '*': ['*'] // Wildcard type
        };

        const outputCompatible = compatibilityRules[outputType] || [];
        return outputCompatible.includes(inputType) || inputType === '*' || outputType === '*';
    }

    /**
     * Find potential connections for a socket
     * @param {Object} socket - Socket object
     * @param {Array} allNodes - All nodes in the workflow
     * @param {string} direction - 'input' or 'output'
     * @returns {Array} Compatible sockets
     */
    findCompatibleSockets(socket, allNodes, direction) {
        const compatibleSockets = [];
        
        for (const node of allNodes) {
            const sockets = direction === 'input' ? node.outputs : node.inputs;
            
            for (const targetSocket of sockets) {
                const isCompatible = direction === 'input' 
                    ? this.areSocketsCompatible(socket.type, targetSocket.type)
                    : this.areSocketsCompatible(targetSocket.type, socket.type);
                
                if (isCompatible && node.id !== socket.nodeId) {
                    compatibleSockets.push({
                        nodeId: node.id,
                        nodeTitle: node.title,
                        socket: targetSocket,
                        compatible: true
                    });
                }
            }
        }

        return compatibleSockets;
    }

    /**
     * Validate workflow for common issues
     * @param {Array} linearizedNodes - Linearized nodes array
     * @returns {Array} Validation issues
     */
    validateWorkflow(linearizedNodes) {
        const issues = [];
        
        // Check for unconnected required inputs
        for (const node of linearizedNodes) {
            for (const input of node.inputs) {
                if (input.required && !input.connected && input.value === null) {
                    issues.push({
                        type: 'error',
                        nodeId: node.id,
                        message: `Required input '${input.name}' is not connected`,
                        severity: 'high'
                    });
                }
            }
        }

        // Check for isolated nodes
        for (const node of linearizedNodes) {
            const hasConnectedInputs = node.inputs.some(input => input.connected);
            const hasConnectedOutputs = this.hasConnectedOutputs(node, linearizedNodes);
            
            if (!hasConnectedInputs && !hasConnectedOutputs && node.type !== 'SaveImage') {
                issues.push({
                    type: 'warning',
                    nodeId: node.id,
                    message: `Node '${node.title}' appears to be isolated`,
                    severity: 'medium'
                });
            }
        }

        return issues;
    }

    /**
     * Check if node has connected outputs
     * @param {Object} node - Node to check
     * @param {Array} allNodes - All nodes in the workflow
     * @returns {boolean} True if has connected outputs
     */
    hasConnectedOutputs(node, allNodes) {
        for (const otherNode of allNodes) {
            for (const input of otherNode.inputs) {
                if (input.connected && input.connection.sourceNodeId === node.id) {
                    return true;
                }
            }
        }
        return false;
    }
}

// Export for use in other files
window.GraphLinearization = GraphLinearization;