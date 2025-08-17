// Statistics page functionality
import { appCore } from './core.js';
import { showToast } from './utils/uiHelpers.js';

// Chart.js import (assuming it's available globally or via CDN)
// If Chart.js isn't available, we'll need to add it to the project

class StatisticsManager {
    constructor() {
        this.charts = {};
        this.data = {};
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;

        console.log('StatisticsManager: Initializing...');
        
        // Initialize tab functionality
        this.initializeTabs();
        
        // Load initial data
        await this.loadAllData();
        
        // Initialize charts and visualizations
        this.initializeVisualizations();
        
        this.initialized = true;
    }

    initializeTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.dataset.tab;
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active tab panel
                tabPanels.forEach(panel => panel.classList.remove('active'));
                const targetPanel = document.getElementById(`${tabId}-panel`);
                if (targetPanel) {
                    targetPanel.classList.add('active');
                    
                    // Refresh charts when tab becomes visible
                    this.refreshChartsInPanel(tabId);
                }
            });
        });
    }

    async loadAllData() {
        try {
            // Load all statistics data in parallel
            const [
                collectionOverview,
                usageAnalytics,
                baseModelDistribution,
                tagAnalytics,
                storageAnalytics,
                insights
            ] = await Promise.all([
                this.fetchData('/api/stats/collection-overview'),
                this.fetchData('/api/stats/usage-analytics'),
                this.fetchData('/api/stats/base-model-distribution'),
                this.fetchData('/api/stats/tag-analytics'),
                this.fetchData('/api/stats/storage-analytics'),
                this.fetchData('/api/stats/insights')
            ]);

            this.data = {
                collection: collectionOverview.data,
                usage: usageAnalytics.data,
                baseModels: baseModelDistribution.data,
                tags: tagAnalytics.data,
                storage: storageAnalytics.data,
                insights: insights.data
            };

            console.log('Statistics data loaded:', this.data);
        } catch (error) {
            console.error('Error loading statistics data:', error);
            showToast('Failed to load statistics data', 'error');
        }
    }

    async fetchData(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${endpoint}: ${response.statusText}`);
        }
        return response.json();
    }

    initializeVisualizations() {
        // Initialize metrics cards
        this.renderMetricsCards();
        
        // Initialize charts
        this.initializeCharts();
        
        // Initialize lists and other components
        this.renderTopModelsLists();
        this.renderTagCloud();
        this.renderInsights();
    }

    renderMetricsCards() {
        const metricsGrid = document.getElementById('metricsGrid');
        if (!metricsGrid || !this.data.collection) return;

        const metrics = [
            {
                icon: 'fas fa-magic',
                value: this.data.collection.total_models,
                label: 'Total Models',
                format: 'number'
            },
            {
                icon: 'fas fa-database',
                value: this.data.collection.total_size,
                label: 'Total Storage',
                format: 'size'
            },
            {
                icon: 'fas fa-play-circle',
                value: this.data.collection.total_generations,
                label: 'Total Generations',
                format: 'number'
            },
            {
                icon: 'fas fa-chart-line',
                value: this.calculateUsageRate(),
                label: 'Usage Rate',
                format: 'percentage'
            },
            {
                icon: 'fas fa-layer-group',
                value: this.data.collection.lora_count,
                label: 'LoRAs',
                format: 'number'
            },
            {
                icon: 'fas fa-check-circle',
                value: this.data.collection.checkpoint_count,
                label: 'Checkpoints',
                format: 'number'
            },
            {
                icon: 'fas fa-code',
                value: this.data.collection.embedding_count,
                label: 'Embeddings',
                format: 'number'
            }
        ];

        metricsGrid.innerHTML = metrics.map(metric => this.createMetricCard(metric)).join('');
    }

    createMetricCard(metric) {
        const formattedValue = this.formatValue(metric.value, metric.format);
        
        return `
            <div class="metric-card">
                <div class="metric-icon">
                    <i class="${metric.icon}"></i>
                </div>
                <div class="metric-value">${formattedValue}</div>
                <div class="metric-label">${metric.label}</div>
            </div>
        `;
    }

    formatValue(value, format) {
        switch (format) {
            case 'number':
                return new Intl.NumberFormat().format(value);
            case 'size':
                return this.formatFileSize(value);
            case 'percentage':
                return `${value.toFixed(1)}%`;
            default:
                return value;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    calculateUsageRate() {
        if (!this.data.collection) return 0;
        
        const totalModels = this.data.collection.total_models;
        const unusedModels = this.data.collection.unused_loras + 
                           this.data.collection.unused_checkpoints + 
                           this.data.collection.unused_embeddings;
        const usedModels = totalModels - unusedModels;
        
        return totalModels > 0 ? (usedModels / totalModels) * 100 : 0;
    }

    initializeCharts() {
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js is not available. Charts will not be rendered.');
            this.showChartPlaceholders();
            return;
        }

        // Collection pie chart
        this.createCollectionPieChart();
        
        // Base model distribution chart
        this.createBaseModelChart();
        
        // Usage timeline chart
        this.createUsageTimelineChart();
        
        // Usage distribution chart
        this.createUsageDistributionChart();
        
        // Storage chart
        this.createStorageChart();
        
        // Storage efficiency chart
        this.createStorageEfficiencyChart();
    }

    createCollectionPieChart() {
        const ctx = document.getElementById('collectionPieChart');
        if (!ctx || !this.data.collection) return;

        const data = {
            labels: ['LoRAs', 'Checkpoints', 'Embeddings'],
            datasets: [{
                data: [
                    this.data.collection.lora_count, 
                    this.data.collection.checkpoint_count,
                    this.data.collection.embedding_count
                ],
                backgroundColor: [
                    'oklch(68% 0.28 256)',
                    'oklch(68% 0.28 200)',
                    'oklch(68% 0.28 120)'
                ],
                borderWidth: 2,
                borderColor: getComputedStyle(document.documentElement).getPropertyValue('--border-color')
            }]
        };

        this.charts.collection = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createBaseModelChart() {
        const ctx = document.getElementById('baseModelChart');
        if (!ctx || !this.data.baseModels) return;

        const loraData = this.data.baseModels.loras;
        const checkpointData = this.data.baseModels.checkpoints;
        const embeddingData = this.data.baseModels.embeddings;
        
        const allModels = new Set([
            ...Object.keys(loraData), 
            ...Object.keys(checkpointData),
            ...Object.keys(embeddingData)
        ]);
        
        const data = {
            labels: Array.from(allModels),
            datasets: [
                {
                    label: 'LoRAs',
                    data: Array.from(allModels).map(model => loraData[model] || 0),
                    backgroundColor: 'oklch(68% 0.28 256 / 0.7)'
                },
                {
                    label: 'Checkpoints',
                    data: Array.from(allModels).map(model => checkpointData[model] || 0),
                    backgroundColor: 'oklch(68% 0.28 200 / 0.7)'
                },
                {
                    label: 'Embeddings',
                    data: Array.from(allModels).map(model => embeddingData[model] || 0),
                    backgroundColor: 'oklch(68% 0.28 120 / 0.7)'
                }
            ]
        };

        this.charts.baseModels = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true
                    },
                    y: {
                        stacked: true
                    }
                }
            }
        });
    }

    createUsageTimelineChart() {
        const ctx = document.getElementById('usageTimelineChart');
        if (!ctx || !this.data.usage) return;

        const timeline = this.data.usage.usage_timeline || [];
        
        const data = {
            labels: timeline.map(item => new Date(item.date).toLocaleDateString()),
            datasets: [
                {
                    label: 'LoRA Usage',
                    data: timeline.map(item => item.lora_usage),
                    borderColor: 'oklch(68% 0.28 256)',
                    backgroundColor: 'oklch(68% 0.28 256 / 0.1)',
                    fill: true
                },
                {
                    label: 'Checkpoint Usage',
                    data: timeline.map(item => item.checkpoint_usage),
                    borderColor: 'oklch(68% 0.28 200)',
                    backgroundColor: 'oklch(68% 0.28 200 / 0.1)',
                    fill: true
                },
                {
                    label: 'Embedding Usage',
                    data: timeline.map(item => item.embedding_usage),
                    borderColor: 'oklch(68% 0.28 120)',
                    backgroundColor: 'oklch(68% 0.28 120 / 0.1)',
                    fill: true
                }
            ]
        };

        this.charts.timeline = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Usage Count'
                        }
                    }
                }
            }
        });
    }

    createUsageDistributionChart() {
        const ctx = document.getElementById('usageDistributionChart');
        if (!ctx || !this.data.usage) return;

        const topLoras = this.data.usage.top_loras || [];
        const topCheckpoints = this.data.usage.top_checkpoints || [];
        const topEmbeddings = this.data.usage.top_embeddings || [];
        
        // Combine and sort all models by usage
        const allModels = [
            ...topLoras.map(m => ({ ...m, type: 'LoRA' })),
            ...topCheckpoints.map(m => ({ ...m, type: 'Checkpoint' })),
            ...topEmbeddings.map(m => ({ ...m, type: 'Embedding' }))
        ].sort((a, b) => b.usage_count - a.usage_count).slice(0, 10);

        const data = {
            labels: allModels.map(model => model.name),
            datasets: [{
                label: 'Usage Count',
                data: allModels.map(model => model.usage_count),
                backgroundColor: allModels.map(model => {
                    switch(model.type) {
                        case 'LoRA': return 'oklch(68% 0.28 256)';
                        case 'Checkpoint': return 'oklch(68% 0.28 200)';
                        case 'Embedding': return 'oklch(68% 0.28 120)';
                        default: return 'oklch(68% 0.28 256)';
                    }
                })
            }]
        };

        this.charts.distribution = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    createStorageChart() {
        const ctx = document.getElementById('storageChart');
        if (!ctx || !this.data.collection) return;

        const data = {
            labels: ['LoRAs', 'Checkpoints', 'Embeddings'],
            datasets: [{
                data: [
                    this.data.collection.lora_size, 
                    this.data.collection.checkpoint_size,
                    this.data.collection.embedding_size
                ],
                backgroundColor: [
                    'oklch(68% 0.28 256)',
                    'oklch(68% 0.28 200)',
                    'oklch(68% 0.28 120)'
                ]
            }]
        };

        this.charts.storage = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = this.formatFileSize(context.raw);
                                return `${context.label}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
    }

    createStorageEfficiencyChart() {
        const ctx = document.getElementById('storageEfficiencyChart');
        if (!ctx || !this.data.storage) return;

        const loraData = this.data.storage.loras || [];
        const checkpointData = this.data.storage.checkpoints || [];
        const embeddingData = this.data.storage.embeddings || [];
        
        const allData = [
            ...loraData.map(item => ({ ...item, type: 'LoRA' })),
            ...checkpointData.map(item => ({ ...item, type: 'Checkpoint' })),
            ...embeddingData.map(item => ({ ...item, type: 'Embedding' }))
        ];

        const data = {
            datasets: [{
                label: 'Models',
                data: allData.map(item => ({
                    x: item.size,
                    y: item.usage_count,
                    name: item.name,
                    type: item.type
                })),
                backgroundColor: allData.map(item => {
                    switch(item.type) {
                        case 'LoRA': return 'oklch(68% 0.28 256 / 0.6)';
                        case 'Checkpoint': return 'oklch(68% 0.28 200 / 0.6)';
                        case 'Embedding': return 'oklch(68% 0.28 120 / 0.6)';
                        default: return 'oklch(68% 0.28 256 / 0.6)';
                    }
                })
            }]
        };

        this.charts.efficiency = new Chart(ctx, {
            type: 'scatter',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'File Size (bytes)'
                        },
                        type: 'logarithmic'
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Usage Count'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const point = context.raw;
                                return `${point.name}: ${this.formatFileSize(point.x)}, ${point.y} uses`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderTopModelsLists() {
        this.renderTopLorasList();
        this.renderTopCheckpointsList();
        this.renderTopEmbeddingsList();
        this.renderLargestModelsList();
    }

    renderTopLorasList() {
        const container = document.getElementById('topLorasList');
        if (!container || !this.data.usage?.top_loras) return;

        const topLoras = this.data.usage.top_loras;
        
        if (topLoras.length === 0) {
            container.innerHTML = '<div class="loading-placeholder">No usage data available</div>';
            return;
        }

        container.innerHTML = topLoras.map(lora => `
            <div class="model-item">
                <img src="${lora.preview_url || '/loras_static/images/no-preview.png'}" 
                     alt="${lora.name}" class="model-preview" 
                     onerror="this.src='/loras_static/images/no-preview.png'">
                <div class="model-info">
                    <div class="model-name" title="${lora.name}">${lora.name}</div>
                    <div class="model-meta">${lora.base_model} • ${lora.folder}</div>
                </div>
                <div class="model-usage">${lora.usage_count}</div>
            </div>
        `).join('');
    }

    renderTopCheckpointsList() {
        const container = document.getElementById('topCheckpointsList');
        if (!container || !this.data.usage?.top_checkpoints) return;

        const topCheckpoints = this.data.usage.top_checkpoints;
        
        if (topCheckpoints.length === 0) {
            container.innerHTML = '<div class="loading-placeholder">No usage data available</div>';
            return;
        }

        container.innerHTML = topCheckpoints.map(checkpoint => `
            <div class="model-item">
                <img src="${checkpoint.preview_url || '/loras_static/images/no-preview.png'}" 
                     alt="${checkpoint.name}" class="model-preview"
                     onerror="this.src='/loras_static/images/no-preview.png'">
                <div class="model-info">
                    <div class="model-name" title="${checkpoint.name}">${checkpoint.name}</div>
                    <div class="model-meta">${checkpoint.base_model} • ${checkpoint.folder}</div>
                </div>
                <div class="model-usage">${checkpoint.usage_count}</div>
            </div>
        `).join('');
    }

    renderTopEmbeddingsList() {
        const container = document.getElementById('topEmbeddingsList');
        if (!container || !this.data.usage?.top_embeddings) return;

        const topEmbeddings = this.data.usage.top_embeddings;
        
        if (topEmbeddings.length === 0) {
            container.innerHTML = '<div class="loading-placeholder">No usage data available</div>';
            return;
        }

        container.innerHTML = topEmbeddings.map(embedding => `
            <div class="model-item">
                <img src="${embedding.preview_url || '/loras_static/images/no-preview.png'}" 
                     alt="${embedding.name}" class="model-preview"
                     onerror="this.src='/loras_static/images/no-preview.png'">
                <div class="model-info">
                    <div class="model-name" title="${embedding.name}">${embedding.name}</div>
                    <div class="model-meta">${embedding.base_model} • ${embedding.folder}</div>
                </div>
                <div class="model-usage">${embedding.usage_count}</div>
            </div>
        `).join('');
    }

    renderLargestModelsList() {
        const container = document.getElementById('largestModelsList');
        if (!container || !this.data.storage) return;

        const loraModels = this.data.storage.loras || [];
        const checkpointModels = this.data.storage.checkpoints || [];
        const embeddingModels = this.data.storage.embeddings || [];
        
        // Combine and sort by size
        const allModels = [
            ...loraModels.map(m => ({ ...m, type: 'LoRA' })),
            ...checkpointModels.map(m => ({ ...m, type: 'Checkpoint' })),
            ...embeddingModels.map(m => ({ ...m, type: 'Embedding' }))
        ].sort((a, b) => b.size - a.size).slice(0, 10);

        if (allModels.length === 0) {
            container.innerHTML = '<div class="loading-placeholder">No storage data available</div>';
            return;
        }

        container.innerHTML = allModels.map(model => `
            <div class="model-item">
                <div class="model-info">
                    <div class="model-name" title="${model.name}">${model.name}</div>
                    <div class="model-meta">${model.type} • ${model.base_model}</div>
                </div>
                <div class="model-usage">${this.formatFileSize(model.size)}</div>
            </div>
        `).join('');
    }

    renderTagCloud() {
        const container = document.getElementById('tagCloud');
        if (!container || !this.data.tags?.top_tags) return;

        const topTags = this.data.tags.top_tags.slice(0, 30); // Show top 30 tags
        const maxCount = Math.max(...topTags.map(tag => tag.count));
        
        container.innerHTML = topTags.map(tagData => {
            const size = Math.ceil((tagData.count / maxCount) * 5);
            return `
                <span class="tag-cloud-item size-${size}" 
                      title="${tagData.tag}: ${tagData.count} models">
                    ${tagData.tag}
                </span>
            `;
        }).join('');
    }

    renderInsights() {
        const container = document.getElementById('insightsList');
        if (!container || !this.data.insights?.insights) return;

        const insights = this.data.insights.insights;
        
        if (insights.length === 0) {
            container.innerHTML = '<div class="loading-placeholder">No insights available</div>';
            return;
        }

        container.innerHTML = insights.map(insight => `
            <div class="insight-card type-${insight.type}">
                <div class="insight-title">${insight.title}</div>
                <div class="insight-description">${insight.description}</div>
                <div class="insight-suggestion">${insight.suggestion}</div>
            </div>
        `).join('');

        // Render collection analysis cards
        this.renderCollectionAnalysis();
    }

    renderCollectionAnalysis() {
        const container = document.getElementById('collectionAnalysis');
        if (!container || !this.data.collection) return;

        const analysis = [
            {
                icon: 'fas fa-percentage',
                value: this.calculateUsageRate(),
                label: 'Usage Rate',
                format: 'percentage'
            },
            {
                icon: 'fas fa-tags',
                value: this.data.tags?.total_unique_tags || 0,
                label: 'Unique Tags',
                format: 'number'
            },
            {
                icon: 'fas fa-clock',
                value: this.data.collection.unused_loras + this.data.collection.unused_checkpoints,
                label: 'Unused Models',
                format: 'number'
            },
            {
                icon: 'fas fa-chart-line',
                value: this.calculateAverageUsage(),
                label: 'Avg. Uses/Model',
                format: 'decimal'
            }
        ];

        container.innerHTML = analysis.map(item => `
            <div class="analysis-card">
                <div class="card-icon">
                    <i class="${item.icon}"></i>
                </div>
                <div class="card-value">${this.formatValue(item.value, item.format)}</div>
                <div class="card-label">${item.label}</div>
            </div>
        `).join('');
    }

    calculateAverageUsage() {
        if (!this.data.usage || !this.data.collection) return 0;
        
        const totalGenerations = this.data.collection.total_generations;
        const totalModels = this.data.collection.total_models;
        
        return totalModels > 0 ? totalGenerations / totalModels : 0;
    }

    showChartPlaceholders() {
        const chartCanvases = document.querySelectorAll('canvas');
        chartCanvases.forEach(canvas => {
            const container = canvas.parentElement;
            container.innerHTML = '<div class="loading-placeholder"><i class="fas fa-chart-bar"></i> Chart requires Chart.js library</div>';
        });
    }

    refreshChartsInPanel(panelId) {
        // Refresh charts when panels become visible
        setTimeout(() => {
            Object.values(this.charts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    chart.resize();
                }
            });
        }, 100);
    }

    destroy() {
        // Clean up charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
        this.initialized = false;
    }
}

// Initialize statistics page when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Wait for app core to initialize
    await appCore.initialize();
    
    // Initialize statistics functionality
    const statsManager = new StatisticsManager();
    await statsManager.initialize();
    
    // Make statsManager globally available for debugging
    window.statsManager = statsManager;
    
    console.log('Statistics page initialized successfully');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.statsManager) {
        window.statsManager.destroy();
    }
});