// API routes configuration
export const apiRoutes = {
    // LoRA routes
    loras: {
        list: '/api/loras',
        detail: (id) => `/api/loras/${id}`,
        delete: (id) => `/api/loras/${id}`,
        update: (id) => `/api/loras/${id}`,
        civitai: (id) => `/api/loras/${id}/civitai`,
        download: '/api/download-model',
        move: '/api/move-lora',
        scan: '/api/scan-loras'
    },
    
    // Recipe routes
    recipes: {
        list: '/api/recipes',
        detail: (id) => `/api/recipes/${id}`,
        delete: (id) => `/api/recipes/${id}`,
        update: (id) => `/api/recipes/${id}`,
        analyze: '/api/analyze-recipe-image',
        save: '/api/save-recipe'
    },
    
    // Checkpoint routes
    checkpoints: {
        list: '/api/checkpoints',
        detail: (id) => `/api/checkpoints/${id}`,
        delete: (id) => `/api/checkpoints/${id}`,
        update: (id) => `/api/checkpoints/${id}`
    },
    
    // WebSocket routes
    ws: {
        fetchProgress: (protocol) => `${protocol}://${window.location.host}/ws/fetch-progress`
    }
};

// Page routes
export const pageRoutes = {
    loras: '/loras',
    recipes: '/loras/recipes',
    checkpoints: '/checkpoints',
    statistics: '/statistics'
};

// Helper function to get current page type
export function getCurrentPageType() {
    const path = window.location.pathname;
    if (path.includes('/loras/recipes')) return 'recipes';
    if (path.includes('/checkpoints')) return 'checkpoints';
    if (path.includes('/statistics')) return 'statistics';
    if (path.includes('/loras')) return 'loras';
    return 'unknown';
}