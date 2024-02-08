const API_URL = {
    GET_EMBEDDINGS: '/embeddings',
    GET_NODE_DEFS: '/object_info',
    GET_HISTORY: (maxItems: number) => `/history?max_items=${maxItems}`,
    GET_SYSTEM_STATS: '/system_stats',
    GET_USER_CONFIG: '/users',
    CREATE_USER: '/users',
    GET_SETTINGS: '/settings',
    GET_SETTING_BY_ID: (id: string) => `/settings/${id}`,
    STORE_SETTINGS: '/settings',
    GET_USER_DATA_FILE: (file: string) => `/userdata/${encodeURIComponent(file)}`,
    STORE_USER_DATA_FILE: (file: string) => `/userdata/${encodeURIComponent(file)}`,
};

export default API_URL;
