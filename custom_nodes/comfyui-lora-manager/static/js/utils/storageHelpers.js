/**
 * Utility functions for localStorage with namespacing to avoid conflicts
 * with other ComfyUI extensions or the main application
 */

// Namespace prefix for all localStorage keys
const STORAGE_PREFIX = 'lora_manager_';

/**
 * Get an item from localStorage with namespace support and fallback to legacy keys
 * @param {string} key - The key without prefix
 * @param {any} defaultValue - Default value if key doesn't exist
 * @returns {any} The stored value or defaultValue
 */
export function getStorageItem(key, defaultValue = null) {
    // Try with prefix first
    const prefixedValue = localStorage.getItem(STORAGE_PREFIX + key);
    
    if (prefixedValue !== null) {
        // If it's a JSON string, parse it
        try {
            return JSON.parse(prefixedValue);
        } catch (e) {
            return prefixedValue;
        }
    }
    
    // Fallback to legacy key (without prefix)
    const legacyValue = localStorage.getItem(key);
    
    if (legacyValue !== null) {
        // If found in legacy storage, migrate it to prefixed storage
        try {
            const parsedValue = JSON.parse(legacyValue);
            setStorageItem(key, parsedValue);
            return parsedValue;
        } catch (e) {
            setStorageItem(key, legacyValue);
            return legacyValue;
        }
    }
    
    // Return default value if neither prefixed nor legacy key exists
    return defaultValue;
}

/**
 * Set an item in localStorage with namespace prefix
 * @param {string} key - The key without prefix
 * @param {any} value - The value to store
 */
export function setStorageItem(key, value) {
    const prefixedKey = STORAGE_PREFIX + key;
    
    // Convert objects and arrays to JSON strings
    if (typeof value === 'object' && value !== null) {
        localStorage.setItem(prefixedKey, JSON.stringify(value));
    } else {
        localStorage.setItem(prefixedKey, value);
    }
}

/**
 * Remove an item from localStorage (both prefixed and legacy)
 * @param {string} key - The key without prefix
 */
export function removeStorageItem(key) {
    localStorage.removeItem(STORAGE_PREFIX + key);
    localStorage.removeItem(key); // Also remove legacy key
}

/**
 * Get an item from sessionStorage with namespace support
 * @param {string} key - The key without prefix
 * @param {any} defaultValue - Default value if key doesn't exist
 * @returns {any} The stored value or defaultValue
 */
export function getSessionItem(key, defaultValue = null) {
    // Try with prefix
    const prefixedValue = sessionStorage.getItem(STORAGE_PREFIX + key);
    
    if (prefixedValue !== null) {
        // If it's a JSON string, parse it
        try {
            return JSON.parse(prefixedValue);
        } catch (e) {
            return prefixedValue;
        }
    }
    
    // Return default value if key doesn't exist
    return defaultValue;
}

/**
 * Set an item in sessionStorage with namespace prefix
 * @param {string} key - The key without prefix
 * @param {any} value - The value to store
 */
export function setSessionItem(key, value) {
    const prefixedKey = STORAGE_PREFIX + key;
    
    // Convert objects and arrays to JSON strings
    if (typeof value === 'object' && value !== null) {
        sessionStorage.setItem(prefixedKey, JSON.stringify(value));
    } else {
        sessionStorage.setItem(prefixedKey, value);
    }
}

/**
 * Remove an item from sessionStorage with namespace prefix
 * @param {string} key - The key without prefix
 */
export function removeSessionItem(key) {
    sessionStorage.removeItem(STORAGE_PREFIX + key);
}

/**
 * Migrate all existing localStorage items to use the prefix
 * This should be called once during application initialization
 */
export function migrateStorageItems() {
    // Check if migration has already been performed
    if (localStorage.getItem(STORAGE_PREFIX + 'migration_completed')) {
        console.log('Lora Manager: Storage migration already completed');
        return;
    }
    
    // List of known keys used in the application
    const knownKeys = [
        'nsfwBlurLevel',
        'theme',
        'activeFolder',
        'folderTagsCollapsed',
        'settings',
        'loras_filters',
        'recipes_filters',
        'checkpoints_filters',
        'loras_search_prefs',
        'recipes_search_prefs',
        'checkpoints_search_prefs',
        'show_update_notifications',
        'last_update_check',
        'dismissed_banners'
    ];
    
    // Migrate each known key
    knownKeys.forEach(key => {
        const prefixedKey = STORAGE_PREFIX + key;
        
        // Only migrate if the prefixed key doesn't already exist
        if (localStorage.getItem(prefixedKey) === null) {
            const value = localStorage.getItem(key);
            if (value !== null) {
                try {
                    // Try to parse as JSON first
                    const parsedValue = JSON.parse(value);
                    setStorageItem(key, parsedValue);
                } catch (e) {
                    // If not JSON, store as is
                    setStorageItem(key, value);
                }
                
                // We can optionally remove the old key after migration
                localStorage.removeItem(key);
            }
        }
    });
    
    // Mark migration as completed
    localStorage.setItem(STORAGE_PREFIX + 'migration_completed', 'true');
    
    console.log('Lora Manager: Storage migration completed');
}

/**
 * Save a Map to localStorage
 * @param {string} key - The localStorage key
 * @param {Map} map - The Map to save
 */
export function saveMapToStorage(key, map) {
    if (!(map instanceof Map)) {
        console.error('Cannot save non-Map object:', map);
        return;
    }

    try {
        const prefixedKey = STORAGE_PREFIX + key;
        // Convert Map to array of entries and save as JSON
        const entries = Array.from(map.entries());
        localStorage.setItem(prefixedKey, JSON.stringify(entries));
    } catch (error) {
        console.error(`Error saving Map to localStorage (${key}):`, error);
    }
}

/**
 * Load a Map from localStorage
 * @param {string} key - The localStorage key
 * @returns {Map} - The loaded Map or a new empty Map
 */
export function getMapFromStorage(key) {
    try {
        const prefixedKey = STORAGE_PREFIX + key;
        const data = localStorage.getItem(prefixedKey);
        if (!data) return new Map();
        
        // Parse JSON and convert back to Map
        const entries = JSON.parse(data);
        return new Map(entries);
    } catch (error) {
        console.error(`Error loading Map from localStorage (${key}):`, error);
        return new Map();
    }
}

/**
 * Get stored version info from localStorage
 * @returns {string|null} The stored version string or null if not found
 */
export function getStoredVersionInfo() {
    return getStorageItem('version_info', null);
}

/**
 * Store version info to localStorage
 * @param {string} versionInfo - The version info string to store
 */
export function setStoredVersionInfo(versionInfo) {
    setStorageItem('version_info', versionInfo);
}

/**
 * Check if version info matches between stored and current
 * @param {string} currentVersionInfo - The current version info from server
 * @returns {boolean} True if versions match or no stored version exists
 */
export function isVersionMatch(currentVersionInfo) {
    const storedVersion = getStoredVersionInfo();
    // If we have no stored version yet, consider it a match
    if (storedVersion === null) {
        setStoredVersionInfo(currentVersionInfo);
        return true;
    }
    return storedVersion === currentVersionInfo;
}

/**
 * Reset the dismissed status of a specific banner
 * @param {string} bannerId - The ID of the banner to un-dismiss
 */
export function resetDismissedBanner(bannerId) {
    const dismissedBanners = getStorageItem('dismissed_banners', []);
    const updatedBanners = dismissedBanners.filter(id => id !== bannerId);
    setStorageItem('dismissed_banners', updatedBanners);
}