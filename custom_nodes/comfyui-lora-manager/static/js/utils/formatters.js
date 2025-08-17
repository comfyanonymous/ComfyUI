/**
 * Format a file size in bytes to a human-readable string
 * @param {number} bytes - The size in bytes
 * @returns {string} Formatted size string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes) {
    if (!bytes || isNaN(bytes)) return '';
    
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(2) + ' ' + sizes[i];
}

/**
 * Convert timestamp to human readable date string
 * @param {number} modified - Timestamp in seconds
 * @returns {string} Formatted date string
 */
export function formatDate(modified) {
    if (!modified) return '';
    const date = new Date(modified * 1000);
    return date.toLocaleString();
}
