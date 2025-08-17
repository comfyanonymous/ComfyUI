// Alphabet component index file
import { AlphabetBar } from './AlphabetBar.js';

// Export the class
export { AlphabetBar };

/**
 * Factory function to create the appropriate alphabet bar
 * @param {string} pageType - The type of page ('loras' or 'checkpoints')
 * @returns {AlphabetBar} - The alphabet bar instance
 */
export function createAlphabetBar(pageType) {
    return new AlphabetBar(pageType);
}