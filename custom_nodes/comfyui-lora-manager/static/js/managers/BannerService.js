import { getStorageItem, setStorageItem } from '../utils/storageHelpers.js';

/**
 * Banner Service for managing notification banners
 */
class BannerService {
    constructor() {
        this.banners = new Map();
        this.container = null;
        this.initialized = false;
    }

    /**
     * Initialize the banner service
     */
    initialize() {
        if (this.initialized) return;

        this.container = document.getElementById('banner-container');
        if (!this.container) {
            console.warn('Banner container not found');
            return;
        }

        // Register default banners
        this.registerBanner('civitai-extension', {
            id: 'civitai-extension',
            title: 'New Tool Available: LM Civitai Extension!',
            content: 'LM Civitai Extension is a browser extension designed to work seamlessly with LoRA Manager to significantly enhance your Civitai browsing experience! See which models you already have, download new ones with a single click, and manage your downloads efficiently.',
            actions: [
                {
                    text: 'Chrome Web Store',
                    icon: 'fab fa-chrome',
                    url: 'https://chromewebstore.google.com/detail/capigligggeijgmocnaflanlbghnamgm?utm_source=item-share-cb',
                    type: 'secondary'
                },
                {
                    text: 'Firefox Extension',
                    icon: 'fab fa-firefox-browser',
                    url: 'https://github.com/willmiao/lm-civitai-extension-firefox/releases/latest/download/extension.xpi',
                    type: 'secondary'
                },
                {
                    text: 'Read more...',
                    icon: 'fas fa-book',
                    url: 'https://github.com/willmiao/ComfyUI-Lora-Manager/wiki/LoRA-Manager-Civitai-Extension-(Chrome-Extension)',
                    type: 'tertiary'
                }
            ],
            dismissible: true,
            priority: 1
        });

        this.showActiveBanners();
        this.initialized = true;
    }

    /**
     * Register a new banner
     * @param {string} id - Unique banner ID
     * @param {Object} bannerConfig - Banner configuration
     */
    registerBanner(id, bannerConfig) {
        this.banners.set(id, bannerConfig);
        
        // If already initialized, render the banner immediately
        if (this.initialized && !this.isBannerDismissed(id) && this.container) {
            this.renderBanner(bannerConfig);
            this.updateContainerVisibility();
        }
    }

    /**
     * Check if a banner has been dismissed
     * @param {string} bannerId - Banner ID
     * @returns {boolean}
     */
    isBannerDismissed(bannerId) {
        const dismissedBanners = getStorageItem('dismissed_banners', []);
        return dismissedBanners.includes(bannerId);
    }

    /**
     * Dismiss a banner
     * @param {string} bannerId - Banner ID
     */
    dismissBanner(bannerId) {
        const dismissedBanners = getStorageItem('dismissed_banners', []);
        if (!dismissedBanners.includes(bannerId)) {
            dismissedBanners.push(bannerId);
            setStorageItem('dismissed_banners', dismissedBanners);
        }

        // Remove banner from DOM
        const bannerElement = document.querySelector(`[data-banner-id="${bannerId}"]`);
        if (bannerElement) {
            // Call onRemove callback if provided
            const banner = this.banners.get(bannerId);
            if (banner && typeof banner.onRemove === 'function') {
                banner.onRemove(bannerElement);
            }
            
            bannerElement.style.animation = 'banner-slide-up 0.3s ease-in-out forwards';
            setTimeout(() => {
                bannerElement.remove();
                this.updateContainerVisibility();
            }, 300);
        }
    }

    /**
     * Show all active (non-dismissed) banners
     */
    showActiveBanners() {
        if (!this.container) return;

        const activeBanners = Array.from(this.banners.values())
            .filter(banner => !this.isBannerDismissed(banner.id))
            .sort((a, b) => (b.priority || 0) - (a.priority || 0));

        activeBanners.forEach(banner => {
            this.renderBanner(banner);
        });

        this.updateContainerVisibility();
    }

    /**
     * Render a banner to the DOM
     * @param {Object} banner - Banner configuration
     */
    renderBanner(banner) {
        const bannerElement = document.createElement('div');
        bannerElement.className = 'banner-item';
        bannerElement.setAttribute('data-banner-id', banner.id);

        const actionsHtml = banner.actions ? banner.actions.map(action => {
            const actionAttribute = action.action ? `data-action="${action.action}"` : '';
            const href = action.url ? `href="${action.url}"` : '#';
            const target = action.url ? 'target="_blank" rel="noopener noreferrer"' : '';
            
            return `<a ${href ? `href="${href}"` : ''} ${target} class="banner-action banner-action-${action.type}" ${actionAttribute}>
                <i class="${action.icon}"></i>
                <span>${action.text}</span>
            </a>`;
        }).join('') : '';

        const dismissButtonHtml = banner.dismissible ? 
            `<button class="banner-dismiss" onclick="bannerService.dismissBanner('${banner.id}')" title="Dismiss">
                <i class="fas fa-times"></i>
            </button>` : '';

        bannerElement.innerHTML = `
            <div class="banner-content">
                <div class="banner-text">
                    <h4 class="banner-title">${banner.title}</h4>
                    <p class="banner-description">${banner.content}</p>
                </div>
                <div class="banner-actions">
                    ${actionsHtml}
                </div>
            </div>
            ${dismissButtonHtml}
        `;

        this.container.appendChild(bannerElement);
        
        // Call onRegister callback if provided
        if (typeof banner.onRegister === 'function') {
            banner.onRegister(bannerElement);
        }
    }

    /**
     * Update container visibility based on active banners
     */
    updateContainerVisibility() {
        if (!this.container) return;

        const hasActiveBanners = this.container.children.length > 0;
        this.container.style.display = hasActiveBanners ? 'block' : 'none';
    }

    /**
     * Clear all dismissed banners (for testing/admin purposes)
     */
    clearDismissedBanners() {
        setStorageItem('dismissed_banners', []);
        location.reload();
    }
}

// Create and export singleton instance
export const bannerService = new BannerService();

// Make it globally available
window.bannerService = bannerService;
