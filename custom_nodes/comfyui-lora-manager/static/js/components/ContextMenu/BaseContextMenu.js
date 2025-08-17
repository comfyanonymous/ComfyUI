export class BaseContextMenu {
    constructor(menuId, cardSelector) {
        this.menu = document.getElementById(menuId);
        this.cardSelector = cardSelector;
        this.currentCard = null;
        
        if (!this.menu) {
            console.error(`Context menu element with ID ${menuId} not found`);
            return;
        }
        
        this.init();
    }

    init() {
        // Hide menu on regular clicks
        document.addEventListener('click', () => this.hideMenu());
        
        // Show menu on right-click on cards
        document.addEventListener('contextmenu', (e) => {
            const card = e.target.closest(this.cardSelector);
            if (!card) {
                this.hideMenu();
                return;
            }
            e.preventDefault();
            this.showMenu(e.clientX, e.clientY, card);
        });

        // Handle menu item clicks
        this.menu.addEventListener('click', (e) => {
            const menuItem = e.target.closest('.context-menu-item');
            if (!menuItem || !this.currentCard) return;

            const action = menuItem.dataset.action;
            if (!action) return;
            
            this.handleMenuAction(action, menuItem);
            this.hideMenu();
        });
    }
    
    handleMenuAction(action, menuItem) {
        // Override in subclass
        console.warn('handleMenuAction not implemented');
    }

    showMenu(x, y, card) {
        this.currentCard = card;
        this.menu.style.display = 'block';

        // Get menu dimensions
        const menuRect = this.menu.getBoundingClientRect();
        
        // Get viewport dimensions
        const viewportWidth = document.documentElement.clientWidth;
        const viewportHeight = document.documentElement.clientHeight;
        
        // Calculate position
        let finalX = x;
        let finalY = y;
        
        // Ensure menu doesn't go offscreen right
        if (x + menuRect.width > viewportWidth) {
            finalX = x - menuRect.width;
        }
        
        // Ensure menu doesn't go offscreen bottom
        if (y + menuRect.height > viewportHeight) {
            finalY = y - menuRect.height;
        }
        
        // Position menu
        this.menu.style.left = `${finalX}px`;
        this.menu.style.top = `${finalY}px`;
    }

    hideMenu() {
        if (this.menu) {
            this.menu.style.display = 'none';
        }
        this.currentCard = null;
    }
}