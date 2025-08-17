export class ModalManager {
    constructor() {
        this.modals = new Map();
        this.scrollPosition = 0;
        this.currentOpenModal = null; // Track currently open modal
        this.mouseDownOnBackground = false; // Track if mousedown happened on modal background
    }

    initialize() {
        if (this.initialized) return;
        
        this.boundHandleEscape = this.handleEscape.bind(this);
        
        // Register all modals - only if they exist in the current page
        const modelModal = document.getElementById('modelModal');
        if (modelModal) {
            this.registerModal('modelModal', {
                element: modelModal,
                onClose: () => {
                    this.getModal('modelModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add checkpointDownloadModal registration
        const checkpointDownloadModal = document.getElementById('checkpointDownloadModal');
        if (checkpointDownloadModal) {
            this.registerModal('checkpointDownloadModal', {
                element: checkpointDownloadModal,
                onClose: () => {
                    this.getModal('checkpointDownloadModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                }
            });
        }
        
        const deleteModal = document.getElementById('deleteModal');
        if (deleteModal) {
            this.registerModal('deleteModal', {
                element: deleteModal,
                onClose: () => {
                    this.getModal('deleteModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }
        
        // Add excludeModal registration
        const excludeModal = document.getElementById('excludeModal');
        if (excludeModal) {
            this.registerModal('excludeModal', {
                element: excludeModal,
                onClose: () => {
                    this.getModal('excludeModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }

        // Add downloadModal registration
        const downloadModal = document.getElementById('downloadModal');
        if (downloadModal) {
            this.registerModal('downloadModal', {
                element: downloadModal,
                onClose: () => {
                    this.getModal('downloadModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                }
            });
        }

        // Add settingsModal registration
        const settingsModal = document.getElementById('settingsModal');
        if (settingsModal) {
            this.registerModal('settingsModal', {
                element: settingsModal,
                onClose: () => {
                    this.getModal('settingsModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add moveModal registration
        const moveModal = document.getElementById('moveModal');
        if (moveModal) {
            this.registerModal('moveModal', {
                element: moveModal,
                onClose: () => {
                    this.getModal('moveModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                }
            });
        }
        
        // Add supportModal registration
        const supportModal = document.getElementById('supportModal');
        if (supportModal) {
            this.registerModal('supportModal', {
                element: supportModal,
                onClose: () => {
                    this.getModal('supportModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add updateModal registration
        const updateModal = document.getElementById('updateModal');
        if (updateModal) {
            this.registerModal('updateModal', {
                element: updateModal,
                onClose: () => {
                    this.getModal('updateModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add importModal registration
        const importModal = document.getElementById('importModal');
        if (importModal) {
            this.registerModal('importModal', {
                element: importModal,
                onClose: () => {
                    this.getModal('importModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');   
                }
            });
        }

        // Add recipeModal registration
        const recipeModal = document.getElementById('recipeModal');
        if (recipeModal) {
            this.registerModal('recipeModal', {
                element: recipeModal,
                onClose: () => {
                    this.getModal('recipeModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add duplicateDeleteModal registration
        const duplicateDeleteModal = document.getElementById('duplicateDeleteModal');
        if (duplicateDeleteModal) {
            this.registerModal('duplicateDeleteModal', {
                element: duplicateDeleteModal,
                onClose: () => {
                    this.getModal('duplicateDeleteModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }

        // Add modelDuplicateDeleteModal registration
        const modelDuplicateDeleteModal = document.getElementById('modelDuplicateDeleteModal');
        if (modelDuplicateDeleteModal) {
            this.registerModal('modelDuplicateDeleteModal', {
                element: modelDuplicateDeleteModal,
                onClose: () => {
                    this.getModal('modelDuplicateDeleteModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }
        
        // Add clearCacheModal registration
        const clearCacheModal = document.getElementById('clearCacheModal');
        if (clearCacheModal) {
            this.registerModal('clearCacheModal', {
                element: clearCacheModal,
                onClose: () => {
                    this.getModal('clearCacheModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }
        
        // Add bulkDeleteModal registration
        const bulkDeleteModal = document.getElementById('bulkDeleteModal');
        if (bulkDeleteModal) {
            this.registerModal('bulkDeleteModal', {
                element: bulkDeleteModal,
                onClose: () => {
                    this.getModal('bulkDeleteModal').element.classList.remove('show');
                    document.body.classList.remove('modal-open');
                }
            });
        }

        // Add helpModal registration
        const helpModal = document.getElementById('helpModal');
        if (helpModal) {
            this.registerModal('helpModal', {
                element: helpModal,
                onClose: () => {
                    this.getModal('helpModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add relinkCivitaiModal registration
        const relinkCivitaiModal = document.getElementById('relinkCivitaiModal');
        if (relinkCivitaiModal) {
            this.registerModal('relinkCivitaiModal', {
                element: relinkCivitaiModal,
                onClose: () => {
                    this.getModal('relinkCivitaiModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        // Add exampleAccessModal registration
        const exampleAccessModal = document.getElementById('exampleAccessModal');
        if (exampleAccessModal) {
            this.registerModal('exampleAccessModal', {
                element: exampleAccessModal,
                onClose: () => {
                    this.getModal('exampleAccessModal').element.style.display = 'none';
                    document.body.classList.remove('modal-open');
                },
                closeOnOutsideClick: true
            });
        }

        document.addEventListener('keydown', this.boundHandleEscape);
        this.initialized = true;
    }

    registerModal(id, config) {
        this.modals.set(id, {
            element: config.element,
            onClose: config.onClose,
            isOpen: false
        });

        // Add click outside handler if specified in config
        if (config.closeOnOutsideClick) {
            // Track mousedown on modal background
            config.element.addEventListener('mousedown', (e) => {
                if (e.target === config.element) {
                    this.mouseDownOnBackground = true;
                } else {
                    this.mouseDownOnBackground = false;
                }
            });
            
            // Only close if mouseup is also on the background
            config.element.addEventListener('mouseup', (e) => {
                if (e.target === config.element && this.mouseDownOnBackground) {
                    this.closeModal(id);
                }
                // Reset flag regardless of target
                this.mouseDownOnBackground = false;
            });
            
            // Cancel the flag if mouse leaves the document entirely
            document.addEventListener('mouseleave', () => {
                this.mouseDownOnBackground = false;
            });
        }
    }

    getModal(id) {
        return this.modals.get(id);
    }

    // Check if any modal is currently open
    isAnyModalOpen() {
        for (const [id, modal] of this.modals) {
            if (modal.isOpen) {
                return id;
            }
        }
        return null;
    }

    showModal(id, content = null, onCloseCallback = null, cleanupCallback = null) {
        const modal = this.getModal(id);
        if (!modal) return;

        // Close any open modal before showing the new one
        const openModalId = this.isAnyModalOpen();
        if (openModalId && openModalId !== id) {
            this.closeModal(openModalId);
        }

        if (content) {
            modal.element.innerHTML = content;
        }

        // Store callback
        modal.onCloseCallback = onCloseCallback;
        modal.cleanupCallback = cleanupCallback;

        // Store current scroll position before showing modal
        this.scrollPosition = window.scrollY;

        if (
          id === "deleteModal" ||
          id === "excludeModal" ||
          id === "duplicateDeleteModal" ||
          id === "modelDuplicateDeleteModal" ||
          id === "clearCacheModal" ||
          id === "bulkDeleteModal"
        ) {
          modal.element.classList.add("show");
        } else {
          modal.element.style.display = "block";
        }

        modal.isOpen = true;
        this.currentOpenModal = id; // Update currently open modal
        document.body.style.top = `-${this.scrollPosition}px`;
        document.body.classList.add('modal-open');
    }

    closeModal(id) {
        const modal = this.getModal(id);
        if (!modal) return;

        modal.onClose();
        modal.isOpen = false;

        // Clear current open modal if this is the one being closed
        if (this.currentOpenModal === id) {
            this.currentOpenModal = null;
        }

        // Remove fixed positioning and restore scroll position
        document.body.classList.remove('modal-open');
        document.body.style.top = '';
        window.scrollTo(0, this.scrollPosition);

        // Execute onClose callback if exists
        if (modal.onCloseCallback) {
            modal.onCloseCallback();
            modal.onCloseCallback = null;
        }

        if (modal.cleanupCallback) {
            modal.cleanupCallback();
            modal.cleanupCallback = null;
        }
    }

    handleEscape(e) {
        if (e.key === 'Escape') {
            // Close the current open modal if it exists
            if (this.currentOpenModal) {
                this.closeModal(this.currentOpenModal);
            }
        }
    }

    toggleModal(id, content = null, onCloseCallback = null) {
        const modal = this.getModal(id);
        if (!modal) return;
        
        // If this modal is already open, close it
        if (modal.isOpen) {
            this.closeModal(id);
            return;
        }
        
        // Otherwise, show the modal
        this.showModal(id, content, onCloseCallback);
    }
}

// Create and export a singleton instance
export const modalManager = new ModalManager();