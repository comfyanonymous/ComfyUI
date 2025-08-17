import { state, getCurrentPageState } from '../state/index.js';
import { getStorageItem, setStorageItem } from './storageHelpers.js';
import { NODE_TYPE_ICONS, DEFAULT_NODE_COLOR } from './constants.js';

/**
 * Utility function to copy text to clipboard with fallback for older browsers
 * @param {string} text - The text to copy to clipboard
 * @param {string} successMessage - Optional success message to show in toast
 * @returns {Promise<boolean>} - Promise that resolves to true if copy was successful
 */
export async function copyToClipboard(text, successMessage = 'Copied to clipboard') {
    try {
        // Modern clipboard API
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'absolute';
            textarea.style.left = '-99999px';
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        }
        
        if (successMessage) {
            showToast(successMessage, 'success');
        }
        return true;
    } catch (err) {
        console.error('Copy failed:', err);
        showToast('Copy failed', 'error');
        return false;
    }
}

export function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Get or create toast container
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.append(toastContainer);
    }
    
    toastContainer.append(toast);

    // Calculate vertical position for stacked toasts
    const existingToasts = Array.from(toastContainer.querySelectorAll('.toast'));
    const toastIndex = existingToasts.indexOf(toast);
    const topOffset = 20; // Base offset from top
    const spacing = 10; // Space between toasts
    
    // Set position based on existing toasts
    toast.style.top = `${topOffset + (toastIndex * (toast.offsetHeight || 60 + spacing))}px`;

    requestAnimationFrame(() => {
        toast.classList.add('show');
        
        // Set timeout based on type
        let timeout = 2000; // Default (info)
        if (type === 'warning' || type === 'error') {
            timeout = 5000;
        }
        
        setTimeout(() => {
            toast.classList.remove('show');
            toast.addEventListener('transitionend', () => {
                toast.remove();
                
                // Reposition remaining toasts
                if (toastContainer) {
                    const remainingToasts = Array.from(toastContainer.querySelectorAll('.toast'));
                    remainingToasts.forEach((t, index) => {
                        t.style.top = `${topOffset + (index * (t.offsetHeight || 60 + spacing))}px`;
                    });
                    
                    // Remove container if empty
                    if (remainingToasts.length === 0) {
                        toastContainer.remove();
                    }
                }
            });
        }, timeout);
    });
}

export function restoreFolderFilter() {
    const activeFolder = getStorageItem('activeFolder');
    const folderTag = activeFolder && document.querySelector(`.tag[data-folder="${activeFolder}"]`);
    if (folderTag) {
        folderTag.classList.add('active');
        filterByFolder(activeFolder);
    }
}

export function initTheme() {
    const savedTheme = getStorageItem('theme') || 'auto';
    applyTheme(savedTheme);
    
    // Update theme when system preference changes (for 'auto' mode)
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        const currentTheme = getStorageItem('theme') || 'auto';
        if (currentTheme === 'auto') {
            applyTheme('auto');
        }
    });
}

export function toggleTheme() {
    const currentTheme = getStorageItem('theme') || 'auto';
    let newTheme;
    
    if (currentTheme === 'light') {
        newTheme = 'dark';
    } else {
        newTheme = 'light';
    }
    
    setStorageItem('theme', newTheme);
    applyTheme(newTheme);
    
    // Force a repaint to ensure theme changes are applied immediately
    document.body.style.display = 'none';
    document.body.offsetHeight; // Trigger a reflow
    document.body.style.display = '';
    
    return newTheme;
}

// Add a new helper function to apply the theme
function applyTheme(theme) {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const htmlElement = document.documentElement;
    
    // Remove any existing theme attributes
    htmlElement.removeAttribute('data-theme');
    
    // Apply the appropriate theme
    if (theme === 'dark' || (theme === 'auto' && prefersDark)) {
        htmlElement.setAttribute('data-theme', 'dark');
        document.body.dataset.theme = 'dark';
    } else {
        htmlElement.setAttribute('data-theme', 'light');
        document.body.dataset.theme = 'light';
    }
    
    // Update the theme-toggle icon state
    updateThemeToggleIcons(theme);
}

// New function to update theme toggle icons
function updateThemeToggleIcons(theme) {
    const themeToggle = document.querySelector('.theme-toggle');
    if (!themeToggle) return;
    
    // Remove any existing active classes
    themeToggle.classList.remove('theme-light', 'theme-dark', 'theme-auto');
    
    // Add the appropriate class based on current theme
    themeToggle.classList.add(`theme-${theme}`);
}

function filterByFolder(folderPath) {
    document.querySelectorAll('.model-card').forEach(card => {
        card.style.display = card.dataset.folder === folderPath ? '' : 'none';
    });
}

export function openCivitai(filePath) {
    const loraCard = document.querySelector(`.model-card[data-filepath="${filePath}"]`);
    if (!loraCard) return;
    
    const metaData = JSON.parse(loraCard.dataset.meta);
    const civitaiId = metaData.modelId;
    const versionId = metaData.id;
    
    if (civitaiId) {
        let url = `https://civitai.com/models/${civitaiId}`;
        if (versionId) {
            url += `?modelVersionId=${versionId}`;
        }
        window.open(url, '_blank');
    } else {
        // 如果没有ID，尝试使用名称搜索
        const modelName = loraCard.dataset.name;
        window.open(`https://civitai.com/models?query=${encodeURIComponent(modelName)}`, '_blank');
    }
}

/**
 * Dynamically positions the search options panel and filter panel
 * based on the current layout and folder tags container height
 */
export function updatePanelPositions() {
    const searchOptionsPanel = document.getElementById('searchOptionsPanel');
    const filterPanel = document.getElementById('filterPanel');
    
    if (!searchOptionsPanel && !filterPanel) return;
    
    // Get the header element
    const header = document.querySelector('.app-header');
    if (!header) return;
    
    // Calculate the position based on the bottom of the header
    const headerRect = header.getBoundingClientRect();
    const topPosition = headerRect.bottom + 5; // Add 5px padding
    
    // Set the positions
    if (searchOptionsPanel) {
      searchOptionsPanel.style.top = `${topPosition}px`;
    }
    
    if (filterPanel) {
      filterPanel.style.top = `${topPosition}px`;
    }
    
    // Adjust panel horizontal position based on the search container
    const searchContainer = document.querySelector('.header-search');
    if (searchContainer) {
      const searchRect = searchContainer.getBoundingClientRect();
      
      // Position the search options panel aligned with the search container
      if (searchOptionsPanel) {
        searchOptionsPanel.style.right = `${window.innerWidth - searchRect.right}px`;
      }
      
      // Position the filter panel aligned with the filter button
      if (filterPanel) {
        const filterButton = document.getElementById('filterButton');
        if (filterButton) {
          const filterRect = filterButton.getBoundingClientRect();
          filterPanel.style.right = `${window.innerWidth - filterRect.right}px`;
        }
      }
    }
}

export function initBackToTop() {
    const button = document.getElementById('backToTopBtn');
    if (!button) return;

    // Get the scrollable container
    const scrollContainer = document.querySelector('.page-content');
    
    // Show/hide button based on scroll position
    const toggleBackToTop = () => {
        const scrollThreshold = window.innerHeight * 0.3;
        if (scrollContainer.scrollTop > scrollThreshold) {
            button.classList.add('visible');
        } else {
            button.classList.remove('visible');
        }
    };

    // Smooth scroll to top
    button.addEventListener('click', () => {
        scrollContainer.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Listen for scroll events on the scrollable container
    scrollContainer.addEventListener('scroll', toggleBackToTop);
    
    // Initial check
    toggleBackToTop();
}

export function getNSFWLevelName(level) {
    if (level === 0) return 'Unknown';
    if (level >= 32) return 'Blocked';
    if (level >= 16) return 'XXX';
    if (level >= 8) return 'X';
    if (level >= 4) return 'R';
    if (level >= 2) return 'PG13';
    if (level >= 1) return 'PG';
    return 'Unknown';
}

export function copyLoraSyntax(card) {
  const usageTips = JSON.parse(card.dataset.usage_tips || "{}");
  const strength = usageTips.strength || 1;
  const baseSyntax = `<lora:${card.dataset.file_name}:${strength}>`;

  // Check if trigger words should be included
  const includeTriggerWords = state.global.settings.includeTriggerWords;

  if (!includeTriggerWords) {
    copyToClipboard(baseSyntax, "LoRA syntax copied to clipboard");
    return;
  }

  // Get trigger words from metadata
  const meta = card.dataset.meta ? JSON.parse(card.dataset.meta) : null;
  const trainedWords = meta?.trainedWords;

  if (
    !trainedWords ||
    !Array.isArray(trainedWords) ||
    trainedWords.length === 0
  ) {
    copyToClipboard(
      baseSyntax,
      "LoRA syntax copied to clipboard (no trigger words found)"
    );
    return;
  }

  let finalSyntax = baseSyntax;

  if (trainedWords.length === 1) {
    // Single group: append trigger words to the same line
    const triggers = trainedWords[0]
      .split(",")
      .map((word) => word.trim())
      .filter((word) => word);
    if (triggers.length > 0) {
      finalSyntax = `${baseSyntax}, ${triggers.join(", ")}`;
    }
    copyToClipboard(
      finalSyntax,
      "LoRA syntax with trigger words copied to clipboard"
    );
  } else {
    // Multiple groups: format with separators
    const groups = trainedWords
      .map((group) => {
        const triggers = group
          .split(",")
          .map((word) => word.trim())
          .filter((word) => word);
        return triggers.join(", ");
      })
      .filter((group) => group);

    if (groups.length > 0) {
      // Use separator between all groups except the first
      finalSyntax = baseSyntax + ", " + groups[0];
      for (let i = 1; i < groups.length; i++) {
        finalSyntax += `\n${"-".repeat(17)}\n${groups[i]}`;
      }
    }
    copyToClipboard(
      finalSyntax,
      "LoRA syntax with trigger word groups copied to clipboard"
    );
  }
}

/**
 * Sends LoRA syntax to the active ComfyUI workflow
 * @param {string} loraSyntax - The LoRA syntax to send
 * @param {boolean} replaceMode - Whether to replace existing LoRAs (true) or append (false)
 * @param {string} syntaxType - The type of syntax ('lora' or 'recipe')
 * @returns {Promise<boolean>} - Whether the operation was successful
 */
export async function sendLoraToWorkflow(loraSyntax, replaceMode = false, syntaxType = 'lora') {
  try {
    // Get registry information from the new endpoint
    const registryResponse = await fetch('/api/get-registry');
    const registryData = await registryResponse.json();
    
    if (!registryData.success) {
      // Handle specific error cases
      if (registryData.error === 'Standalone Mode Active') {
        // Standalone mode - show warning with specific message
        showToast(registryData.message || 'Cannot interact with ComfyUI in standalone mode', 'warning');
        return false;
      } else {
        // Other errors - show error toast
        showToast(registryData.message || registryData.error || 'Failed to get workflow information', 'error');
        return false;
      }
    }
    
    // Success case - check node count
    if (registryData.data.node_count === 0) {
      // No nodes found - show warning
      showToast('No supported target nodes found in workflow', 'warning');
      return false;
    } else if (registryData.data.node_count > 1) {
      // Multiple nodes - show selector
      showNodeSelector(registryData.data.nodes, loraSyntax, replaceMode, syntaxType);
      return true;
    } else {
      // Single node - send directly
      const nodeId = Object.keys(registryData.data.nodes)[0];
      return await sendToSpecificNode([nodeId], loraSyntax, replaceMode, syntaxType);
    }
  } catch (error) {
    console.error('Failed to get registry:', error);
    showToast('Failed to communicate with ComfyUI', 'error');
    return false;
  }
}

/**
 * Send LoRA to specific nodes
 * @param {Array|undefined} nodeIds - Array of node IDs or undefined for desktop mode
 * @param {string} loraSyntax - The LoRA syntax to send
 * @param {boolean} replaceMode - Whether to replace existing LoRAs
 * @param {string} syntaxType - The type of syntax ('lora' or 'recipe')
 */
async function sendToSpecificNode(nodeIds, loraSyntax, replaceMode, syntaxType) {
  try {
    // Call the backend API to update the lora code
    const response = await fetch('/api/update-lora-code', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        node_ids: nodeIds,
        lora_code: loraSyntax,
        mode: replaceMode ? 'replace' : 'append'
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      // Use different toast messages based on syntax type
      if (syntaxType === 'recipe') {
        showToast(`Recipe ${replaceMode ? 'replaced' : 'added'} to workflow`, 'success');
      } else {
        showToast(`LoRA ${replaceMode ? 'replaced' : 'added'} to workflow`, 'success');
      }
      return true;
    } else {
      showToast(result.error || `Failed to send ${syntaxType === 'recipe' ? 'recipe' : 'LoRA'} to workflow`, 'error');
      return false;
    }
  } catch (error) {
    console.error('Failed to send to workflow:', error);
    showToast(`Failed to send ${syntaxType === 'recipe' ? 'recipe' : 'LoRA'} to workflow`, 'error');
    return false;
  }
}

// Global variable to track active node selector state
let nodeSelectorState = {
  isActive: false,
  clickHandler: null,
  selectorClickHandler: null
};

/**
 * Show node selector popup near mouse position
 * @param {Object} nodes - Registry nodes data
 * @param {string} loraSyntax - The LoRA syntax to send
 * @param {boolean} replaceMode - Whether to replace existing LoRAs
 * @param {string} syntaxType - The type of syntax ('lora' or 'recipe')
 */
function showNodeSelector(nodes, loraSyntax, replaceMode, syntaxType) {
  const selector = document.getElementById('nodeSelector');
  if (!selector) return;
  
  // Clean up any existing state
  hideNodeSelector();
  
  // Generate node list HTML with icons and proper colors
  const nodeItems = Object.values(nodes).map(node => {
    const iconClass = NODE_TYPE_ICONS[node.type] || 'fas fa-question-circle';
    const bgColor = node.bgcolor || DEFAULT_NODE_COLOR;
    
    return `
      <div class="node-item" data-node-id="${node.id}">
        <div class="node-icon-indicator" style="background-color: ${bgColor}">
          <i class="${iconClass}"></i>
        </div>
        <span>#${node.id} ${node.title}</span>
      </div>
    `;
  }).join('');
  
  // Add header with action mode indicator
  const actionType = syntaxType === 'recipe' ? 'Recipe' : 'LoRA';
  const actionMode = replaceMode ? 'Replace' : 'Append';
  
  selector.innerHTML = `
    <div class="node-selector-header">
      <span class="selector-action-type">${actionMode} ${actionType}</span>
      <span class="selector-instruction">Select target node</span>
    </div>
    ${nodeItems}
    <div class="node-item send-all-item" data-action="send-all">
      <div class="node-icon-indicator all-nodes">
        <i class="fas fa-broadcast-tower"></i>
      </div>
      <span>Send to All</span>
    </div>
  `;
  
  // Position near mouse
  positionNearMouse(selector);
  
  // Show selector
  selector.style.display = 'block';
  nodeSelectorState.isActive = true;
  
  // Setup event listeners with proper cleanup
  setupNodeSelectorEvents(selector, nodes, loraSyntax, replaceMode, syntaxType);
}

/**
 * Setup event listeners for node selector
 * @param {HTMLElement} selector - The selector element
 * @param {Object} nodes - Registry nodes data
 * @param {string} loraSyntax - The LoRA syntax to send
 * @param {boolean} replaceMode - Whether to replace existing LoRAs
 * @param {string} syntaxType - The type of syntax ('lora' or 'recipe')
 */
function setupNodeSelectorEvents(selector, nodes, loraSyntax, replaceMode, syntaxType) {
  // Clean up any existing event listeners
  cleanupNodeSelectorEvents();
  
  // Handle clicks outside to close
  nodeSelectorState.clickHandler = (e) => {
    if (!selector.contains(e.target)) {
      hideNodeSelector();
    }
  };
  
  // Handle node selection
  nodeSelectorState.selectorClickHandler = async (e) => {
    const nodeItem = e.target.closest('.node-item');
    if (!nodeItem) return;
    
    e.stopPropagation();
    
    const action = nodeItem.dataset.action;
    const nodeId = nodeItem.dataset.nodeId;
    
    if (action === 'send-all') {
      // Send to all nodes
      const allNodeIds = Object.keys(nodes);
      await sendToSpecificNode(allNodeIds, loraSyntax, replaceMode, syntaxType);
    } else if (nodeId) {
      // Send to specific node
      await sendToSpecificNode([nodeId], loraSyntax, replaceMode, syntaxType);
    }
    
    hideNodeSelector();
  };
  
  // Add event listeners with a small delay to prevent immediate triggering
  setTimeout(() => {
    if (nodeSelectorState.isActive) {
      document.addEventListener('click', nodeSelectorState.clickHandler);
      selector.addEventListener('click', nodeSelectorState.selectorClickHandler);
    }
  }, 100);
}

/**
 * Clean up node selector event listeners
 */
function cleanupNodeSelectorEvents() {
  if (nodeSelectorState.clickHandler) {
    document.removeEventListener('click', nodeSelectorState.clickHandler);
    nodeSelectorState.clickHandler = null;
  }
  
  if (nodeSelectorState.selectorClickHandler) {
    const selector = document.getElementById('nodeSelector');
    if (selector) {
      selector.removeEventListener('click', nodeSelectorState.selectorClickHandler);
    }
    nodeSelectorState.selectorClickHandler = null;
  }
}

/**
 * Hide node selector
 */
function hideNodeSelector() {
  const selector = document.getElementById('nodeSelector');
  if (selector) {
    selector.style.display = 'none';
    selector.innerHTML = ''; // Clear content to prevent memory leaks
  }
  
  // Clean up event listeners
  cleanupNodeSelectorEvents();
  nodeSelectorState.isActive = false;
}

/**
 * Position element near mouse cursor
 * @param {HTMLElement} element - Element to position
 */
function positionNearMouse(element) {
  // Get current mouse position from last mouse event or use default
  const mouseX = window.lastMouseX || window.innerWidth / 2;
  const mouseY = window.lastMouseY || window.innerHeight / 2;
  
  // Show element temporarily to get dimensions
  element.style.visibility = 'hidden';
  element.style.display = 'block';
  
  const rect = element.getBoundingClientRect();
  const viewportWidth = document.documentElement.clientWidth;
  const viewportHeight = document.documentElement.clientHeight;
  
  // Calculate position with offset from mouse
  let x = mouseX + 10;
  let y = mouseY + 10;
  
  // Ensure element doesn't go offscreen
  if (x + rect.width > viewportWidth) {
    x = mouseX - rect.width - 10;
  }
  
  if (y + rect.height > viewportHeight) {
    y = mouseY - rect.height - 10;
  }
  
  // Apply position
  element.style.left = `${x}px`;
  element.style.top = `${y}px`;
  element.style.visibility = 'visible';
}

// Track mouse position for node selector positioning
document.addEventListener('mousemove', (e) => {
  window.lastMouseX = e.clientX;
  window.lastMouseY = e.clientY;
});

/**
 * Opens the example images folder for a specific model
 * @param {string} modelHash - The SHA256 hash of the model
 */
export async function openExampleImagesFolder(modelHash) {
  try {
    const response = await fetch('/api/open-example-images-folder', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_hash: modelHash
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      showToast('Opening example images folder', 'success');
      return true;
    } else {
      showToast(result.error || 'Failed to open example images folder', 'error');
      return false;
    }
  } catch (error) {
    console.error('Failed to open example images folder:', error);
    showToast('Failed to open example images folder', 'error');
    return false;
  }
}

/**
     * Update the folder tags display with new folder list
     * @param {Array} folders - List of folder names
     */
export function updateFolderTags(folders) {
  const folderTagsContainer = document.querySelector('.folder-tags');
  if (!folderTagsContainer) return;

  // Keep track of currently selected folder
  const pageState = getCurrentPageState();
  const currentFolder = pageState.activeFolder;

  // Create HTML for folder tags
  const tagsHTML = folders.map(folder => {
      const isActive = folder === currentFolder;
      return `<div class="tag ${isActive ? 'active' : ''}" data-folder="${folder}">${folder}</div>`;
  }).join('');

  // Update the container
  folderTagsContainer.innerHTML = tagsHTML;

  // Scroll active folder into view (no need to reattach click handlers)
  const activeTag = folderTagsContainer.querySelector(`.tag[data-folder="${currentFolder}"]`);
  if (activeTag) {
      activeTag.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}