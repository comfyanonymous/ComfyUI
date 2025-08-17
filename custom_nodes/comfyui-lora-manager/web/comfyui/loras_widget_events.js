import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { createMenuItem, createDropIndicator } from "./loras_widget_components.js";
import { parseLoraValue, formatLoraValue, syncClipStrengthIfCollapsed, saveRecipeDirectly, copyToClipboard, showToast, moveLoraByDirection, getDropTargetIndex } from "./loras_widget_utils.js";

// Function to handle strength adjustment via dragging
export function handleStrengthDrag(name, initialStrength, initialX, event, widget, isClipStrength = false) {
  // Calculate drag sensitivity (how much the strength changes per pixel)
  // Using 0.01 per 10 pixels of movement
  const sensitivity = 0.001;
  
  // Get the current mouse position
  const currentX = event.clientX;
  
  // Calculate the distance moved
  const deltaX = currentX - initialX;
  
  // Calculate the new strength value based on movement
  // Moving right increases, moving left decreases
  let newStrength = Number(initialStrength) + (deltaX * sensitivity);
  
  // Limit the strength to reasonable bounds (now between -10 and 10)
  newStrength = Math.max(-10, Math.min(10, newStrength));
  newStrength = Number(newStrength.toFixed(2));
  
  // Update the lora data
  const lorasData = parseLoraValue(widget.value);
  const loraIndex = lorasData.findIndex(l => l.name === name);
  
  if (loraIndex >= 0) {
    // Update the appropriate strength property based on isClipStrength flag
    if (isClipStrength) {
      lorasData[loraIndex].clipStrength = newStrength;
    } else {
      lorasData[loraIndex].strength = newStrength;
      // Sync clipStrength if collapsed
      syncClipStrengthIfCollapsed(lorasData[loraIndex]);
    }
    
    // Update the widget value
    widget.value = formatLoraValue(lorasData);
    
    // Force re-render via callback
    if (widget.callback) {
      widget.callback(widget.value);
    }
  }
}

// Function to handle proportional strength adjustment for all LoRAs via header dragging
export function handleAllStrengthsDrag(initialStrengths, initialX, event, widget) {
  // Define sensitivity (less sensitive than individual adjustment)
  const sensitivity = 0.0005;
  
  // Get current mouse position
  const currentX = event.clientX;
  
  // Calculate the distance moved
  const deltaX = currentX - initialX;
  
  // Calculate adjustment factor (1.0 means no change, >1.0 means increase, <1.0 means decrease)
  // For positive deltaX, we want to increase strengths, for negative we want to decrease
  const adjustmentFactor = 1.0 + (deltaX * sensitivity);
  
  // Ensure adjustment factor is reasonable (prevent extreme changes)
  const limitedFactor = Math.max(0.01, Math.min(3.0, adjustmentFactor));
  
  // Get current loras data
  const lorasData = parseLoraValue(widget.value);
  
  // Apply the adjustment factor to each LoRA's strengths
  lorasData.forEach((loraData, index) => {
    // Get initial strengths for this LoRA
    const initialModelStrength = initialStrengths[index].modelStrength;
    const initialClipStrength = initialStrengths[index].clipStrength;
    
    // Apply the adjustment factor to both strengths
    let newModelStrength = (initialModelStrength * limitedFactor).toFixed(2);
    let newClipStrength = (initialClipStrength * limitedFactor).toFixed(2);
    
    // Limit the values to reasonable bounds (-10 to 10)
    newModelStrength = Math.max(-10, Math.min(10, newModelStrength));
    newClipStrength = Math.max(-10, Math.min(10, newClipStrength));
    
    // Update strengths
    lorasData[index].strength = Number(newModelStrength);
    lorasData[index].clipStrength = Number(newClipStrength);
  });
  
  // Update widget value
  widget.value = formatLoraValue(lorasData);
  
  // Force re-render via callback
  if (widget.callback) {
    widget.callback(widget.value);
  }
}

// Function to initialize drag operation
export function initDrag(dragEl, name, widget, isClipStrength = false, previewTooltip, renderFunction) {
  let isDragging = false;
  let initialX = 0;
  let initialStrength = 0;
  
  // Create a style element for drag cursor override if it doesn't exist
  if (!document.getElementById('comfy-lora-drag-style')) {
    const styleEl = document.createElement('style');
    styleEl.id = 'comfy-lora-drag-style';
    styleEl.textContent = `
      body.comfy-lora-dragging,
      body.comfy-lora-dragging * {
        cursor: ew-resize !important;
      }
    `;
    document.head.appendChild(styleEl);
  }
  
  // Create a drag handler
  dragEl.addEventListener('mousedown', (e) => {
    // Skip if clicking on toggle or strength control areas
    if (e.target.closest('.comfy-lora-toggle') || 
        e.target.closest('input') || 
        e.target.closest('.comfy-lora-arrow')) {
      return;
    }
    
    // Store initial values
    const lorasData = parseLoraValue(widget.value);
    const loraData = lorasData.find(l => l.name === name);
    
    if (!loraData) return;
    
    initialX = e.clientX;
    initialStrength = isClipStrength ? loraData.clipStrength : loraData.strength;
    isDragging = true;
    
    // Add class to body to enforce cursor style globally
    document.body.classList.add('comfy-lora-dragging');
    
    // Prevent text selection during drag
    e.preventDefault();
  });
  
  // Use the document for move and up events to ensure drag continues
  // even if mouse leaves the element
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    // Call the strength adjustment function
    handleStrengthDrag(name, initialStrength, initialX, e, widget, isClipStrength);
    
    // Force re-render to show updated strength value
    if (renderFunction) {
      renderFunction(widget.value, widget);
    }
    
    // Prevent showing the preview tooltip during drag
    if (previewTooltip) {
      previewTooltip.hide();
    }
  });
  
  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      // Remove the class to restore normal cursor behavior
      document.body.classList.remove('comfy-lora-dragging');
    }
  });
}

// Function to initialize header drag for proportional strength adjustment
export function initHeaderDrag(headerEl, widget, renderFunction) {
  let isDragging = false;
  let initialX = 0;
  let initialStrengths = [];
  
  // Add cursor style to indicate draggable
  headerEl.style.cursor = 'ew-resize';
  
  // Create a drag handler
  headerEl.addEventListener('mousedown', (e) => {
    // Skip if clicking on toggle or other interactive elements
    if (e.target.closest('.comfy-lora-toggle') || 
        e.target.closest('input')) {
      return;
    }
    
    // Store initial X position
    initialX = e.clientX;
    
    // Store initial strengths of all LoRAs
    const lorasData = parseLoraValue(widget.value);
    initialStrengths = lorasData.map(lora => ({
      modelStrength: Number(lora.strength),
      clipStrength: Number(lora.clipStrength)
    }));
    
    isDragging = true;
    
    // Add class to body to enforce cursor style globally
    document.body.classList.add('comfy-lora-dragging');
    
    // Prevent text selection during drag
    e.preventDefault();
  });
  
  // Handle mouse move for dragging
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    // Call the strength adjustment function
    handleAllStrengthsDrag(initialStrengths, initialX, e, widget);
    
    // Force re-render to show updated strength values
    if (renderFunction) {
      renderFunction(widget.value, widget);
    }
  });
  
  // Handle mouse up to end dragging
  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      // Remove the class to restore normal cursor behavior
      document.body.classList.remove('comfy-lora-dragging');
    }
  });
}

// Function to initialize drag-and-drop for reordering
export function initReorderDrag(dragHandle, loraName, widget, renderFunction) {
  let isDragging = false;
  let draggedElement = null;
  let dropIndicator = null;
  let container = null;
  let scale = 1;
  
  dragHandle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    isDragging = true;
    draggedElement = dragHandle.closest('.comfy-lora-entry');
    container = draggedElement.parentElement;
    
    // Add dragging class and visual feedback
    draggedElement.classList.add('comfy-lora-dragging');
    draggedElement.style.opacity = '0.5';
    draggedElement.style.transform = 'scale(0.98)';
    
    // Create single drop indicator with absolute positioning
    dropIndicator = createDropIndicator();
    
    // Make container relatively positioned for absolute indicator
    const originalPosition = container.style.position;
    container.style.position = 'relative';
    container.appendChild(dropIndicator);
    
    // Store original position for cleanup
    container._originalPosition = originalPosition;
    
    // Add global cursor style
    document.body.style.cursor = 'grabbing';

    // Store workflow scale for accurate positioning
    scale = app.canvas.ds.scale;
  });
  
  document.addEventListener('mousemove', (e) => {
    if (!isDragging || !draggedElement || !dropIndicator) return;
    
    const targetIndex = getDropTargetIndex(container, e.clientY);
    const entries = container.querySelectorAll('.comfy-lora-entry, .comfy-lora-clip-entry');
    
    if (targetIndex === 0) {
      // Show at top
      const firstEntry = entries[0];
      if (firstEntry) {
        const rect = firstEntry.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        dropIndicator.style.top = `${(rect.top - containerRect.top - 2) / scale}px`;
        dropIndicator.style.opacity = '1';
      }
    } else if (targetIndex < entries.length) {
      // Show between entries
      const targetEntry = entries[targetIndex];
      if (targetEntry) {
        const rect = targetEntry.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        dropIndicator.style.top = `${(rect.top - containerRect.top - 2) / scale}px`;
        dropIndicator.style.opacity = '1';
      }
    } else {
      // Show at bottom
      const lastEntry = entries[entries.length - 1];
      if (lastEntry) {
        const rect = lastEntry.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        dropIndicator.style.top = `${(rect.bottom - containerRect.top + 2) / scale}px`;
        dropIndicator.style.opacity = '1';
      }
    }
  });
  
  document.addEventListener('mouseup', (e) => {
    if (!isDragging || !draggedElement) return;
    
    const targetIndex = getDropTargetIndex(container, e.clientY);
    
    // Get current LoRA data
    const lorasData = parseLoraValue(widget.value);
    const currentIndex = lorasData.findIndex(l => l.name === loraName);
    
    if (currentIndex !== -1 && currentIndex !== targetIndex) {
      // Calculate actual target index (excluding clip entries from count)
      const loraEntries = container.querySelectorAll('.comfy-lora-entry');
      let actualTargetIndex = targetIndex;
      
      // Adjust target index if it's beyond the number of actual LoRA entries
      if (actualTargetIndex > loraEntries.length) {
        actualTargetIndex = loraEntries.length;
      }
      
      // Move the LoRA
      const newLoras = [...lorasData];
      const [moved] = newLoras.splice(currentIndex, 1);
      newLoras.splice(actualTargetIndex > currentIndex ? actualTargetIndex - 1 : actualTargetIndex, 0, moved);
      
      widget.value = formatLoraValue(newLoras);
      
      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      // Re-render
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
    
    // Cleanup
    isDragging = false;
    if (draggedElement) {
      draggedElement.classList.remove('comfy-lora-dragging');
      draggedElement.style.opacity = '';
      draggedElement.style.transform = '';
      draggedElement = null;
    }
    
    if (dropIndicator && container) {
      container.removeChild(dropIndicator);
      // Restore original position
      container.style.position = container._originalPosition || '';
      delete container._originalPosition;
      dropIndicator = null;
    }
    
    // Reset cursor
    document.body.style.cursor = '';
    container = null;
  });
}

// Function to handle keyboard navigation
export function handleKeyboardNavigation(event, selectedLora, widget, renderFunction, selectLora) {
  if (!selectedLora) return false;
  
  const lorasData = parseLoraValue(widget.value);
  let handled = false;
  
  // Check for Ctrl/Cmd modifier for reordering
  if (event.ctrlKey || event.metaKey) {
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        const newLorasUp = moveLoraByDirection(lorasData, selectedLora, 'up');
        widget.value = formatLoraValue(newLorasUp);
        if (widget.callback) widget.callback(widget.value);
        if (renderFunction) renderFunction(widget.value, widget);
        handled = true;
        break;
        
      case 'ArrowDown':
        event.preventDefault();
        const newLorasDown = moveLoraByDirection(lorasData, selectedLora, 'down');
        widget.value = formatLoraValue(newLorasDown);
        if (widget.callback) widget.callback(widget.value);
        if (renderFunction) renderFunction(widget.value, widget);
        handled = true;
        break;
        
      case 'Home':
        event.preventDefault();
        const newLorasTop = moveLoraByDirection(lorasData, selectedLora, 'top');
        widget.value = formatLoraValue(newLorasTop);
        if (widget.callback) widget.callback(widget.value);
        if (renderFunction) renderFunction(widget.value, widget);
        handled = true;
        break;
        
      case 'End':
        event.preventDefault();
        const newLorasBottom = moveLoraByDirection(lorasData, selectedLora, 'bottom');
        widget.value = formatLoraValue(newLorasBottom);
        if (widget.callback) widget.callback(widget.value);
        if (renderFunction) renderFunction(widget.value, widget);
        handled = true;
        break;
    }
  } else {
    // Normal navigation without Ctrl/Cmd
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        const currentIndex = lorasData.findIndex(l => l.name === selectedLora);
        if (currentIndex > 0) {
          selectLora(lorasData[currentIndex - 1].name);
        }
        handled = true;
        break;
        
      case 'ArrowDown':
        event.preventDefault();
        const currentIndexDown = lorasData.findIndex(l => l.name === selectedLora);
        if (currentIndexDown < lorasData.length - 1) {
          selectLora(lorasData[currentIndexDown + 1].name);
        }
        handled = true;
        break;
        
      case 'Delete':
      case 'Backspace':
        event.preventDefault();
        const filtered = lorasData.filter(l => l.name !== selectedLora);
        widget.value = formatLoraValue(filtered);
        if (widget.callback) widget.callback(widget.value);
        if (renderFunction) renderFunction(widget.value, widget);
        selectLora(null); // Clear selection
        handled = true;
        break;
    }
  }
  
  return handled;
}

// Function to create context menu
export function createContextMenu(x, y, loraName, widget, previewTooltip, renderFunction) {
  // Hide preview tooltip first
  if (previewTooltip) {
    previewTooltip.hide();
  }

  // Remove existing context menu if any
  const existingMenu = document.querySelector('.comfy-lora-context-menu');
  if (existingMenu) {
    existingMenu.remove();
  }

  const menu = document.createElement('div');
  menu.className = 'comfy-lora-context-menu';
  Object.assign(menu.style, {
    position: 'fixed',
    left: `${x}px`,
    top: `${y}px`,
    backgroundColor: 'rgba(30, 30, 30, 0.95)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    padding: '4px 0',
    zIndex: 1000,
    boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
    minWidth: '180px',
  });

  // View on Civitai option with globe icon
  const viewOnCivitaiOption = createMenuItem(
    'View on Civitai',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>',
    async () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      try {
        // Get Civitai URL from API
        const response = await api.fetchApi(`/loras/civitai-url?name=${encodeURIComponent(loraName)}`, {
          method: 'GET'
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || 'Failed to get Civitai URL');
        }
        
        const data = await response.json();
        if (data.success && data.civitai_url) {
          // Open the URL in a new tab
          window.open(data.civitai_url, '_blank');
        } else {
          // Show error message if no Civitai URL
          showToast('This LoRA has no associated Civitai URL', 'warning');
        }
      } catch (error) {
        console.error('Error getting Civitai URL:', error);
        showToast(error.message || 'Failed to get Civitai URL', 'error');
      }
    }
  );

  // Delete option with trash icon
  const deleteOption = createMenuItem(
    'Delete', 
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      const lorasData = parseLoraValue(widget.value).filter(l => l.name !== loraName);
      widget.value = formatLoraValue(lorasData);

      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      // Re-render
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
  );

  // New option: Copy Notes with note icon
  const copyNotesOption = createMenuItem(
    'Copy Notes',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
    async () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      try {
        // Get notes from API
        const response = await api.fetchApi(`/loras/get-notes?name=${encodeURIComponent(loraName)}`, {
          method: 'GET'
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || 'Failed to get notes');
        }
        
        const data = await response.json();
        if (data.success) {
          const notes = data.notes || '';
          if (notes.trim()) {
            await copyToClipboard(notes, 'Notes copied to clipboard');
          } else {
            showToast('No notes available for this LoRA', 'info');
          }
        } else {
          throw new Error(data.error || 'Failed to get notes');
        }
      } catch (error) {
        console.error('Error getting notes:', error);
        showToast(error.message || 'Failed to get notes', 'error');
      }
    }
  );

  // New option: Copy Trigger Words with tag icon
  const copyTriggerWordsOption = createMenuItem(
    'Copy Trigger Words',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>',
    async () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      try {
        // Get trigger words from API
        const response = await api.fetchApi(`/loras/get-trigger-words?name=${encodeURIComponent(loraName)}`, {
          method: 'GET'
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || 'Failed to get trigger words');
        }
        
        const data = await response.json();
        if (data.success) {
          const triggerWords = data.trigger_words || [];
          if (triggerWords.length > 0) {
            // Join trigger words with commas
            const triggerWordsText = triggerWords.join(', ');
            await copyToClipboard(triggerWordsText, 'Trigger words copied to clipboard');
          } else {
            showToast('No trigger words available for this LoRA', 'info');
          }
        } else {
          throw new Error(data.error || 'Failed to get trigger words');
        }
      } catch (error) {
        console.error('Error getting trigger words:', error);
        showToast(error.message || 'Failed to get trigger words', 'error');
      }
    }
  );

  // Save recipe option with bookmark icon
  const saveOption = createMenuItem(
    'Save Recipe',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      saveRecipeDirectly();
    }
  );

  // Move Up option with arrow up icon
  const moveUpOption = createMenuItem(
    'Move Up',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 15l-6-6-6 6"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      const lorasData = parseLoraValue(widget.value);
      const newLoras = moveLoraByDirection(lorasData, loraName, 'up');
      widget.value = formatLoraValue(newLoras);
      
      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
  );

  // Move Down option with arrow down icon
  const moveDownOption = createMenuItem(
    'Move Down',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      const lorasData = parseLoraValue(widget.value);
      const newLoras = moveLoraByDirection(lorasData, loraName, 'down');
      widget.value = formatLoraValue(newLoras);
      
      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
  );

  // Move to Top option with chevrons up icon
  const moveTopOption = createMenuItem(
    'Move to Top',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 11l-5-5-5 5M17 18l-5-5-5 5"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      const lorasData = parseLoraValue(widget.value);
      const newLoras = moveLoraByDirection(lorasData, loraName, 'top');
      widget.value = formatLoraValue(newLoras);
      
      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
  );

  // Move to Bottom option with chevrons down icon
  const moveBottomOption = createMenuItem(
    'Move to Bottom',
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 13l5 5 5-5M7 6l5 5 5-5"></path></svg>',
    () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
      
      const lorasData = parseLoraValue(widget.value);
      const newLoras = moveLoraByDirection(lorasData, loraName, 'bottom');
      widget.value = formatLoraValue(newLoras);
      
      if (widget.callback) {
        widget.callback(widget.value);
      }
      
      if (renderFunction) {
        renderFunction(widget.value, widget);
      }
    }
  );

  // Add separator
  const separator1 = document.createElement('div');
  Object.assign(separator1.style, {
    margin: '4px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  });
  
  // Add second separator
  const separator2 = document.createElement('div');
  Object.assign(separator2.style, {
    margin: '4px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  });

  // Add separator for order options
  const orderSeparator = document.createElement('div');
  Object.assign(orderSeparator.style, {
    margin: '4px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  });

  menu.appendChild(viewOnCivitaiOption);
  menu.appendChild(deleteOption);
  menu.appendChild(separator1);
  menu.appendChild(moveUpOption);
  menu.appendChild(moveDownOption);
  menu.appendChild(moveTopOption);
  menu.appendChild(moveBottomOption);
  menu.appendChild(orderSeparator);
  menu.appendChild(copyNotesOption);
  menu.appendChild(copyTriggerWordsOption);
  menu.appendChild(separator2);
  menu.appendChild(saveOption);
  
  document.body.appendChild(menu);

  // Close menu when clicking outside
  const closeMenu = (e) => {
    if (!menu.contains(e.target)) {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    }
  };
  setTimeout(() => document.addEventListener('click', closeMenu), 0);
}
