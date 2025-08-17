import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

export function addLorasWidget(node, name, opts, callback) {
  // Create container for loras
  const container = document.createElement("div");
  container.className = "comfy-loras-container";
  Object.assign(container.style, {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    padding: "6px",
    backgroundColor: "rgba(40, 44, 52, 0.6)",
    borderRadius: "6px",
    width: "100%",
  });

  // Initialize default value
  const defaultValue = opts?.defaultVal || [];

  // Parse LoRA entries from value
  const parseLoraValue = (value) => {
    if (!value) return [];
    return Array.isArray(value) ? value : [];
  };

  // Format LoRA data
  const formatLoraValue = (loras) => {
    return loras;
  };

  // Function to create toggle element
  const createToggle = (active, onChange) => {
    const toggle = document.createElement("div");
    toggle.className = "comfy-lora-toggle";
    
    updateToggleStyle(toggle, active);
    
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      onChange(!active);
    });
    
    return toggle;
  };

  // Helper function to update toggle style
  function updateToggleStyle(toggleEl, active) {
    Object.assign(toggleEl.style, {
      width: "18px",
      height: "18px",
      borderRadius: "4px",
      cursor: "pointer",
      transition: "all 0.2s ease",
      backgroundColor: active ? "rgba(66, 153, 225, 0.9)" : "rgba(45, 55, 72, 0.7)",
      border: `1px solid ${active ? "rgba(66, 153, 225, 0.9)" : "rgba(226, 232, 240, 0.2)"}`,
    });

    // Add hover effect
    toggleEl.onmouseenter = () => {
      toggleEl.style.transform = "scale(1.05)";
      toggleEl.style.boxShadow = "0 2px 4px rgba(0,0,0,0.15)";
    };

    toggleEl.onmouseleave = () => {
      toggleEl.style.transform = "scale(1)";
      toggleEl.style.boxShadow = "none";
    };
  }

  // Create arrow button for strength adjustment
  const createArrowButton = (direction, onClick) => {
    const button = document.createElement("div");
    button.className = `comfy-lora-arrow comfy-lora-arrow-${direction}`;
    
    Object.assign(button.style, {
      width: "16px",
      height: "16px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      cursor: "pointer",
      userSelect: "none",
      fontSize: "12px",
      color: "rgba(226, 232, 240, 0.8)",
      transition: "all 0.2s ease",
    });
    
    button.textContent = direction === "left" ? "◀" : "▶";
    
    button.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });
    
    // Add hover effect
    button.onmouseenter = () => {
      button.style.color = "white";
      button.style.transform = "scale(1.2)";
    };
    
    button.onmouseleave = () => {
      button.style.color = "rgba(226, 232, 240, 0.8)";
      button.style.transform = "scale(1)";
    };
    
    return button;
  };

  // 添加预览弹窗组件
  class PreviewTooltip {
    constructor() {
      this.element = document.createElement('div');
      Object.assign(this.element.style, {
        position: 'fixed',
        zIndex: 9999,
        background: 'rgba(0, 0, 0, 0.85)',
        borderRadius: '6px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        display: 'none',
        overflow: 'hidden',
        maxWidth: '300px',
      });
      document.body.appendChild(this.element);
      this.hideTimeout = null;  // 添加超时处理变量
      
      // 添加全局点击事件来隐藏tooltip
      document.addEventListener('click', () => this.hide());
      
      // 添加滚动事件监听
      document.addEventListener('scroll', () => this.hide(), true);
    }

    async show(loraName, x, y) {
      try {
        // 清除之前的隐藏定时器
        if (this.hideTimeout) {
          clearTimeout(this.hideTimeout);
          this.hideTimeout = null;
        }

        // 如果已经显示同一个lora的预览，则不重复显示
        if (this.element.style.display === 'block' && this.currentLora === loraName) {
          return;
        }

        this.currentLora = loraName;
        
        // 获取预览URL
        const response = await api.fetchApi(`/loras/preview-url?name=${encodeURIComponent(loraName)}`, {
          method: 'GET'
        });

        if (!response.ok) {
          throw new Error('Failed to fetch preview URL');
        }

        const data = await response.json();
        if (!data.success || !data.preview_url) {
          throw new Error('No preview available');
        }

        // 清除现有内容
        while (this.element.firstChild) {
          this.element.removeChild(this.element.firstChild);
        }

        // Create media container with relative positioning
        const mediaContainer = document.createElement('div');
        Object.assign(mediaContainer.style, {
          position: 'relative',
          maxWidth: '300px',
          maxHeight: '300px',
        });

        const isVideo = data.preview_url.endsWith('.mp4');
        const mediaElement = isVideo ? document.createElement('video') : document.createElement('img');

        Object.assign(mediaElement.style, {
          maxWidth: '300px',
          maxHeight: '300px',
          objectFit: 'contain',
          display: 'block',
        });

        if (isVideo) {
          mediaElement.autoplay = true;
          mediaElement.loop = true;
          mediaElement.muted = true;
          mediaElement.controls = false;
        }

        mediaElement.src = data.preview_url;

        // Create name label with absolute positioning
        const nameLabel = document.createElement('div');
        nameLabel.textContent = loraName;
        Object.assign(nameLabel.style, {
          position: 'absolute',
          bottom: '0',
          left: '0',
          right: '0',
          padding: '8px',
          color: 'rgba(255, 255, 255, 0.95)',
          fontSize: '13px',
          fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
          background: 'linear-gradient(transparent, rgba(0, 0, 0, 0.8))',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          textAlign: 'center',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
        });

        mediaContainer.appendChild(mediaElement);
        mediaContainer.appendChild(nameLabel);
        this.element.appendChild(mediaContainer);
        
        // 添加淡入效果
        this.element.style.opacity = '0';
        this.element.style.display = 'block';
        this.position(x, y);
        
        requestAnimationFrame(() => {
          this.element.style.transition = 'opacity 0.15s ease';
          this.element.style.opacity = '1';
        });
      } catch (error) {
        console.warn('Failed to load preview:', error);
      }
    }

    position(x, y) {
      // 确保预览框不超出视窗边界
      const rect = this.element.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let left = x + 10; // 默认在鼠标右侧偏移10px
      let top = y + 10;  // 默认在鼠标下方偏移10px

      // 检查右边界
      if (left + rect.width > viewportWidth) {
        left = x - rect.width - 10;
      }

      // 检查下边界
      if (top + rect.height > viewportHeight) {
        top = y - rect.height - 10;
      }

      Object.assign(this.element.style, {
        left: `${left}px`,
        top: `${top}px`
      });
    }

    hide() {
      // 使用淡出效果
      if (this.element.style.display === 'block') {
        this.element.style.opacity = '0';
        this.hideTimeout = setTimeout(() => {
          this.element.style.display = 'none';
          this.currentLora = null;
          // 停止视频播放
          const video = this.element.querySelector('video');
          if (video) {
            video.pause();
          }
          this.hideTimeout = null;
        }, 150);
      }
    }

    cleanup() {
      if (this.hideTimeout) {
        clearTimeout(this.hideTimeout);
      }
      // 移除所有事件监听器
      document.removeEventListener('click', () => this.hide());
      document.removeEventListener('scroll', () => this.hide(), true);
      this.element.remove();
    }
  }

  // 创建预览tooltip实例
  const previewTooltip = new PreviewTooltip();

  // Function to handle strength adjustment via dragging
  const handleStrengthDrag = (name, initialStrength, initialX, event, widget) => {
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
      lorasData[loraIndex].strength = newStrength;
      
      // Update the widget value
      widget.value = formatLoraValue(lorasData);
      
      // Force re-render to show updated strength value
      renderLoras(widget.value, widget);
    }
  };
  
  // Function to initialize drag operation
  const initDrag = (loraEl, nameEl, name, widget) => {
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
    
    // Create a drag handler that's applied to the entire lora entry
    // except toggle and strength controls
    loraEl.addEventListener('mousedown', (e) => {
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
      initialStrength = loraData.strength;
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
      handleStrengthDrag(name, initialStrength, initialX, e, widget);
      
      // Prevent showing the preview tooltip during drag
      previewTooltip.hide();
    });
    
    document.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        // Remove the class to restore normal cursor behavior
        document.body.classList.remove('comfy-lora-dragging');
      }
    });
  };

  // Function to create menu item
  const createMenuItem = (text, icon, onClick) => {
    const menuItem = document.createElement('div');
    Object.assign(menuItem.style, {
      padding: '6px 20px',
      cursor: 'pointer',
      color: 'rgba(226, 232, 240, 0.9)',
      fontSize: '13px',
      userSelect: 'none',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    });

    // Create icon element
    const iconEl = document.createElement('div');
    iconEl.innerHTML = icon;
    Object.assign(iconEl.style, {
      width: '14px',
      height: '14px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    });

    // Create text element
    const textEl = document.createElement('span');
    textEl.textContent = text;

    menuItem.appendChild(iconEl);
    menuItem.appendChild(textEl);

    menuItem.addEventListener('mouseenter', () => {
      menuItem.style.backgroundColor = 'rgba(66, 153, 225, 0.2)';
    });

    menuItem.addEventListener('mouseleave', () => {
      menuItem.style.backgroundColor = 'transparent';
    });

    if (onClick) {
      menuItem.addEventListener('click', onClick);
    }

    return menuItem;
  };

  // Function to create context menu
  const createContextMenu = (x, y, loraName, widget) => {
    // Hide preview tooltip first
    previewTooltip.hide();

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
            if (app && app.extensionManager && app.extensionManager.toast) {
              app.extensionManager.toast.add({
                severity: 'warning',
                summary: 'Not Found',
                detail: 'This LoRA has no associated Civitai URL',
                life: 3000
              });
            } else {
              alert('This LoRA has no associated Civitai URL');
            }
          }
        } catch (error) {
          console.error('Error getting Civitai URL:', error);
          if (app && app.extensionManager && app.extensionManager.toast) {
            app.extensionManager.toast.add({
              severity: 'error',
              summary: 'Error',
              detail: error.message || 'Failed to get Civitai URL',
              life: 5000
            });
          } else {
            alert('Error: ' + (error.message || 'Failed to get Civitai URL'));
          }
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
      }
    );

    // Save recipe option with bookmark icon
    const saveOption = createMenuItem(
      'Save Recipe',
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path></svg>',
      () => {
        menu.remove();
        document.removeEventListener('click', closeMenu);
        saveRecipeDirectly(widget);
      }
    );

    // Add separator
    const separator = document.createElement('div');
    Object.assign(separator.style, {
      margin: '4px 0',
      borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    });

    menu.appendChild(viewOnCivitaiOption); // Add the new menu option
    menu.appendChild(deleteOption);
    menu.appendChild(separator);
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
  };

  // Function to render loras from data
  const renderLoras = (value, widget) => {
    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Parse the loras data
    const lorasData = parseLoraValue(value);

    if (lorasData.length === 0) {
      // Show message when no loras are added
      const emptyMessage = document.createElement("div");
      emptyMessage.textContent = "No LoRAs added";
      Object.assign(emptyMessage.style, {
        textAlign: "center",
        padding: "20px 0",
        color: "rgba(226, 232, 240, 0.8)",
        fontStyle: "italic",
        userSelect: "none",     // Add this line to prevent text selection
        WebkitUserSelect: "none",  // For Safari support
        MozUserSelect: "none",     // For Firefox support
        msUserSelect: "none",      // For IE/Edge support
      });
      container.appendChild(emptyMessage);
      return;
    }

    // Create header
    const header = document.createElement("div");
    header.className = "comfy-loras-header";
    Object.assign(header.style, {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "4px 8px",
      borderBottom: "1px solid rgba(226, 232, 240, 0.2)",
      marginBottom: "8px"
    });

    // Add toggle all control
    const allActive = lorasData.every(lora => lora.active);
    const toggleAll = createToggle(allActive, (active) => {
      // Update all loras active state
      const lorasData = parseLoraValue(widget.value);
      lorasData.forEach(lora => lora.active = active);
      
      const newValue = formatLoraValue(lorasData);
      widget.value = newValue;
    });

    // Add label to toggle all
    const toggleLabel = document.createElement("div");
    toggleLabel.textContent = "Toggle All";
    Object.assign(toggleLabel.style, {
      color: "rgba(226, 232, 240, 0.8)",
      fontSize: "13px",
      marginLeft: "8px",
      userSelect: "none",     // Add this line to prevent text selection
      WebkitUserSelect: "none",  // For Safari support
      MozUserSelect: "none",     // For Firefox support
      msUserSelect: "none",      // For IE/Edge support
    });

    const toggleContainer = document.createElement("div");
    Object.assign(toggleContainer.style, {
      display: "flex",
      alignItems: "center",
    });
    toggleContainer.appendChild(toggleAll);
    toggleContainer.appendChild(toggleLabel);

    // Strength label
    const strengthLabel = document.createElement("div");
    strengthLabel.textContent = "Strength";
    Object.assign(strengthLabel.style, {
      color: "rgba(226, 232, 240, 0.8)",
      fontSize: "13px",
      marginRight: "8px",
      userSelect: "none",     // Add this line to prevent text selection
      WebkitUserSelect: "none",  // For Safari support
      MozUserSelect: "none",     // For Firefox support
      msUserSelect: "none",      // For IE/Edge support
    });

    header.appendChild(toggleContainer);
    header.appendChild(strengthLabel);
    container.appendChild(header);

    // Render each lora entry
    lorasData.forEach((loraData) => {
      const { name, strength, active } = loraData;
      
      const loraEl = document.createElement("div");
      loraEl.className = "comfy-lora-entry";
      Object.assign(loraEl.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "8px",
        borderRadius: "6px",
        backgroundColor: active ? "rgba(45, 55, 72, 0.7)" : "rgba(35, 40, 50, 0.5)",
        transition: "all 0.2s ease",
        marginBottom: "6px",
      });

      // Create toggle for this lora
      const toggle = createToggle(active, (newActive) => {
        // Update this lora's active state
        const lorasData = parseLoraValue(widget.value);
        const loraIndex = lorasData.findIndex(l => l.name === name);
        
        if (loraIndex >= 0) {
          lorasData[loraIndex].active = newActive;
          
          const newValue = formatLoraValue(lorasData);
          widget.value = newValue;
        }
      });

      // Create name display
      const nameEl = document.createElement("div");
      nameEl.textContent = name;
      Object.assign(nameEl.style, {
        marginLeft: "10px",
        flex: "1",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        color: active ? "rgba(226, 232, 240, 0.9)" : "rgba(226, 232, 240, 0.6)",
        fontSize: "13px",
        cursor: "pointer", // Add pointer cursor to indicate hoverable area
        userSelect: "none",     // Add this line to prevent text selection
        WebkitUserSelect: "none",  // For Safari support
        MozUserSelect: "none",     // For Firefox support
        msUserSelect: "none",      // For IE/Edge support
      });

      // Move preview tooltip events to nameEl instead of loraEl
      nameEl.addEventListener('mouseenter', async (e) => {
        e.stopPropagation();
        const rect = nameEl.getBoundingClientRect();
        await previewTooltip.show(name, rect.right, rect.top);
      });

      nameEl.addEventListener('mouseleave', (e) => {
        e.stopPropagation();
        previewTooltip.hide();
      });

      // Remove the preview tooltip events from loraEl
      loraEl.onmouseenter = () => {
        loraEl.style.backgroundColor = active ? "rgba(50, 60, 80, 0.8)" : "rgba(40, 45, 55, 0.6)";
      };
      
      loraEl.onmouseleave = () => {
        loraEl.style.backgroundColor = active ? "rgba(45, 55, 72, 0.7)" : "rgba(35, 40, 50, 0.5)";
      };

      // Add context menu event
      loraEl.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        createContextMenu(e.clientX, e.clientY, name, widget);
      });

      // Create strength control
      const strengthControl = document.createElement("div");
      Object.assign(strengthControl.style, {
        display: "flex",
        alignItems: "center",
        gap: "8px",
      });

      // Left arrow
      const leftArrow = createArrowButton("left", () => {
        // Decrease strength
        const lorasData = parseLoraValue(widget.value);
        const loraIndex = lorasData.findIndex(l => l.name === name);
        
        if (loraIndex >= 0) {
          lorasData[loraIndex].strength = (lorasData[loraIndex].strength - 0.05).toFixed(2);
          
          const newValue = formatLoraValue(lorasData);
          widget.value = newValue;
        }
      });

      // Strength display
      const strengthEl = document.createElement("input");
      strengthEl.type = "text";
      strengthEl.value = typeof strength === 'number' ? strength.toFixed(2) : Number(strength).toFixed(2);
      Object.assign(strengthEl.style, {
        minWidth: "50px",
        width: "50px",
        textAlign: "center",
        color: active ? "rgba(226, 232, 240, 0.9)" : "rgba(226, 232, 240, 0.6)",
        fontSize: "13px",
        background: "none",
        border: "1px solid transparent",
        padding: "2px 4px",
        borderRadius: "3px",
        outline: "none",
      });

      // 添加hover效果
      strengthEl.addEventListener('mouseenter', () => {
        strengthEl.style.border = "1px solid rgba(226, 232, 240, 0.2)";
      });

      strengthEl.addEventListener('mouseleave', () => {
        if (document.activeElement !== strengthEl) {
          strengthEl.style.border = "1px solid transparent";
        }
      });

      // 处理焦点
      strengthEl.addEventListener('focus', () => {
        strengthEl.style.border = "1px solid rgba(66, 153, 225, 0.6)";
        strengthEl.style.background = "rgba(0, 0, 0, 0.2)";
        // 自动选中所有内容
        strengthEl.select();
      });

      strengthEl.addEventListener('blur', () => {
        strengthEl.style.border = "1px solid transparent";
        strengthEl.style.background = "none";
      });

      // 处理输入变化
      strengthEl.addEventListener('change', () => {
        let newValue = parseFloat(strengthEl.value);
        
        // 验证输入
        if (isNaN(newValue)) {
          newValue = 1.0;
        }
        
        // 更新数值
        const lorasData = parseLoraValue(widget.value);
        const loraIndex = lorasData.findIndex(l => l.name === name);
        
        if (loraIndex >= 0) {
          lorasData[loraIndex].strength = newValue.toFixed(2);
          
          // 更新值并触发回调
          const newLorasValue = formatLoraValue(lorasData);
          widget.value = newLorasValue;
        }
      });

      // 处理按键事件
      strengthEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          strengthEl.blur();
        }
      });

      // Right arrow
      const rightArrow = createArrowButton("right", () => {
        // Increase strength
        const lorasData = parseLoraValue(widget.value);
        const loraIndex = lorasData.findIndex(l => l.name === name);
        
        if (loraIndex >= 0) {
          lorasData[loraIndex].strength = (parseFloat(lorasData[loraIndex].strength) + 0.05).toFixed(2);
          
          const newValue = formatLoraValue(lorasData);
          widget.value = newValue;
        }
      });

      strengthControl.appendChild(leftArrow);
      strengthControl.appendChild(strengthEl);
      strengthControl.appendChild(rightArrow);

      // Assemble entry
      const leftSection = document.createElement("div");
      Object.assign(leftSection.style, {
        display: "flex",
        alignItems: "center",
        flex: "1",
        minWidth: "0", // Allow shrinking
      });
      
      leftSection.appendChild(toggle);
      leftSection.appendChild(nameEl);
      
      loraEl.appendChild(leftSection);
      loraEl.appendChild(strengthControl);

      container.appendChild(loraEl);

      // Initialize drag functionality
      initDrag(loraEl, nameEl, name, widget);
    });
  };

  // Store the value in a variable to avoid recursion
  let widgetValue = defaultValue;

  // Create widget with initial properties
  const widget = node.addDOMWidget(name, "loras", container, {
    getValue: function() {
      return widgetValue;
    },
    setValue: function(v) {
      // Remove duplicates by keeping the last occurrence of each lora name
      const uniqueValue = (v || []).reduce((acc, lora) => {
        // Remove any existing lora with the same name
        const filtered = acc.filter(l => l.name !== lora.name);
        // Add the current lora
        return [...filtered, lora];
      }, []);

      widgetValue = uniqueValue;
      renderLoras(widgetValue, widget);
      
      // Update container height after rendering
      requestAnimationFrame(() => {
        const minHeight = this.getMinHeight();
        container.style.height = `${minHeight}px`;
        
        // Force node to update size
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      });
    },
    getMinHeight: function() {
      // Calculate height based on content
      const lorasCount = parseLoraValue(widgetValue).length;
      return Math.max(
        100,
        lorasCount > 0 ? 60 + lorasCount * 44 : 60
      );
    },
  });

  widget.value = defaultValue;

  widget.callback = callback;

  widget.serializeValue = () => {
    // Add dummy items to avoid the 2-element serialization issue, a bug in comfyui
    return [...widgetValue, 
        { name: "__dummy_item1__", strength: 0, active: false, _isDummy: true },
        { name: "__dummy_item2__", strength: 0, active: false, _isDummy: true }
      ];
  }

  widget.onRemove = () => {
    container.remove(); 
    previewTooltip.cleanup();
  };

  return { minWidth: 400, minHeight: 200, widget };
}

// Function to directly save the recipe without dialog
async function saveRecipeDirectly(widget) {
  try {
    // Show loading toast
    if (app && app.extensionManager && app.extensionManager.toast) {
      app.extensionManager.toast.add({
        severity: 'info',
        summary: 'Saving Recipe',
        detail: 'Please wait...',
        life: 2000
      });
    }
    
    // Send the request
    const response = await fetch('/api/recipes/save-from-widget', {
      method: 'POST'
    });
    
    const result = await response.json();
    
    // Show result toast
    if (app && app.extensionManager && app.extensionManager.toast) {
      if (result.success) {
        app.extensionManager.toast.add({
          severity: 'success',
          summary: 'Recipe Saved',
          detail: 'Recipe has been saved successfully',
          life: 3000
        });
      } else {
        app.extensionManager.toast.add({
          severity: 'error',
          summary: 'Error',
          detail: result.error || 'Failed to save recipe',
          life: 5000
        });
      }
    }
  } catch (error) {
    console.error('Error saving recipe:', error);
    
    // Show error toast
    if (app && app.extensionManager && app.extensionManager.toast) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: 'Error',
        detail: 'Failed to save recipe: ' + (error.message || 'Unknown error'),
        life: 5000
      });
    }
  }
}
