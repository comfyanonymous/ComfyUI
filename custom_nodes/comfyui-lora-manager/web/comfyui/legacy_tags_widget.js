export function addTagsWidget(node, name, opts, callback) {
  // Create container for tags
  const container = document.createElement("div");
  container.className = "comfy-tags-container";
  Object.assign(container.style, {
    display: "flex",
    flexWrap: "wrap",
    gap: "4px",    // 从8px减小到4px
    padding: "6px",
    minHeight: "30px",
    backgroundColor: "rgba(40, 44, 52, 0.6)",  // Darker, more modern background
    borderRadius: "6px",    // Slightly larger radius
    width: "100%",
  });

  // Initialize default value as array
  const initialTagsData = opts?.defaultVal || [];

  // Function to render tags from array data
  const renderTags = (tagsData, widget) => {
    // Clear existing tags
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const normalizedTags = tagsData;

    if (normalizedTags.length === 0) {
      // Show message when no tags are present
      const emptyMessage = document.createElement("div");
      emptyMessage.textContent = "No trigger words detected";
      Object.assign(emptyMessage.style, {
        textAlign: "center",
        padding: "20px 0",
        color: "rgba(226, 232, 240, 0.8)",
        fontStyle: "italic",
        userSelect: "none",
        WebkitUserSelect: "none",
        MozUserSelect: "none",
        msUserSelect: "none",
      });
      container.appendChild(emptyMessage);
      return;
    }

    normalizedTags.forEach((tagData, index) => {
      const { text, active } = tagData;
      const tagEl = document.createElement("div");
      tagEl.className = "comfy-tag";
      
      updateTagStyle(tagEl, active);

      tagEl.textContent = text;
      tagEl.title = text; // Set tooltip for full content

      // Add click handler to toggle state
      tagEl.addEventListener("click", (e) => {
        e.stopPropagation();

        // Toggle active state for this specific tag using its index
        const updatedTags = [...widget.value];
        updatedTags[index].active = !updatedTags[index].active;
        updateTagStyle(tagEl, updatedTags[index].active);

        widget.value = updatedTags;
      });

      container.appendChild(tagEl);
    });
  };

  // Helper function to update tag style based on active state
  function updateTagStyle(tagEl, active) {
    const baseStyles = {
      padding: "4px 12px",    // 垂直内边距从6px减小到4px
      borderRadius: "6px",    // Matching container radius
      maxWidth: "200px",      // Increased max width
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
      fontSize: "13px",       // Slightly larger font
      cursor: "pointer",
      transition: "all 0.2s ease",  // Smoother transition
      border: "1px solid transparent",
      display: "inline-block",
      boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
      margin: "2px",          // 从4px减小到2px
      userSelect: "none",     // Add this line to prevent text selection
      WebkitUserSelect: "none",  // For Safari support
      MozUserSelect: "none",     // For Firefox support
      msUserSelect: "none",      // For IE/Edge support
    };

    if (active) {
      Object.assign(tagEl.style, {
        ...baseStyles,
        backgroundColor: "rgba(66, 153, 225, 0.9)",  // Modern blue
        color: "white",
        borderColor: "rgba(66, 153, 225, 0.9)",
      });
    } else {
      Object.assign(tagEl.style, {
        ...baseStyles,
        backgroundColor: "rgba(45, 55, 72, 0.7)",    // Darker inactive state
        color: "rgba(226, 232, 240, 0.8)",          // Lighter text for contrast
        borderColor: "rgba(226, 232, 240, 0.2)",
      });
    }

    // Add hover effect
    tagEl.onmouseenter = () => {
      tagEl.style.transform = "translateY(-1px)";
      tagEl.style.boxShadow = "0 2px 4px rgba(0,0,0,0.15)";
    };

    tagEl.onmouseleave = () => {
      tagEl.style.transform = "translateY(0)";
      tagEl.style.boxShadow = "0 1px 2px rgba(0,0,0,0.1)";
    };
  }

  // Store the value as array
  let widgetValue = initialTagsData;

  // Create widget with initial properties
  const widget = node.addDOMWidget(name, "tags", container, {
    getValue: function() {
      return widgetValue;
    },
    setValue: function(v) {
      widgetValue = v;
      renderTags(widgetValue, widget);

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
      const minHeight = 150;
      // If no tags or only showing the empty message, return a minimum height
      if (widgetValue.length === 0) {
        return minHeight; // Height for empty state with message
      }
      
      // Get all tag elements
      const tagElements = container.querySelectorAll('.comfy-tag');
      
      if (tagElements.length === 0) {
        return minHeight; // Fallback if elements aren't rendered yet
      }
      
      // Calculate the actual height based on tag positions
      let maxBottom = 0;
      
      tagElements.forEach(tag => {
        const rect = tag.getBoundingClientRect();
        const tagBottom = rect.bottom - container.getBoundingClientRect().top;
        maxBottom = Math.max(maxBottom, tagBottom);
      });
      
      // Add padding (top and bottom padding of container)
      const computedStyle = window.getComputedStyle(container);
      const paddingTop = parseInt(computedStyle.paddingTop, 10) || 0;
      const paddingBottom = parseInt(computedStyle.paddingBottom, 10) || 0;
      
      // Add extra buffer for potential wrapping issues and to ensure no clipping
      const extraBuffer = 20;
      
      // Round up to nearest 5px for clean sizing and ensure minimum height
      return Math.max(minHeight, Math.ceil((maxBottom + paddingBottom + extraBuffer) / 5) * 5);
    },
  });

  widget.value = initialTagsData;

  widget.callback = callback;

  widget.serializeValue = () => {
    // Add dummy items to avoid the 2-element serialization issue, a bug in comfyui
    return [...widgetValue, 
        { text: "__dummy_item__", active: false, _isDummy: true },
        { text: "__dummy_item__", active: false, _isDummy: true }
      ];
  };

  return { minWidth: 300, minHeight: 150, widget };
}
