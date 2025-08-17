export function addTagsWidget(node, name, opts, callback) {
  // Create container for tags
  const container = document.createElement("div");
  container.className = "comfy-tags-container";
  
  // Set initial height
  const defaultHeight = 150;
  
  Object.assign(container.style, {
    display: "flex",
    flexWrap: "wrap",
    gap: "4px",
    padding: "6px",
    backgroundColor: "rgba(40, 44, 52, 0.6)",
    borderRadius: "6px",
    width: "100%",
    boxSizing: "border-box",
    overflow: "auto",
    alignItems: "flex-start" // Ensure tags align at the top of each row
  });

  // Initialize default value as array
  const initialTagsData = opts?.defaultVal || [];

  // Fixed sizes for tag elements to avoid zoom-related calculation issues
  const TAG_HEIGHT = 26; // Adjusted height of a single tag including margins
  const TAGS_PER_ROW = 3; // Approximate number of tags per row
  const ROW_GAP = 2; // Reduced gap between rows
  const CONTAINER_PADDING = 12; // Top and bottom padding
  const EMPTY_CONTAINER_HEIGHT = 60; // Height when no tags are present

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
        width: "100%"
      });
      container.appendChild(emptyMessage);
      
      // Set fixed height for empty state
      updateWidgetHeight(EMPTY_CONTAINER_HEIGHT);
      return;
    }

    // Create a row container approach for better layout control
    let rowContainer = document.createElement("div");
    rowContainer.className = "comfy-tags-row";
    Object.assign(rowContainer.style, {
      display: "flex",
      flexWrap: "wrap",
      gap: "4px",
      width: "100%",
      marginBottom: "2px" // Small gap between rows
    });
    container.appendChild(rowContainer);

    let tagCount = 0;
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

      rowContainer.appendChild(tagEl);
      tagCount++;
    });
    
    // Calculate height based on number of tags and fixed sizes
    const tagsCount = normalizedTags.length;
    const rows = Math.ceil(tagsCount / TAGS_PER_ROW);
    const calculatedHeight = CONTAINER_PADDING + (rows * TAG_HEIGHT) + ((rows - 1) * ROW_GAP);
    
    // Update widget height with calculated value
    updateWidgetHeight(calculatedHeight);
  };

  // Function to update widget height consistently
  const updateWidgetHeight = (height) => {
    // Ensure minimum height
    const finalHeight = Math.max(defaultHeight, height);
    
    // Update CSS variables
    container.style.setProperty('--comfy-widget-min-height', `${finalHeight}px`);
    container.style.setProperty('--comfy-widget-height', `${finalHeight}px`);
    
    // Force node to update size after a short delay to ensure DOM is updated
    if (node) {
      setTimeout(() => {
        node.setDirtyCanvas(true, true);
      }, 10);
    }
  };

  // Helper function to update tag style based on active state
  function updateTagStyle(tagEl, active) {
    const baseStyles = {
      padding: "3px 10px", // Adjusted vertical padding to balance text
      borderRadius: "6px",
      maxWidth: "200px",
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
      fontSize: "13px",
      cursor: "pointer",
      transition: "all 0.2s ease",
      border: "1px solid transparent",
      display: "inline-block", // inline-block for better text truncation
      boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
      margin: "1px", 
      userSelect: "none",
      WebkitUserSelect: "none",
      MozUserSelect: "none",
      msUserSelect: "none",
      height: "22px", // Increased height to better fit text with descenders
      minHeight: "22px", // Matching minHeight
      boxSizing: "border-box",
      width: "fit-content",
      maxWidth: "200px",
      lineHeight: "16px", // Added explicit line-height
      verticalAlign: "middle", // Added vertical alignment
      position: "relative", // For better text positioning
      textAlign: "center", // Center text horizontally
    };

    if (active) {
      Object.assign(tagEl.style, {
        ...baseStyles,
        backgroundColor: "rgba(66, 153, 225, 0.9)",
        color: "white",
        borderColor: "rgba(66, 153, 225, 0.9)",
      });
    } else {
      Object.assign(tagEl.style, {
        ...baseStyles,
        backgroundColor: "rgba(45, 55, 72, 0.7)",
        color: "rgba(226, 232, 240, 0.8)",
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

  // Create widget with new DOM Widget API
  const widget = node.addDOMWidget(name, "custom", container, {
    getValue: function() {
      return widgetValue;
    },
    setValue: function(v) {
      widgetValue = v;
      renderTags(widgetValue, widget);
    },
    hideOnZoom: true,
    selectOn: ['click', 'focus']
  });

  // Set initial value
  widget.value = initialTagsData;

  // Set callback
  widget.callback = callback;

  widget.serializeValue = () => {
    return widgetValue
  };

  return { minWidth: 300, minHeight: defaultHeight, widget };
}