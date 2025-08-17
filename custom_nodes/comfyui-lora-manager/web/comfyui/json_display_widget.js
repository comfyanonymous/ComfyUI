export function addJsonDisplayWidget(node, name, opts) {
  // Create container for JSON display
  const container = document.createElement("div");
  container.className = "comfy-json-display-container";
  
  // Set initial height
  const defaultHeight = 200;
  
  Object.assign(container.style, {
    display: "block",
    padding: "8px",
    backgroundColor: "rgba(40, 44, 52, 0.6)",
    borderRadius: "6px",
    width: "100%",
    boxSizing: "border-box",
    overflow: "auto",
    overflowY: "scroll",
    maxHeight: `${defaultHeight}px`,
    fontFamily: "monospace",
    fontSize: "12px",
    lineHeight: "1.5",
    whiteSpace: "pre-wrap",
    color: "rgba(226, 232, 240, 0.9)"
  });

  // Initialize default value
  const initialValue = opts?.defaultVal || "";
  
  // Function to format and display JSON content with syntax highlighting
  const displayJson = (jsonString, widget) => {
    try {
      // If string is empty, show placeholder
      if (!jsonString || jsonString.trim() === '') {
        container.textContent = "No metadata available";
        container.style.fontStyle = "italic";
        container.style.color = "rgba(226, 232, 240, 0.6)";
        container.style.textAlign = "center";
        container.style.padding = "20px 0";
        return;
      }
      
      // Try to parse and pretty-print if it's valid JSON
      try {
        const jsonObj = JSON.parse(jsonString);
        container.innerHTML = syntaxHighlight(JSON.stringify(jsonObj, null, 2));
      } catch (e) {
        // If not valid JSON, display as-is
        container.textContent = jsonString;
      }
      
      container.style.fontStyle = "normal";
      container.style.textAlign = "left";
      container.style.padding = "8px";
    } catch (error) {
      console.error("Error displaying JSON:", error);
      container.textContent = "Error displaying content";
    }
  };

  // Function to add syntax highlighting to JSON
  function syntaxHighlight(json) {
    // Color scheme
    const colors = {
      key: "#6ad6f5",       // Light blue for keys
      string: "#98c379",    // Soft green for strings
      number: "#e5c07b",    // Amber for numbers
      boolean: "#c678dd",   // Purple for booleans
      null: "#7f848e"       // Gray for null
    };
    
    // Replace JSON syntax with highlighted HTML
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
      let cls = 'number';
      let color = colors.number;
      
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          cls = 'key';
          color = colors.key;
          // Remove the colon from the key and add it back without color
          match = match.replace(/:$/, '');
          return '<span style="color:' + color + ';">' + match + '</span>:';
        } else {
          cls = 'string';
          color = colors.string;
        }
      } else if (/true|false/.test(match)) {
        cls = 'boolean';
        color = colors.boolean;
      } else if (/null/.test(match)) {
        cls = 'null';
        color = colors.null;
      }
      
      return '<span style="color:' + color + ';">' + match + '</span>';
    });
  }
  
  // Store the value
  let widgetValue = initialValue;

  // Create widget with DOM Widget API
  const widget = node.addDOMWidget(name, "custom", container, {
    getValue: function() {
      return widgetValue;
    },
    setValue: function(v) {
      widgetValue = v;
      displayJson(widgetValue, widget);
    },
    hideOnZoom: true
  });

  // Set initial value
  widget.value = initialValue;

  widget.serializeValue = () => {
    return widgetValue;
  };

  // Update widget when node is resized
  const onNodeResize = node.onResize;
  node.onResize = function(size) {
    if(onNodeResize) {
      onNodeResize.call(this, size);
    }
    
    // Adjust container height to node height
    if(size && size[1]) {
      // Reduce the offset to minimize the gap at the bottom
      const widgetHeight = Math.min(size[1] - 30, defaultHeight * 2); // Reduced from 80 to 30
      container.style.maxHeight = `${widgetHeight}px`;
      container.style.setProperty('--comfy-widget-height', `${widgetHeight}px`);
    }
  };

  return { minWidth: 300, minHeight: defaultHeight, widget };
}
