import { app } from '../../../scripts/app.js'

//from melmass
export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

export const loadScript = (
    FILE_URL,
    async = true,
    type = 'text/javascript',
  ) => {
    return new Promise((resolve, reject) => {
      try {
        // Check if the script already exists
        const existingScript = document.querySelector(`script[src="${FILE_URL}"]`)
        if (existingScript) {
          resolve({ status: true, message: 'Script already loaded' })
          return
        }
  
        const scriptEle = document.createElement('script')
        scriptEle.type = type
        scriptEle.async = async
        scriptEle.src = FILE_URL
  
        scriptEle.addEventListener('load', (ev) => {
          resolve({ status: true })
        })
  
        scriptEle.addEventListener('error', (ev) => {
          reject({
            status: false,
            message: `Failed to load the script ${FILE_URL}`,
          })
        })
  
        document.body.appendChild(scriptEle)
      } catch (error) {
        reject(error)
      }
    })
  }
  const create_documentation_stylesheet = () => {
    const tag = 'kj-splineditor-stylesheet'

    let styleTag = document.head.querySelector(tag)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = tag
      styleTag.innerHTML = `
       .spline-editor {

        position: absolute;

        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        z-index: 0;
        overflow: hidden;
       }
        `
      document.head.appendChild(styleTag)
    }
  }

loadScript('kjweb_async/svg-path-properties.min.js').catch((e) => {
    console.log(e)
})
loadScript('kjweb_async/protovis.min.js').catch((e) => {
  console.log(e)
})
create_documentation_stylesheet()

function chainCallback(object, property, callback) {
  if (object == undefined) {
      //This should not happen.
      console.error("Tried to add callback to non-existant object")
      return;
  }
  if (property in object) {
      const callback_orig = object[property]
      object[property] = function () {
          const r = callback_orig.apply(this, arguments);
          callback.apply(this, arguments);
          return r
      };
  } else {
      object[property] = callback;
  }
}
app.registerExtension({
    name: 'KJNodes.SplineEditor', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === 'SplineEditor') {
          chainCallback(nodeType.prototype, "onNodeCreated", function () {
            
            this.widgets.find(w => w.name === "coordinates").hidden = true

            var element = document.createElement("div");
            this.uuid = makeUUID()
            element.id = `spline-editor-${this.uuid}`

            this.previewMediaType = 'image'

            this.splineEditor = this.addDOMWidget(nodeData.name, "SplineEditorWidget", element, {
            serialize: false,
            hideOnZoom: false,
            });

            // context menu
            this.contextMenu = document.createElement("div");
            this.contextMenu.className = 'spline-editor-context-menu';
            this.contextMenu.id = "context-menu";
            this.contextMenu.style.display = "none";
            this.contextMenu.style.position = "absolute";
            this.contextMenu.style.backgroundColor = "#202020";
            this.contextMenu.style.minWidth = "100px";
            this.contextMenu.style.boxShadow = "0px 8px 16px 0px rgba(0,0,0,0.2)";
            this.contextMenu.style.zIndex = "100";
            this.contextMenu.style.padding = "5px";

            function styleMenuItem(menuItem) {
              menuItem.style.display = "block";
              menuItem.style.padding = "5px";
              menuItem.style.color = "#FFF";
              menuItem.style.fontFamily = "Arial, sans-serif";
              menuItem.style.fontSize = "16px";
              menuItem.style.textDecoration = "none";
              menuItem.style.marginBottom = "5px";
            }
            function createMenuItem(id, textContent) {
              let menuItem = document.createElement("a");
              menuItem.href = "#";
              menuItem.id = `menu-item-${id}`;
              menuItem.textContent = textContent;
              styleMenuItem(menuItem);
              return menuItem;
            }
            
            // Create an array of menu items using the createMenuItem function
            this.menuItems = [
              createMenuItem(0, "Toggle handles"),
              createMenuItem(1, "Display sample points"),
              createMenuItem(2, "Switch point shape"),
              createMenuItem(3, "Background image"),
              createMenuItem(4, "Invert point order"),
              createMenuItem(5, "Clear Image"),
              createMenuItem(6, "Add new spline"),
              createMenuItem(7, "Add new single point"),
              createMenuItem(8, "Delete current spline"),
              createMenuItem(9, "Next spline"),
            ];
            
            // Add mouseover and mouseout event listeners to each menu item for styling
            this.menuItems.forEach(menuItem => {
              menuItem.addEventListener('mouseover', function() {
                this.style.backgroundColor = "gray";
              });
            
              menuItem.addEventListener('mouseout', function() {
                this.style.backgroundColor = "#202020";
              });
            });
            
            // Append each menu item to the context menu
            this.menuItems.forEach(menuItem => {
              this.contextMenu.appendChild(menuItem);
            });

            document.body.appendChild(this.contextMenu);

            this.addWidget("button", "New canvas", null, () => {
              if (!this.properties || !("points" in this.properties)) {
              this.editor = new SplineEditor(this);
                this.addProperty("points", this.constructor.type, "string");
              }
              else {
              this.editor = new SplineEditor(this, true);
              }
            });
            
            this.setSize([550, 1000]);
            this.resizable = false;
            this.splineEditor.parentEl = document.createElement("div");
            this.splineEditor.parentEl.className = "spline-editor";
            this.splineEditor.parentEl.id = `spline-editor-${this.uuid}`
            element.appendChild(this.splineEditor.parentEl);
            
          chainCallback(this, "onConfigure", function () {
            try {
              this.editor = new SplineEditor(this);
            } catch (error) {
              console.error("An error occurred while configuring the editor:", error);
            }
          });
          chainCallback(this, "onExecuted", function (message) {
            let bg_image = message["bg_image"];
            this.properties.imgData = {
              name: "bg_image",
              base64: bg_image
            };
            this.editor.refreshBackgroundImage(this);
                });
              
          }); // onAfterGraphConfigured
        }//node created
      } //before register
})//register


class SplineEditor{
  constructor(context, reset = false) {
  this.node = context;
  this.reset=reset;
  const self = this;
  console.log("creatingSplineEditor")

  this.node.pasteFile = (file) => {
    if (file.type.startsWith("image/")) {
      this.handleImageFile(file);
      return true;
    }
    return false;
  };

  this.node.onDragOver = function (e) {
    if (e.dataTransfer && e.dataTransfer.items) {
      return [...e.dataTransfer.items].some(f => f.kind === "file" && f.type.startsWith("image/"));
    }
    return false;
  };

  // On drop upload files
  this.node.onDragDrop = (e) => {
    console.log("onDragDrop called");
    let handled = false;
    for (const file of e.dataTransfer.files) {
      if (file.type.startsWith("image/")) {
        this.handleImageFile(file);
        handled = true;
      }
    }
    return handled;
  };

  // context menu
  this.createContextMenu();
  

  this.dotShape = "circle";
  this.drawSamplePoints = false;
  
  if (reset && context.splineEditor.element) {
    context.splineEditor.element.innerHTML = ''; // Clear the container
  }
  this.coordWidget = context.widgets.find(w => w.name === "coordinates");
  this.interpolationWidget = context.widgets.find(w => w.name === "interpolation");
  this.pointsWidget = context.widgets.find(w => w.name === "points_to_sample");
  this.pointsStoreWidget = context.widgets.find(w => w.name === "points_store");
  this.tensionWidget = context.widgets.find(w => w.name === "tension");
  this.minValueWidget = context.widgets.find(w => w.name === "min_value");
  this.maxValueWidget = context.widgets.find(w => w.name === "max_value");
  this.samplingMethodWidget = context.widgets.find(w => w.name === "sampling_method");
  this.widthWidget = context.widgets.find(w => w.name === "mask_width");
  this.heightWidget = context.widgets.find(w => w.name === "mask_height");

  this.interpolation = this.interpolationWidget.value
  this.tension = this.tensionWidget.value
  this.points_to_sample = this.pointsWidget.value
  this.rangeMin = this.minValueWidget.value
  this.rangeMax = this.maxValueWidget.value
  this.pointsLayer = null;
  this.samplingMethod = this.samplingMethodWidget.value
  
  if (this.samplingMethod == "path"||this.samplingMethod == "speed") {
    this.dotShape = "triangle"
  }
  
  
  this.interpolationWidget.callback = () => {
    this.interpolation = this.interpolationWidget.value
    this.updatePath();
  }
  this.samplingMethodWidget.callback = () => {
    this.samplingMethod = this.samplingMethodWidget.value
    if (this.samplingMethod == "path") {
      this.dotShape = "triangle"
    }
    else if (this.samplingMethod == "controlpoints") {
      this.dotShape = "circle"
      this.drawSamplePoints = true;
    }
    this.updatePath();
  }
  this.tensionWidget.callback = () => {
    this.tension = this.tensionWidget.value
    this.updatePath();
  }
  this.pointsWidget.callback = () => {
    this.points_to_sample = this.pointsWidget.value
    this.updatePath();
  }
  this.minValueWidget.callback = () => {
    this.rangeMin = this.minValueWidget.value
    this.updatePath();
  }
  this.maxValueWidget.callback = () => {
    this.rangeMax = this.maxValueWidget.value
    this.updatePath();
  }
  this.widthWidget.callback = () => {
    this.width = this.widthWidget.value;
    if (this.width > 256) {
        context.setSize([this.width + 45, context.size[1]]);
    }
    this.vis.width(this.width);
    this.updatePath();
}
this.heightWidget.callback = () => {
    this.height = this.heightWidget.value
    this.vis.height(this.height)
    context.setSize([context.size[0], this.height + 450]);
    this.updatePath();
  }
  this.pointsStoreWidget.callback = () => {
    points = JSON.parse(this.pointsStoreWidget.value);
    this.updatePath();
  }
  
 // Initialize or reset points array
 this.drawHandles = false;
 this.drawRuler = true;
 var hoverIndex = -1;
 var isDragging = false;
 this.width = this.widthWidget.value;
 this.height = this.heightWidget.value;
 var i = 3;
 this.splines = [];
 this.activeSplineIndex = 0; // Track which spline is being edited
 // init mouse position
this.lastMousePosition = { x: this.width/2, y: this.height/2 };

 if (!reset && this.pointsStoreWidget.value != "") {
  try {
    const parsedData = JSON.parse(this.pointsStoreWidget.value);
    // Check if it's already in the new format (array of splines)
    if (Array.isArray(parsedData) && parsedData.length > 0 && parsedData[0].hasOwnProperty('points')) {
      this.splines = parsedData;
    } else {
      // Convert old format (single array of points) to new format
      this.splines = [{ 
        points: parsedData, 
        color: "#1f77b4", 
        name: "Spline 1" 
      }];
    }
  } catch (e) {
    console.error("Error parsing spline data:", e);
    this.initializeDefaultSplines();
  }
} else {
  this.initializeDefaultSplines();
  this.pointsStoreWidget.value = JSON.stringify(this.splines);
}

  this.vis = new pv.Panel()
  .width(this.width)
  .height(this.height)
  .fillStyle("#222")
  .strokeStyle("gray")
  .lineWidth(2)
  .antialias(false)
  .margin(10)
  .event("mousedown", function () {
    if (pv.event.shiftKey) { // Use pv.event to access the event object
        let scaledMouse = {
        x: this.mouse().x / app.canvas.ds.scale,
        y: this.mouse().y / app.canvas.ds.scale
        };
        i = self.splines[self.activeSplineIndex].points.push(scaledMouse) - 1;
        self.updatePath();
        return this;
    }
    else if (pv.event.ctrlKey) {
       // Capture the clicked location
       let clickedPoint = {
        x: this.mouse().x / app.canvas.ds.scale,
        y: this.mouse().y / app.canvas.ds.scale
        };

        // Find the two closest points to the clicked location
        const activePoints = self.splines[self.activeSplineIndex].points;
        let { point1Index, point2Index } = self.findClosestPoints(self.splines[self.activeSplineIndex].points, clickedPoint);

        // Calculate the midpoint between the two closest points
        let midpoint = {
            x: (activePoints[point1Index].x + activePoints[point2Index].x) / 2,
            y: (activePoints[point1Index].y + activePoints[point2Index].y) / 2
        };

        // Insert the midpoint into the array
        activePoints.splice(point2Index, 0, midpoint);
        i = point2Index;
        self.updatePath();
    }
    else if (pv.event.button === 2) {
      // Store the current mouse position adjusted for scale
      self.lastMousePosition = {
        x: this.mouse().x / app.canvas.ds.scale,
        y: this.mouse().y / app.canvas.ds.scale
      };

      self.node.contextMenu.style.display = 'block';
      self.node.contextMenu.style.left = `${pv.event.clientX}px`;
      self.node.contextMenu.style.top = `${pv.event.clientY}px`;
      }
    })
    this.backgroundImage = this.vis.add(pv.Image).visible(false)

  this.vis.add(pv.Rule)
    .data(pv.range(0, this.height, 64))
    .bottom(d =>  d)
    .strokeStyle("gray")
    .lineWidth(3)
    .visible(() => self.drawRuler)

  this.hoverSplineIndex = -1;

  this.splines.forEach((spline, splineIndex) => {
    const strokeObj = this.vis.add(pv.Line)
      .data(() => spline.points)
      .left(d => d.x)
      .top(d => d.y)
      .interpolate(() => this.interpolation)
      .tension(() => this.tension)
      .segmented(() => false)
      .strokeStyle("black") // Stroke color
      .lineWidth(() => {
        // Make stroke slightly wider than the main line
        if (splineIndex === this.activeSplineIndex) return 5;
        if (splineIndex === this.hoverSplineIndex) return 4;
        return 3.5;
      });

    this.vis.add(pv.Line)
      .data(() => spline.points)
      .left(d => d.x)
      .top(d => d.y)
      .interpolate(() => this.interpolation)
      .tension(() => this.tension)
      .segmented(() => false)
      .strokeStyle(spline.color)
      .lineWidth(() => {
        // Change line width based on active or hover state
        if (splineIndex === this.activeSplineIndex) return 3;
        if (splineIndex === this.hoverSplineIndex) return 2;
        return 1.5;
      })
      .event("mouseover", () => {
        this.hoverSplineIndex = splineIndex;
        this.vis.render();
      })
      .event("mouseout", () => {
        this.hoverSplineIndex = -1;
        this.vis.render();
      })
      .event("mousedown", () => {
        if (this.activeSplineIndex !== splineIndex) {
          this.activeSplineIndex = splineIndex;
          this.refreshSplineElements();
        }
      });
  });
    
    this.vis.add(pv.Dot)
      .data(() => {
        const activeSpline = this.splines[this.activeSplineIndex];
        // If this is a single point, don't show it in the main visualization
        if (activeSpline.isSinglePoint || (activeSpline.points && activeSpline.points.length === 1)) {
          return []; // Return empty array to hide in main visualization
        }
        return activeSpline.points;
      })
      .left(d => d.x)
      .top(d => d.y)
      .radius(12)
      .shape(function() {
        return self.dotShape;
    })
    .angle(function() {
      const index = this.index;
      let angle = 0;

      if (self.dotShape === "triangle") {
        const activePoints = self.splines[self.activeSplineIndex].points;
        let dxNext = 0, dyNext = 0;
        if (index < activePoints.length - 1) {
          dxNext = activePoints[index + 1].x - activePoints[index].x;
          dyNext = activePoints[index + 1].y - activePoints[index].y;
        }
      
        let dxPrev = 0, dyPrev = 0;
        if (index > 0) {
          dxPrev = activePoints[index].x - activePoints[index - 1].x;
          dyPrev = activePoints[index].y - activePoints[index - 1].y;
        }

        const dx = (dxNext + dxPrev) / 2;
        const dy = (dyNext + dyPrev) / 2;

        angle = Math.atan2(dy, dx);
        angle -= Math.PI / 2;
        angle = (angle + 2 * Math.PI) % (2 * Math.PI);
      }

    return angle;
 })
    .cursor("move")
    .strokeStyle(function () { return i == this.index ? "#ff7f0e" : "#1f77b4"; })
    .fillStyle(function () { return "rgba(100, 100, 100, 0.3)"; })
    .event("mousedown", pv.Behavior.drag())
    .event("dragstart", function () {
        i = this.index;
        hoverIndex = this.index;
        isDragging = true;
        const activePoints = self.splines[self.activeSplineIndex].points;
          if (pv.event.button === 2 && i !== 0 && i !== activePoints.length - 1) {
            activePoints.splice(i--, 1);
            self.vis.render();
          }
        return this;
    })
    .event("dragend", function() {
      if (this.pathElements !== null) {
        self.updatePath();
      }
        isDragging = false;
    })
    .event("drag", function () {
      let adjustedX = this.mouse().x / app.canvas.ds.scale; // Adjust the new X position by the inverse of the scale factor
      let adjustedY = this.mouse().y / app.canvas.ds.scale; // Adjust the new Y position by the inverse of the scale factor
       // Determine the bounds of the vis.Panel
      const panelWidth = self.vis.width();
      const panelHeight = self.vis.height();

      // Adjust the new position if it would place the dot outside the bounds of the vis.Panel
      adjustedX = Math.max(0, Math.min(panelWidth, adjustedX));
      adjustedY = Math.max(0, Math.min(panelHeight, adjustedY));
      self.splines[self.activeSplineIndex].points[this.index] = { x: adjustedX, y: adjustedY }; // Update the point's position
      self.vis.render(); // Re-render the visualization to reflect the new position
   })
    .event("mouseover", function() {
      hoverIndex = this.index; // Set the hover index to the index of the hovered dot
      self.vis.render(); // Re-render the visualization
    })
    .event("mouseout", function() {
      !isDragging && (hoverIndex = -1); // Reset the hover index when the mouse leaves the dot
      self.vis.render(); // Re-render the visualization
    })
    .anchor("center")
    .add(pv.Label)
    .visible(function() {
      return hoverIndex === this.index; // Only show the label for the hovered dot
    })
    .left(d => d.x < this.width / 2 ? d.x + 80 : d.x - 70) // Shift label to right if on left half, otherwise shift to left
    .top(d => d.y < this.height / 2 ? d.y + 20 : d.y - 20)  // Shift label down if on top half, otherwise shift up
    .font(12 + "px sans-serif")
    .text(d => {
      if (this.samplingMethod == "path") {
        return `X: ${Math.round(d.x)}, Y: ${Math.round(d.y)}`;
      } else {
          let frame = Math.round((d.x / self.width) * self.points_to_sample);
          let normalizedY = (1.0 - (d.y / self.height) - 0.0) * (self.rangeMax - self.rangeMin) + self.rangeMin;
          let normalizedX = (d.x / self.width);
          return `F: ${frame}, X: ${normalizedX.toFixed(2)}, Y: ${normalizedY.toFixed(2)}`;
      }
      })
    .textStyle("orange")

    // single points
    this.vis.add(pv.Dot)
      .data(() => {
        // Collect all single points from all splines
        const singlePoints = [];
        this.splines.forEach((spline, splineIndex) => {
          if (spline.isSinglePoint || (spline.points && spline.points.length === 1)) {
            singlePoints.push({
              x: spline.points[0].x, 
              y: spline.points[0].y,
              splineIndex: splineIndex,
              color: spline.color
            });
          }
        });
        return singlePoints;
      })
      .left(d => d.x)
      .top(d => d.y)
      .radius(6)
      .shape("square")
      .strokeStyle(d => d.splineIndex === this.activeSplineIndex ? "#ff7f0e" : d.color)
      .fillStyle(d => "rgba(100, 100, 100, 0.9)")
      .lineWidth(d => d.splineIndex === this.activeSplineIndex ? 3 : 1.5)
      .cursor("move")
      .event("mousedown", pv.Behavior.drag())
      .event("dragstart", function(d) {
        self.activeSplineIndex = d.splineIndex;
        self.refreshSplineElements();
        return this;
      })
      .event("drag", function(d) {
        let adjustedX = this.mouse().x / app.canvas.ds.scale;
        let adjustedY = this.mouse().y / app.canvas.ds.scale;
        
        // Determine the bounds of the vis.Panel
        const panelWidth = self.vis.width();
        const panelHeight = self.vis.height();

        // Adjust the new position if it would place the dot outside the bounds
        adjustedX = Math.max(0, Math.min(panelWidth, adjustedX));
        adjustedY = Math.max(0, Math.min(panelHeight, adjustedY));
        
        // Update the point position
        const spline = self.splines[d.splineIndex];
        spline.points[0] = { x: adjustedX, y: adjustedY };
        
        // For single points, we need to refresh the entire spline element
        // to prevent the line-drawing effect
        
      })
      .event("dragend", function(d) {
        self.refreshSplineElements();
        self.updatePath();
      })
      .visible(d => true);  // Make always visible
 
    if (this.splines.length != 0) {
        this.vis.render();
      }
      var svgElement = this.vis.canvas();
      svgElement.style['zIndex'] = "2"
      svgElement.style['position'] = "relative"
      this.node.splineEditor.element.appendChild(svgElement);
      this.pathElements = svgElement.getElementsByTagName('path'); // Get all path elements

      if (this.width > 256) {
        this.node.setSize([this.width + 45, this.node.size[1]]);
      }
      this.node.setSize([this.node.size[0], this.height + 450]);
      this.updatePath();
      this.refreshBackgroundImage();
  }

    updatePath = () => {
      if (!this.splines || this.splines.length === 0) {
        console.log("no splines");
        return;
      }
      // Get active spline points
      console.log("this.activeSplineIndex", this.activeSplineIndex);
      const activeSpline = this.splines[this.activeSplineIndex];
      const activePoints = activeSpline.points;
      
      if (!activePoints || activePoints.length === 0) {
        console.log("no points in active spline");
        return;
      }

      
      let coords;
      if (this.samplingMethod != "controlpoints") {
        coords = this.samplePoints(this.pathElements[this.activeSplineIndex], this.points_to_sample, this.samplingMethod, this.width, this.activeSplineIndex);
      } else {
        coords = activePoints;
      }

      let allSplineCoords = [];
      for (let i = 0; i < this.splines.length; i++) {
        // Use the same sampling method for all splines
        let splineCoords;
        const pathElement = this.pathElements[i];
        
        if (this.samplingMethod != "controlpoints" && pathElement) {
          splineCoords = this.samplePoints(pathElement, this.points_to_sample, this.samplingMethod, this.width, i);
        } else {
          // Fall back to control points if no path element or sampling method is "controlpoints"
          splineCoords = this.splines[i].points;
        }

        allSplineCoords.push(splineCoords);
      }

      if (this.drawSamplePoints) {
        if (this.pointsLayer) {
          // Update the data of the existing points layer
          this.pointsLayer.data(coords);
        } else {
            // Create the points layer if it doesn't exist
            this.pointsLayer = this.vis.add(pv.Dot)
                .data(coords)
                .left(function(d) { return d.x; })
                .top(function(d) { return d.y; })
                .radius(5) // Adjust the radius as needed
                .fillStyle("red") // Change the color as needed
                .strokeStyle("black") // Change the stroke color as needed
                .lineWidth(1); // Adjust the line width as needed
          }
      } else {
          if (this.pointsLayer) {
            // Remove the points layer
            this.pointsLayer.data([]);
            this.vis.render();
          }
      }
      this.pointsStoreWidget.value = JSON.stringify(this.splines);
      if (this.coordWidget) {
        this.coordWidget.value = JSON.stringify(allSplineCoords);
      }
      this.vis.render();
    };

    handleImageLoad = (img, file, base64String) => {
      //console.log(img.width, img.height); // Access width and height here
      this.widthWidget.value = img.width;
      this.heightWidget.value = img.height;
      this.drawRuler = false;

      if (img.width != this.vis.width() || img.height != this.vis.height()) {
        if (img.width > 256) {
          this.node.setSize([img.width + 45, this.node.size[1]]);
        }
        this.node.setSize([this.node.size[0], img.height + 520]);
        this.vis.width(img.width);
        this.vis.height(img.height);
        this.height = img.height;
        this.width = img.width;
        
        this.updatePath();
      }
      this.backgroundImage.url(file ? URL.createObjectURL(file) : `data:${this.node.properties.imgData.type};base64,${base64String}`).visible(true).root.render();
      };

    processImage = (img, file) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      const maxWidth = 800; // maximum width
      const maxHeight = 600; // maximum height
      let width = img.width;
      let height = img.height;

      // Calculate the new dimensions while preserving the aspect ratio
      if (width > height) {
        if (width > maxWidth) {
          height *= maxWidth / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width *= maxHeight / height;
          height = maxHeight;
        }
      }

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);

      // Get the compressed image data as a Base64 string
      const base64String = canvas.toDataURL('image/jpeg', 0.5).replace('data:', '').replace(/^.+,/, ''); // 0.5 is the quality from 0 to 1

      this.node.properties.imgData = {
        name: file.name,
        lastModified: file.lastModified,
        size: file.size,
        type: file.type,
        base64: base64String
      };
      handleImageLoad(img, file, base64String);
    };

    handleImageFile = (file) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const img = new Image();
        img.src = reader.result;
        img.onload = () => processImage(img, file);
      };
      reader.readAsDataURL(file);

      const imageUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => this.handleImageLoad(img, file, null);
    };

    refreshBackgroundImage = () => {
      if (this.node.properties.imgData && this.node.properties.imgData.base64) {
        const base64String = this.node.properties.imgData.base64;
        const imageUrl = `data:${this.node.properties.imgData.type};base64,${base64String}`;
        const img = new Image();
        img.src = imageUrl;
        img.onload = () => this.handleImageLoad(img, null, base64String);
      }
    };

    refreshSplineElements = () => {
      // Clear existing line elements and recreate them
      const svgElement = this.vis.canvas();
      
      // Remove all existing line elements
      const oldLines = svgElement.querySelectorAll('path');
      oldLines.forEach(line => line.remove());
      
      this.pathElements = [];
      this.lineObjects = [];

      const originalChildren = [...this.vis.children];
  
      // Find line objects to remove (those that represent splines)
      const linesToRemove = originalChildren.filter(child => 
        child instanceof pv.Line
      );
      linesToRemove.forEach(line => line.visible(false));
      
      // Re-add all spline lines and store references to them
      this.splines.forEach((spline, splineIndex) => {
        // For single points, we need a special handling
        if (spline.isSinglePoint || (spline.points && spline.points.length === 1)) {
          const point = spline.points[0];
          // For single points, create a tiny line at the same point
          // This ensures we have a path element for the point
          const lineObj = this.vis.add(pv.Line)
            .data([point, {x: point.x + 0.001, y: point.y + 0.001}])
            .left(d => d.x)
            .top(d => d.y)
            .strokeStyle(spline.color)
            .lineWidth(() => {
              if (splineIndex === this.activeSplineIndex) return 3;
              if (splineIndex === this.hoverSplineIndex) return 2;
              return 1.5;
            })
            .event("mouseover", () => {
              this.hoverSplineIndex = splineIndex;
              this.vis.render();
            })
            .event("mouseout", () => {
              this.hoverSplineIndex = -1;
              this.vis.render();
            })
            .event("mousedown", () => {
              if (this.activeSplineIndex !== splineIndex) {
                this.activeSplineIndex = splineIndex;
                this.refreshSplineElements();
              }
            });
          this.lineObjects.push(lineObj);
        } else {
          // For normal multi-point splines
          const strokeObj = this.vis.add(pv.Line)
            .data(() => spline.points)
            .left(d => d.x)
            .top(d => d.y)
            .interpolate(() => this.interpolation)
            .tension(() => this.tension)
            .segmented(() => false)
            .strokeStyle("black") // Stroke color
            .lineWidth(() => {
              // Make stroke slightly wider than the main line
              if (splineIndex === this.activeSplineIndex) return 5;
              if (splineIndex === this.hoverSplineIndex) return 4;
              return 3.5;
            });
          const lineObj = this.vis.add(pv.Line)
          .data(() => spline.points)
          .left(d => d.x)
          .top(d => d.y)
          .interpolate(() => this.interpolation)
          .tension(() => this.tension)
          .segmented(() => false)
          .strokeStyle(spline.color)
          .lineWidth(() => {
            if (splineIndex === this.activeSplineIndex) return 3;
            if (splineIndex === this.hoverSplineIndex) return 2;
            return 1.5;
          })
          .event("mouseover", () => {
            this.hoverSplineIndex = splineIndex;
            this.vis.render();
          })
          .event("mouseout", () => {
            this.hoverSplineIndex = -1;
            this.vis.render();
          })
          .event("mousedown", () => {
            if (this.activeSplineIndex !== splineIndex) {
              this.activeSplineIndex = splineIndex;
              this.refreshSplineElements();
            }
          });

          // // Add invisible wider hit area for easier selection
          // this.vis.add(pv.Line)
          // .data(() => spline.points)
          // .left(d => d.x)
          // .top(d => d.y)
          // .interpolate(() => this.interpolation)
          // .tension(() => this.tension)
          // .segmented(() => false)
          // .strokeStyle("rgba(0,0,0,0.01)") // Nearly invisible
          // .lineWidth(15) // Much wider hit area
          // .event("mouseover", () => {
          //   this.hoverSplineIndex = splineIndex;
          //   this.vis.render();
          // })
          // .event("mouseout", () => {
          //   this.hoverSplineIndex = -1;
          //   this.vis.render();
          // })
          // .event("mousedown", () => {
          //   if (pv.event.shiftKey) {
          //     if (this.activeSplineIndex !== splineIndex) {
          //       this.activeSplineIndex = splineIndex;
          //       this.refreshSplineElements();
          //     }
          //   }}
          // );

          this.lineObjects.push(lineObj);
        }
      });
      
      this.vis.render();
      
      requestAnimationFrame(() => {
        const allPaths = Array.from(svgElement.querySelectorAll('path'));
        this.pathElements = [];
        
        // First try: look at paths with specific childIndex values
        this.lineObjects.forEach((lineObj, i) => {
          // Find paths that correspond to our line objects
          const childIndex = lineObj.childIndex;
          const matchingPath = allPaths.find(path => 
            path.$scene && path.$scene.scenes && 
            path.$scene.scenes.childIndex === childIndex
          );
          
          if (matchingPath) {
            //console.log("matchingPath:", matchingPath);
            this.pathElements[i] = matchingPath;
          }
        });
        
        // Check if we found all paths
        if (this.pathElements.filter(p => p).length !== this.splines.length) {
          // Fallback to color matching
          this.pathElements = [];
          for (let i = 0; i < this.splines.length; i++) {
            const color = this.splines[i].color;
            const matchingPath = allPaths.find(p => 
              p.getAttribute('style')?.includes(color) && 
              !this.pathElements.includes(p)
            );
            
            if (matchingPath) {
              this.pathElements[i] = matchingPath;
            }
          }
        }
        
        // If we still don't have the right number of paths, use the first N paths
        if (this.pathElements.filter(p => p).length !== this.splines.length) {
          this.pathElements = allPaths.slice(0, this.splines.length);
        }
        
        this.updatePath();
      });
    };
    
    
    initializeDefaultSplines() {
      this.splines = [{
        points: pv.range(1, 4).map((i, index) => {
          if (index === 0) {
            return { x: 0, y: this.height };
          } else if (index === 2) {
            return { x: this.width, y: 0 };
          } else {
            return {
              x: i * this.width / 5,
              y: 50 + Math.random() * (this.height - 100)
            };
          }
        }),
        color: this.getSplineColor(0),
        name: "Spline 1"
      }];
    }

    getSplineColor(index) {
      const colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
        "#bcbd22", "#17becf"
      ];
      return colors[index % colors.length];
    }

  createContextMenu = () => {
    const self = this;
    const oldMenu = this.node.contextMenu;
    const newMenu = oldMenu.cloneNode(true);
    oldMenu.parentNode.replaceChild(newMenu, oldMenu);
    this.node.contextMenu = newMenu;

    document.addEventListener('contextmenu', function (e) {
      e.preventDefault();
    });

    document.addEventListener('click', function (e) {
      document.querySelectorAll('.spline-editor-context-menu').forEach(menu => {
        menu.style.display = 'none';
    });
    });

    this.node.contextMenu.addEventListener('click', function(e) {
      e.preventDefault();
      if (e.target.tagName === 'A') {
        const id = parseInt(e.target.id.split('-')[2]);
        
        switch(id) {
          case 0:
            e.preventDefault();
            if (!self.drawHandles) {
              self.drawHandles = true
              self.vis.add(pv.Line)
              .data(() => self.splines[self.activeSplineIndex].points.map((point, index) => ({
                  start: point,
                  end: [index]
              })))
              .left(d => d.start.x)
              .top(d => d.start.y)
              .interpolate("linear")
              .tension(0) // Straight lines
              .strokeStyle("#ff7f0e") // Same color as control points
              .lineWidth(1)
              .visible(() => self.drawHandles);
              self.vis.render();
            } else {
              self.drawHandles = false
              self.vis.render();
            }
            self.node.contextMenu.style.display = 'none';
            break;
          case 1:
   
            self.drawSamplePoints = !self.drawSamplePoints;
            self.updatePath();
            break;
          case 2:
            if (self.dotShape == "circle"){
              self.dotShape = "triangle"
            }
            else {
              self.dotShape = "circle"
            }
            self.updatePath();
            break;
          case 3:
            // Create file input element
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*'; // Accept only image files

            // Listen for file selection
            fileInput.addEventListener('change', function (event) {
              const file = event.target.files[0]; // Get the selected file

              if (file) {
                const imageUrl = URL.createObjectURL(file);
                let img = new Image();
                img.src = imageUrl;
                img.onload = () => self.handleImageLoad(img, file, null);
              }
            });

              fileInput.click();
            
            self.node.contextMenu.style.display = 'none';
            break;
          case 4:
            self.splines[self.activeSplineIndex].points.reverse();
            self.updatePath();
          break;
          case 5:
            self.backgroundImage.visible(false).root.render();
            self.node.properties.imgData = null;
            self.node.contextMenu.style.display = 'none';
          break;
          case 6: // Add new spline
            const newSplineIndex = self.splines.length;
            self.splines.push({
              points: [
                // Create default points for the new spline
                { x: 0, y: self.height },
                { x: self.width/2, y: self.height/2 },
                { x: self.width, y: 0 }
              ],
              color: self.getSplineColor(newSplineIndex),
              name: `Spline ${newSplineIndex + 1}`
            });
            self.activeSplineIndex = newSplineIndex;
            self.refreshSplineElements();
            self.node.contextMenu.style.display = 'none';
            break;
          case 7: // Add new single point
            const newSingleSplineIndex = self.splines.length;
            self.splines.push({
              points: [
                { x: self.lastMousePosition.x, y: self.lastMousePosition.y },
              ],
              color: self.getSplineColor(newSingleSplineIndex),
              name: `Spline ${newSingleSplineIndex + 1}`,
              isSinglePoint: true
            });
            self.activeSplineIndex = newSingleSplineIndex;
            self.refreshSplineElements();
            self.node.contextMenu.style.display = 'none';
            break;
          case 8: // Delete current spline
            if (self.splines.length > 1) {
              self.splines.splice(self.activeSplineIndex, 1);
              self.activeSplineIndex = Math.min(self.activeSplineIndex, self.splines.length - 1);
              self.refreshSplineElements();
            }
            self.node.contextMenu.style.display = 'none';
            break;
          case 9: // Next spline
            self.activeSplineIndex = (self.activeSplineIndex + 1) % self.splines.length;
            self.refreshSplineElements();
            self.node.contextMenu.style.display = 'none';
            break;
        }
      }
    });
  }

  samplePoints(svgPathElement, numSamples, samplingMethod, width, splineIndex) {
    const spline = this.splines[splineIndex];
  
    // Check if this is a single point spline
    if (spline && (spline.isSinglePoint || (spline.points && spline.points.length === 1))) {
      // For a single point, return an array with the same coordinates repeated
      const point = spline.points[0];
      return Array(numSamples).fill().map(() => ({ x: point.x, y: point.y }));
    }

    if (!svgPathElement) {
      console.warn(`Path element not found for spline index: ${splineIndex}. Available paths: ${this.pathElements.length}`);


      const splinePoints = this.splines[splineIndex].points;
      
      // If we have no points, return an empty array
      if (!splinePoints || splinePoints.length === 0) {
        return [];
      }
      
      // Create a simple interpolation between control points
      const result = [];
      for (let i = 0; i < numSamples; i++) {
        const t = i / (numSamples - 1);
        const idx = Math.min(
          Math.floor(t * (splinePoints.length - 1)), 
          splinePoints.length - 2
        );
        const fraction = (t * (splinePoints.length - 1)) - idx;
        
        const x = splinePoints[idx].x + fraction * (splinePoints[idx + 1].x - splinePoints[idx].x);
        const y = splinePoints[idx].y + fraction * (splinePoints[idx + 1].y - splinePoints[idx].y);
        
        result.push({ x, y });
      }
      return result;
    }
    
    var svgWidth = width; // Fixed width of the SVG element
    var pathLength = svgPathElement.getTotalLength();
    var points = [];

    if (samplingMethod === "speed") {
      // Calculate control point distances along the path
      const controlPoints = this.splines[splineIndex].points;
      const pathPositions = [];
      
      // Find approximate path positions for each control point
      for (const cp of controlPoints) {
        let bestDist = Infinity;
        let bestPos = 0;
        
        // Sample the path to find closest point to each control point
        for (let pos = 0; pos <= pathLength; pos += pathLength / 100) {
          const pt = svgPathElement.getPointAtLength(pos);
          const dist = Math.sqrt(Math.pow(pt.x - cp.x, 2) + Math.pow(pt.y - cp.y, 2));
          
          if (dist < bestDist) {
            bestDist = dist;
            bestPos = pos;
          }
        }
        pathPositions.push(bestPos);
      }
      
      // Sort positions along path
      pathPositions.sort((a, b) => a - b);
      
      // Create a smooth speed mapping function with synchronization
      const createSynchronizedMapping = () => {
        // Calculate segment lengths and densities
        const segments = [];
        let totalLength = pathPositions[pathPositions.length - 1] - pathPositions[0];
        
        for (let i = 0; i < pathPositions.length - 1; i++) {
          const segLength = pathPositions[i+1] - pathPositions[i];
          // Inverse relationship - shorter segments = higher density = slower speed
          const density = 1 / Math.max(segLength, 0.0001);
          segments.push({ 
            position: pathPositions[i],
            length: segLength, 
            density: density
          });
        }
        
        // Create mapping function with forced synchronization at endpoints
        return t => {
          // Force synchronization at t=0 and t=1
          if (t === 0) return 0;
          if (t === 1) return pathLength;
          
          // For intermediate points, use the speed control
          // Scale t to fit between first and last control points
          const firstPos = pathPositions[0];
          const lastPos = pathPositions[pathPositions.length - 1];
          
          // Create a density-weighted position mapping
          let totalWeight = 0;
          let weights = [];
          
          for (let i = 0; i < segments.length; i++) {
            totalWeight += segments[i].density;
            weights.push(segments[i].density);
          }
          
          // Normalize weights
          const normalizedWeights = weights.map(w => w / totalWeight);
          
          // Calculate cumulative weights
          let cumulativeWeight = 0;
          const cumulativeWeights = normalizedWeights.map(w => {
            cumulativeWeight += w;
            return cumulativeWeight;
          });
          
          // Find the segment for this t value
          let segmentIndex = 0;
          for (let i = 0; i < cumulativeWeights.length; i++) {
            if (t <= cumulativeWeights[i]) {
              segmentIndex = i;
              break;
            }
          }
          
          // Calculate position within segment
          const segmentStart = segmentIndex > 0 ? cumulativeWeights[segmentIndex - 1] : 0;
          const segmentEnd = cumulativeWeights[segmentIndex];
          const segmentT = (t - segmentStart) / (segmentEnd - segmentStart);
          
          // Map to path position
          const pathStart = pathPositions[segmentIndex];
          const pathEnd = pathPositions[segmentIndex + 1];
          const pos = pathStart + segmentT * (pathEnd - pathStart);
          
          // Scale to fill entire path
          return pos;
        };
      };
      
      const mapToPath = createSynchronizedMapping();
      
      // Sample using the synchronized mapping function
      for (let i = 0; i < numSamples; i++) {
        const t = i / (numSamples - 1);
        const pathPos = mapToPath(t);
        const point = svgPathElement.getPointAtLength(pathPos);
        points.push({ x: point.x, y: point.y });
      }
      
      return points;
    
    }
    else{
    for (var i = 0; i < numSamples; i++) {
        if (samplingMethod === "time") {
          // Calculate the x-coordinate for the current sample based on the SVG's width
          var x = (svgWidth / (numSamples - 1)) * i;
          // Find the point on the path that intersects the vertical line at the calculated x-coordinate
          var point = this.findPointAtX(svgPathElement, x, pathLength);
          }
        else if (samplingMethod === "path") {
          // Calculate the distance along the path for the current sample
          var distance = (pathLength / (numSamples - 1)) * i;
          // Get the point at the current distance
          var point = svgPathElement.getPointAtLength(distance);
        }

        // Add the point to the array of points
        points.push({ x: point.x, y: point.y });
        }
    return points;
    }
  }

  findClosestPoints(points, clickedPoint) {
      // Calculate distances from clickedPoint to each point in the array
      let distances = points.map(point => {
          let dx = clickedPoint.x - point.x;
          let dy = clickedPoint.y - point.y;
          return { index: points.indexOf(point), distance: Math.sqrt(dx * dx + dy * dy) };
      });
      // Sort distances and get the indices of the two closest points
      let sortedDistances = distances.sort((a, b) => a.distance - b.distance);
      let closestPoint1Index = sortedDistances[0].index;
      let closestPoint2Index = sortedDistances[1].index;
      // Ensure point1Index is always the smaller index
    if (closestPoint1Index > closestPoint2Index) {
      [closestPoint1Index, closestPoint2Index] = [closestPoint2Index, closestPoint1Index];
      }
      return { point1Index: closestPoint1Index, point2Index: closestPoint2Index };
    }

  findPointAtX(svgPathElement, targetX, pathLength) {
    let low = 0;
    let high = pathLength;
    let bestPoint = svgPathElement.getPointAtLength(0);

    while (low <= high) {
        let mid = low + (high - low) / 2;
        let point = svgPathElement.getPointAtLength(mid);

        if (Math.abs(point.x - targetX) < 1) {
            return point; // The point is close enough to the target
        }

        if (point.x < targetX) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }

        // Keep track of the closest point found so far
        if (Math.abs(point.x - targetX) < Math.abs(bestPoint.x - targetX)) {
            bestPoint = point;
        }
    }

    // Return the closest point found
    return bestPoint;
  }
}