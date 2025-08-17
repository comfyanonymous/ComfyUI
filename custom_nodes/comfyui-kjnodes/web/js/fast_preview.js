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
  name: 'KJNodes.FastPreview',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'FastPreview') {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {

        var element = document.createElement("div");
        this.uuid = makeUUID()
        element.id = `fast-preview-${this.uuid}`

        this.previewWidget = this.addDOMWidget(nodeData.name, "FastPreviewWidget", element, {
          serialize: false,
          hideOnZoom: false,
        });

        this.previewer = new Previewer(this);

        this.setSize([550, 550]);
        this.resizable = false;
        this.previewWidget.parentEl = document.createElement("div");
        this.previewWidget.parentEl.className = "fast-preview";
        this.previewWidget.parentEl.id = `fast-preview-${this.uuid}`
        element.appendChild(this.previewWidget.parentEl);
        
        chainCallback(this, "onExecuted", function (message) {
          let bg_image = message["bg_image"];
          this.properties.imgData = {
            name: "bg_image",
            base64: bg_image
          };
          this.previewer.refreshBackgroundImage(this);
        });
       

      }); // onAfterGraphConfigured
    }//node created
  } //before register
})//register

class Previewer {
  constructor(context) {
    this.node = context;
    this.previousWidth = null;
    this.previousHeight = null;
  }
  refreshBackgroundImage = () => {
    const imgData = this.node?.properties?.imgData;
    if (imgData?.base64) {
      const base64String = imgData.base64;
      const imageUrl = `data:${imgData.type};base64,${base64String}`;
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => {
        const { width, height } = img;
        if (width !== this.previousWidth || height !== this.previousHeight) {
          this.node.setSize([width, height]);
          this.previousWidth = width;
          this.previousHeight = height;
        }
        this.node.previewWidget.element.style.backgroundImage = `url(${imageUrl})`;
      };
    }
  };
  }