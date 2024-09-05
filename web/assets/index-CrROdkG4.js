var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { C as ComfyDialog, $ as $el, a as ComfyApp, b as app, L as LGraphCanvas, c as LiteGraph, d as LGraphNode, e as applyTextReplacements, f as ComfyWidgets, g as addValueControlWidgets, D as DraggableList, h as api, i as LGraphGroup, u as useToastStore } from "./index-Dfv2aLsq.js";
class ClipspaceDialog extends ComfyDialog {
  static {
    __name(this, "ClipspaceDialog");
  }
  static items = [];
  static instance = null;
  static registerButton(name, contextPredicate, callback) {
    const item = $el("button", {
      type: "button",
      textContent: name,
      contextPredicate,
      onclick: callback
    });
    ClipspaceDialog.items.push(item);
  }
  static invalidatePreview() {
    if (ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0) {
      const img_preview = document.getElementById(
        "clipspace_preview"
      );
      if (img_preview) {
        img_preview.src = ComfyApp.clipspace.imgs[ComfyApp.clipspace["selectedIndex"]].src;
        img_preview.style.maxHeight = "100%";
        img_preview.style.maxWidth = "100%";
      }
    }
  }
  static invalidate() {
    if (ClipspaceDialog.instance) {
      const self = ClipspaceDialog.instance;
      const children = $el("div.comfy-modal-content", [
        self.createImgSettings(),
        ...self.createButtons()
      ]);
      if (self.element) {
        self.element.removeChild(self.element.firstChild);
        self.element.appendChild(children);
      } else {
        self.element = $el("div.comfy-modal", { parent: document.body }, [
          children
        ]);
      }
      if (self.element.children[0].children.length <= 1) {
        self.element.children[0].appendChild(
          $el("p", {}, [
            "Unable to find the features to edit content of a format stored in the current Clipspace."
          ])
        );
      }
      ClipspaceDialog.invalidatePreview();
    }
  }
  constructor() {
    super();
  }
  createButtons() {
    const buttons = [];
    for (let idx in ClipspaceDialog.items) {
      const item = ClipspaceDialog.items[idx];
      if (!item.contextPredicate || item.contextPredicate())
        buttons.push(ClipspaceDialog.items[idx]);
    }
    buttons.push(
      $el("button", {
        type: "button",
        textContent: "Close",
        onclick: /* @__PURE__ */ __name(() => {
          this.close();
        }, "onclick")
      })
    );
    return buttons;
  }
  createImgSettings() {
    if (ComfyApp.clipspace.imgs) {
      const combo_items = [];
      const imgs = ComfyApp.clipspace.imgs;
      for (let i = 0; i < imgs.length; i++) {
        combo_items.push($el("option", { value: i }, [`${i}`]));
      }
      const combo1 = $el(
        "select",
        {
          id: "clipspace_img_selector",
          onchange: /* @__PURE__ */ __name((event) => {
            ComfyApp.clipspace["selectedIndex"] = event.target.selectedIndex;
            ClipspaceDialog.invalidatePreview();
          }, "onchange")
        },
        combo_items
      );
      const row1 = $el("tr", {}, [
        $el("td", {}, [$el("font", { color: "white" }, ["Select Image"])]),
        $el("td", {}, [combo1])
      ]);
      const combo2 = $el(
        "select",
        {
          id: "clipspace_img_paste_mode",
          onchange: /* @__PURE__ */ __name((event) => {
            ComfyApp.clipspace["img_paste_mode"] = event.target.value;
          }, "onchange")
        },
        [
          $el("option", { value: "selected" }, "selected"),
          $el("option", { value: "all" }, "all")
        ]
      );
      combo2.value = ComfyApp.clipspace["img_paste_mode"];
      const row2 = $el("tr", {}, [
        $el("td", {}, [$el("font", { color: "white" }, ["Paste Mode"])]),
        $el("td", {}, [combo2])
      ]);
      const td = $el(
        "td",
        { align: "center", width: "100px", height: "100px", colSpan: "2" },
        [$el("img", { id: "clipspace_preview", ondragstart: /* @__PURE__ */ __name(() => false, "ondragstart") }, [])]
      );
      const row3 = $el("tr", {}, [td]);
      return $el("table", {}, [row1, row2, row3]);
    } else {
      return [];
    }
  }
  createImgPreview() {
    if (ComfyApp.clipspace.imgs) {
      return $el("img", { id: "clipspace_preview", ondragstart: /* @__PURE__ */ __name(() => false, "ondragstart") });
    } else return [];
  }
  show() {
    const img_preview = document.getElementById("clipspace_preview");
    ClipspaceDialog.invalidate();
    this.element.style.display = "block";
  }
}
app.registerExtension({
  name: "Comfy.Clipspace",
  init(app2) {
    app2.openClipspace = function() {
      if (!ClipspaceDialog.instance) {
        ClipspaceDialog.instance = new ClipspaceDialog();
        ComfyApp.clipspace_invalidate_handler = ClipspaceDialog.invalidate;
      }
      if (ComfyApp.clipspace) {
        ClipspaceDialog.instance.show();
      } else app2.ui.dialog.show("Clipspace is Empty!");
    };
  }
});
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.clipspace = window.comfyAPI.clipspace || {};
window.comfyAPI.clipspace.ClipspaceDialog = ClipspaceDialog;
const colorPalettes = {
  dark: {
    id: "dark",
    name: "Dark (Default)",
    colors: {
      node_slot: {
        CLIP: "#FFD500",
        // bright yellow
        CLIP_VISION: "#A8DADC",
        // light blue-gray
        CLIP_VISION_OUTPUT: "#ad7452",
        // rusty brown-orange
        CONDITIONING: "#FFA931",
        // vibrant orange-yellow
        CONTROL_NET: "#6EE7B7",
        // soft mint green
        IMAGE: "#64B5F6",
        // bright sky blue
        LATENT: "#FF9CF9",
        // light pink-purple
        MASK: "#81C784",
        // muted green
        MODEL: "#B39DDB",
        // light lavender-purple
        STYLE_MODEL: "#C2FFAE",
        // light green-yellow
        VAE: "#FF6E6E",
        // bright red
        NOISE: "#B0B0B0",
        // gray
        GUIDER: "#66FFFF",
        // cyan
        SAMPLER: "#ECB4B4",
        // very soft red
        SIGMAS: "#CDFFCD",
        // soft lime green
        TAESD: "#DCC274"
        // cheesecake
      },
      litegraph_base: {
        BACKGROUND_IMAGE: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII=",
        CLEAR_BACKGROUND_COLOR: "#222",
        NODE_TITLE_COLOR: "#999",
        NODE_SELECTED_TITLE_COLOR: "#FFF",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#AAA",
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#333",
        NODE_DEFAULT_BGCOLOR: "#353535",
        NODE_DEFAULT_BOXCOLOR: "#666",
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#FFF",
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: "#222",
        WIDGET_OUTLINE_COLOR: "#666",
        WIDGET_TEXT_COLOR: "#DDD",
        WIDGET_SECONDARY_TEXT_COLOR: "#999",
        LINK_COLOR: "#9A9",
        EVENT_LINK_COLOR: "#A86",
        CONNECTING_LINK_COLOR: "#AFA"
      },
      comfy_base: {
        "fg-color": "#fff",
        "bg-color": "#202020",
        "comfy-menu-bg": "#353535",
        "comfy-input-bg": "#222",
        "input-text": "#ddd",
        "descrip-text": "#999",
        "drag-text": "#ccc",
        "error-text": "#ff4444",
        "border-color": "#4e4e4e",
        "tr-even-bg-color": "#222",
        "tr-odd-bg-color": "#353535",
        "content-bg": "#4e4e4e",
        "content-fg": "#fff",
        "content-hover-bg": "#222",
        "content-hover-fg": "#fff"
      }
    }
  },
  light: {
    id: "light",
    name: "Light",
    colors: {
      node_slot: {
        CLIP: "#FFA726",
        // orange
        CLIP_VISION: "#5C6BC0",
        // indigo
        CLIP_VISION_OUTPUT: "#8D6E63",
        // brown
        CONDITIONING: "#EF5350",
        // red
        CONTROL_NET: "#66BB6A",
        // green
        IMAGE: "#42A5F5",
        // blue
        LATENT: "#AB47BC",
        // purple
        MASK: "#9CCC65",
        // light green
        MODEL: "#7E57C2",
        // deep purple
        STYLE_MODEL: "#D4E157",
        // lime
        VAE: "#FF7043"
        // deep orange
      },
      litegraph_base: {
        BACKGROUND_IMAGE: "data:image/gif;base64,R0lGODlhZABkALMAAAAAAP///+vr6+rq6ujo6Ofn5+bm5uXl5d3d3f///wAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAAkALAAAAABkAGQAAAT/UMhJq7046827HkcoHkYxjgZhnGG6si5LqnIM0/fL4qwwIMAg0CAsEovBIxKhRDaNy2GUOX0KfVFrssrNdpdaqTeKBX+dZ+jYvEaTf+y4W66mC8PUdrE879f9d2mBeoNLfH+IhYBbhIx2jkiHiomQlGKPl4uZe3CaeZifnnijgkESBqipqqusra6vsLGys62SlZO4t7qbuby7CLa+wqGWxL3Gv3jByMOkjc2lw8vOoNSi0czAncXW3Njdx9Pf48/Z4Kbbx+fQ5evZ4u3k1fKR6cn03vHlp7T9/v8A/8Gbp4+gwXoFryXMB2qgwoMMHyKEqA5fxX322FG8tzBcRnMW/zlulPbRncmQGidKjMjyYsOSKEF2FBlJQMCbOHP6c9iSZs+UnGYCdbnSo1CZI5F64kn0p1KnTH02nSoV3dGTV7FFHVqVq1dtWcMmVQZTbNGu72zqXMuW7danVL+6e4t1bEy6MeueBYLXrNO5Ze36jQtWsOG97wIj1vt3St/DjTEORss4nNq2mDP3e7w4r1bFkSET5hy6s2TRlD2/mSxXtSHQhCunXo26NevCpmvD/UU6tuullzULH76q92zdZG/Ltv1a+W+osI/nRmyc+fRi1Xdbh+68+0vv10dH3+77KD/i6IdnX669/frn5Zsjh4/2PXju8+8bzc9/6fj27LFnX11/+IUnXWl7BJfegm79FyB9JOl3oHgSklefgxAC+FmFGpqHIYcCfkhgfCohSKKJVo044YUMttggiBkmp6KFXw1oII24oYhjiDByaKOOHcp3Y5BD/njikSkO+eBREQAAOw==",
        CLEAR_BACKGROUND_COLOR: "lightgray",
        NODE_TITLE_COLOR: "#222",
        NODE_SELECTED_TITLE_COLOR: "#000",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#444",
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#F7F7F7",
        NODE_DEFAULT_BGCOLOR: "#F5F5F5",
        NODE_DEFAULT_BOXCOLOR: "#CCC",
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#000",
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.1)",
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: "#D4D4D4",
        WIDGET_OUTLINE_COLOR: "#999",
        WIDGET_TEXT_COLOR: "#222",
        WIDGET_SECONDARY_TEXT_COLOR: "#555",
        LINK_COLOR: "#4CAF50",
        EVENT_LINK_COLOR: "#FF9800",
        CONNECTING_LINK_COLOR: "#2196F3"
      },
      comfy_base: {
        "fg-color": "#222",
        "bg-color": "#DDD",
        "comfy-menu-bg": "#F5F5F5",
        "comfy-input-bg": "#C9C9C9",
        "input-text": "#222",
        "descrip-text": "#444",
        "drag-text": "#555",
        "error-text": "#F44336",
        "border-color": "#888",
        "tr-even-bg-color": "#f9f9f9",
        "tr-odd-bg-color": "#fff",
        "content-bg": "#e0e0e0",
        "content-fg": "#222",
        "content-hover-bg": "#adadad",
        "content-hover-fg": "#222"
      }
    }
  },
  solarized: {
    id: "solarized",
    name: "Solarized",
    colors: {
      node_slot: {
        CLIP: "#2AB7CA",
        // light blue
        CLIP_VISION: "#6c71c4",
        // blue violet
        CLIP_VISION_OUTPUT: "#859900",
        // olive green
        CONDITIONING: "#d33682",
        // magenta
        CONTROL_NET: "#d1ffd7",
        // light mint green
        IMAGE: "#5940bb",
        // deep blue violet
        LATENT: "#268bd2",
        // blue
        MASK: "#CCC9E7",
        // light purple-gray
        MODEL: "#dc322f",
        // red
        STYLE_MODEL: "#1a998a",
        // teal
        UPSCALE_MODEL: "#054A29",
        // dark green
        VAE: "#facfad"
        // light pink-orange
      },
      litegraph_base: {
        NODE_TITLE_COLOR: "#fdf6e3",
        // Base3
        NODE_SELECTED_TITLE_COLOR: "#A9D400",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#657b83",
        // Base00
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#094656",
        NODE_DEFAULT_BGCOLOR: "#073642",
        // Base02
        NODE_DEFAULT_BOXCOLOR: "#839496",
        // Base0
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#fdf6e3",
        // Base3
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: "#002b36",
        // Base03
        WIDGET_OUTLINE_COLOR: "#839496",
        // Base0
        WIDGET_TEXT_COLOR: "#fdf6e3",
        // Base3
        WIDGET_SECONDARY_TEXT_COLOR: "#93a1a1",
        // Base1
        LINK_COLOR: "#2aa198",
        // Solarized Cyan
        EVENT_LINK_COLOR: "#268bd2",
        // Solarized Blue
        CONNECTING_LINK_COLOR: "#859900"
        // Solarized Green
      },
      comfy_base: {
        "fg-color": "#fdf6e3",
        // Base3
        "bg-color": "#002b36",
        // Base03
        "comfy-menu-bg": "#073642",
        // Base02
        "comfy-input-bg": "#002b36",
        // Base03
        "input-text": "#93a1a1",
        // Base1
        "descrip-text": "#586e75",
        // Base01
        "drag-text": "#839496",
        // Base0
        "error-text": "#dc322f",
        // Solarized Red
        "border-color": "#657b83",
        // Base00
        "tr-even-bg-color": "#002b36",
        "tr-odd-bg-color": "#073642",
        "content-bg": "#657b83",
        "content-fg": "#fdf6e3",
        "content-hover-bg": "#002b36",
        "content-hover-fg": "#fdf6e3"
      }
    }
  },
  arc: {
    id: "arc",
    name: "Arc",
    colors: {
      node_slot: {
        BOOLEAN: "",
        CLIP: "#eacb8b",
        CLIP_VISION: "#A8DADC",
        CLIP_VISION_OUTPUT: "#ad7452",
        CONDITIONING: "#cf876f",
        CONTROL_NET: "#00d78d",
        CONTROL_NET_WEIGHTS: "",
        FLOAT: "",
        GLIGEN: "",
        IMAGE: "#80a1c0",
        IMAGEUPLOAD: "",
        INT: "",
        LATENT: "#b38ead",
        LATENT_KEYFRAME: "",
        MASK: "#a3bd8d",
        MODEL: "#8978a7",
        SAMPLER: "",
        SIGMAS: "",
        STRING: "",
        STYLE_MODEL: "#C2FFAE",
        T2I_ADAPTER_WEIGHTS: "",
        TAESD: "#DCC274",
        TIMESTEP_KEYFRAME: "",
        UPSCALE_MODEL: "",
        VAE: "#be616b"
      },
      litegraph_base: {
        BACKGROUND_IMAGE: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAACXBIWXMAAAsTAAALEwEAmpwYAAABcklEQVR4nO3YMUoDARgF4RfxBqZI6/0vZqFn0MYtrLIQMFN8U6V4LAtD+Jm9XG/v30OGl2e/AP7yevz4+vx45nvgF/+QGITEICQGITEIiUFIjNNC3q43u3/YnRJyPOzeQ+0e220nhRzReC8e7R7bbdvl+Jal1Bs46jEIiUFIDEJiEBKDkBhKPbZT6qHdptRTu02p53DUYxASg5AYhMQgJAYhMZR6bKfUQ7tNqad2m1LP4ajHICQGITEIiUFIDEJiKPXYTqmHdptST+02pZ7DUY9BSAxCYhASg5AYhMRQ6rGdUg/tNqWe2m1KPYejHoOQGITEICQGITEIiaHUYzulHtptSj2125R6Dkc9BiExCIlBSAxCYhASQ6nHdko9tNuUemq3KfUcjnoMQmIQEoOQGITEICSGUo/tlHpotyn11G5T6jkc9RiExCAkBiExCIlBSAylHtsp9dBuU+qp3abUczjqMQiJQUgMQmIQEoOQGITE+AHFISNQrFTGuwAAAABJRU5ErkJggg==",
        CLEAR_BACKGROUND_COLOR: "#2b2f38",
        NODE_TITLE_COLOR: "#b2b7bd",
        NODE_SELECTED_TITLE_COLOR: "#FFF",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#AAA",
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#2b2f38",
        NODE_DEFAULT_BGCOLOR: "#242730",
        NODE_DEFAULT_BOXCOLOR: "#6e7581",
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#FFF",
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
        DEFAULT_GROUP_FONT: 22,
        WIDGET_BGCOLOR: "#2b2f38",
        WIDGET_OUTLINE_COLOR: "#6e7581",
        WIDGET_TEXT_COLOR: "#DDD",
        WIDGET_SECONDARY_TEXT_COLOR: "#b2b7bd",
        LINK_COLOR: "#9A9",
        EVENT_LINK_COLOR: "#A86",
        CONNECTING_LINK_COLOR: "#AFA"
      },
      comfy_base: {
        "fg-color": "#fff",
        "bg-color": "#2b2f38",
        "comfy-menu-bg": "#242730",
        "comfy-input-bg": "#2b2f38",
        "input-text": "#ddd",
        "descrip-text": "#b2b7bd",
        "drag-text": "#ccc",
        "error-text": "#ff4444",
        "border-color": "#6e7581",
        "tr-even-bg-color": "#2b2f38",
        "tr-odd-bg-color": "#242730",
        "content-bg": "#6e7581",
        "content-fg": "#fff",
        "content-hover-bg": "#2b2f38",
        "content-hover-fg": "#fff"
      }
    }
  },
  nord: {
    id: "nord",
    name: "Nord",
    colors: {
      node_slot: {
        BOOLEAN: "",
        CLIP: "#eacb8b",
        CLIP_VISION: "#A8DADC",
        CLIP_VISION_OUTPUT: "#ad7452",
        CONDITIONING: "#cf876f",
        CONTROL_NET: "#00d78d",
        CONTROL_NET_WEIGHTS: "",
        FLOAT: "",
        GLIGEN: "",
        IMAGE: "#80a1c0",
        IMAGEUPLOAD: "",
        INT: "",
        LATENT: "#b38ead",
        LATENT_KEYFRAME: "",
        MASK: "#a3bd8d",
        MODEL: "#8978a7",
        SAMPLER: "",
        SIGMAS: "",
        STRING: "",
        STYLE_MODEL: "#C2FFAE",
        T2I_ADAPTER_WEIGHTS: "",
        TAESD: "#DCC274",
        TIMESTEP_KEYFRAME: "",
        UPSCALE_MODEL: "",
        VAE: "#be616b"
      },
      litegraph_base: {
        BACKGROUND_IMAGE: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAFu2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOjUwNDFhMmZjLTEzNzQtMTk0ZC1hZWY4LTYxMzM1MTVmNjUwMCIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDoyMzFiMTBiMC1iNGZiLTAyNGUtYjEyZS0zMDUzMDNjZDA3YzgiIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDoyMzFiMTBiMC1iNGZiLTAyNGUtYjEyZS0zMDUzMDNjZDA3YzgiPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJjcmVhdGVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDo1MDQxYTJmYy0xMzc0LTE5NGQtYWVmOC02MTMzNTE1ZjY1MDAiIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz73jWg/AAAAyUlEQVR42u3WKwoAIBRFQRdiMb1idv9Lsxn9gEFw4Dbb8JCTojbbXEJwjJVL2HKwYMGCBQuWLbDmjr+9zrBGjHl1WVcvy2DBggULFizTWQpewSt4HzwsgwULFiwFr7MUvMtS8D54WLBgGSxYCl7BK3iXZbBgwYIFC5bpLAWv4BW8Dx6WwYIFC5aC11kK3mUpeB88LFiwDBYsBa/gFbzLMliwYMGCBct0loJX8AreBw/LYMGCBUvB6ywF77IUvA8eFixYBgsWrNfWAZPltufdad+1AAAAAElFTkSuQmCC",
        CLEAR_BACKGROUND_COLOR: "#212732",
        NODE_TITLE_COLOR: "#999",
        NODE_SELECTED_TITLE_COLOR: "#e5eaf0",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#bcc2c8",
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#2e3440",
        NODE_DEFAULT_BGCOLOR: "#161b22",
        NODE_DEFAULT_BOXCOLOR: "#545d70",
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#e5eaf0",
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: "#2e3440",
        WIDGET_OUTLINE_COLOR: "#545d70",
        WIDGET_TEXT_COLOR: "#bcc2c8",
        WIDGET_SECONDARY_TEXT_COLOR: "#999",
        LINK_COLOR: "#9A9",
        EVENT_LINK_COLOR: "#A86",
        CONNECTING_LINK_COLOR: "#AFA"
      },
      comfy_base: {
        "fg-color": "#e5eaf0",
        "bg-color": "#2e3440",
        "comfy-menu-bg": "#161b22",
        "comfy-input-bg": "#2e3440",
        "input-text": "#bcc2c8",
        "descrip-text": "#999",
        "drag-text": "#ccc",
        "error-text": "#ff4444",
        "border-color": "#545d70",
        "tr-even-bg-color": "#2e3440",
        "tr-odd-bg-color": "#161b22",
        "content-bg": "#545d70",
        "content-fg": "#e5eaf0",
        "content-hover-bg": "#2e3440",
        "content-hover-fg": "#e5eaf0"
      }
    }
  },
  github: {
    id: "github",
    name: "Github",
    colors: {
      node_slot: {
        BOOLEAN: "",
        CLIP: "#eacb8b",
        CLIP_VISION: "#A8DADC",
        CLIP_VISION_OUTPUT: "#ad7452",
        CONDITIONING: "#cf876f",
        CONTROL_NET: "#00d78d",
        CONTROL_NET_WEIGHTS: "",
        FLOAT: "",
        GLIGEN: "",
        IMAGE: "#80a1c0",
        IMAGEUPLOAD: "",
        INT: "",
        LATENT: "#b38ead",
        LATENT_KEYFRAME: "",
        MASK: "#a3bd8d",
        MODEL: "#8978a7",
        SAMPLER: "",
        SIGMAS: "",
        STRING: "",
        STYLE_MODEL: "#C2FFAE",
        T2I_ADAPTER_WEIGHTS: "",
        TAESD: "#DCC274",
        TIMESTEP_KEYFRAME: "",
        UPSCALE_MODEL: "",
        VAE: "#be616b"
      },
      litegraph_base: {
        BACKGROUND_IMAGE: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGlmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOmIyYzRhNjA5LWJmYTctYTg0MC1iOGFlLTk3MzE2ZjM1ZGIyNyIgeG1wTU06RG9jdW1lbnRJRD0iYWRvYmU6ZG9jaWQ6cGhvdG9zaG9wOjk0ZmNlZGU4LTE1MTctZmQ0MC04ZGU3LWYzOTgxM2E3ODk5ZiIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6MjMxYjEwYjAtYjRmYi0wMjRlLWIxMmUtMzA1MzAzY2QwN2M4IiBzdEV2dDp3aGVuPSIyMDIzLTExLTEzVDAwOjE4OjAyKzAxOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjUuMSAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjQ4OWY1NzlmLTJkNjUtZWQ0Zi04OTg0LTA4NGE2MGE1ZTMzNSIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xNVQwMjowNDo1OSswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDpiMmM0YTYwOS1iZmE3LWE4NDAtYjhhZS05NzMxNmYzNWRiMjciIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz4OTe6GAAAAx0lEQVR42u3WMQoAIQxFwRzJys77X8vSLiRgITif7bYbgrwYc/mKXyBoY4VVBgsWLFiwYFmOlTv+9jfDOjHmr8u6eVkGCxYsWLBgmc5S8ApewXvgYRksWLBgKXidpeBdloL3wMOCBctgwVLwCl7BuyyDBQsWLFiwTGcpeAWv4D3wsAwWLFiwFLzOUvAuS8F74GHBgmWwYCl4Ba/gXZbBggULFixYprMUvIJX8B54WAYLFixYCl5nKXiXpeA98LBgwTJYsGC9tg1o8f4TTtqzNQAAAABJRU5ErkJggg==",
        CLEAR_BACKGROUND_COLOR: "#040506",
        NODE_TITLE_COLOR: "#999",
        NODE_SELECTED_TITLE_COLOR: "#e5eaf0",
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: "#bcc2c8",
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: "#161b22",
        NODE_DEFAULT_BGCOLOR: "#13171d",
        NODE_DEFAULT_BOXCOLOR: "#30363d",
        NODE_DEFAULT_SHAPE: "box",
        NODE_BOX_OUTLINE_COLOR: "#e5eaf0",
        DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: "#161b22",
        WIDGET_OUTLINE_COLOR: "#30363d",
        WIDGET_TEXT_COLOR: "#bcc2c8",
        WIDGET_SECONDARY_TEXT_COLOR: "#999",
        LINK_COLOR: "#9A9",
        EVENT_LINK_COLOR: "#A86",
        CONNECTING_LINK_COLOR: "#AFA"
      },
      comfy_base: {
        "fg-color": "#e5eaf0",
        "bg-color": "#161b22",
        "comfy-menu-bg": "#13171d",
        "comfy-input-bg": "#161b22",
        "input-text": "#bcc2c8",
        "descrip-text": "#999",
        "drag-text": "#ccc",
        "error-text": "#ff4444",
        "border-color": "#30363d",
        "tr-even-bg-color": "#161b22",
        "tr-odd-bg-color": "#13171d",
        "content-bg": "#30363d",
        "content-fg": "#e5eaf0",
        "content-hover-bg": "#161b22",
        "content-hover-fg": "#e5eaf0"
      }
    }
  }
};
const id$4 = "Comfy.ColorPalette";
const idCustomColorPalettes = "Comfy.CustomColorPalettes";
const defaultColorPaletteId = "dark";
const els = {
  select: null
};
app.registerExtension({
  name: id$4,
  init() {
    LGraphCanvas.prototype.updateBackground = function(image, clearBackgroundColor) {
      this._bg_img = new Image();
      this._bg_img.name = image;
      this._bg_img.src = image;
      this._bg_img.onload = () => {
        this.draw(true, true);
      };
      this.background_image = image;
      this.clear_background = true;
      this.clear_background_color = clearBackgroundColor;
      this._pattern = null;
    };
  },
  addCustomNodeDefs(node_defs) {
    const sortObjectKeys = /* @__PURE__ */ __name((unordered) => {
      return Object.keys(unordered).sort().reduce((obj, key) => {
        obj[key] = unordered[key];
        return obj;
      }, {});
    }, "sortObjectKeys");
    function getSlotTypes() {
      var types = [];
      const defs = node_defs;
      for (const nodeId in defs) {
        const nodeData = defs[nodeId];
        var inputs = nodeData["input"]["required"];
        if (nodeData["input"]["optional"] !== void 0) {
          inputs = Object.assign(
            {},
            nodeData["input"]["required"],
            nodeData["input"]["optional"]
          );
        }
        for (const inputName in inputs) {
          const inputData = inputs[inputName];
          const type = inputData[0];
          if (!Array.isArray(type)) {
            types.push(type);
          }
        }
        for (const o in nodeData["output"]) {
          const output = nodeData["output"][o];
          types.push(output);
        }
      }
      return types;
    }
    __name(getSlotTypes, "getSlotTypes");
    function completeColorPalette(colorPalette) {
      var types = getSlotTypes();
      for (const type of types) {
        if (!colorPalette.colors.node_slot[type]) {
          colorPalette.colors.node_slot[type] = "";
        }
      }
      colorPalette.colors.node_slot = sortObjectKeys(
        colorPalette.colors.node_slot
      );
      return colorPalette;
    }
    __name(completeColorPalette, "completeColorPalette");
    const getColorPaletteTemplate = /* @__PURE__ */ __name(async () => {
      let colorPalette = {
        id: "my_color_palette_unique_id",
        name: "My Color Palette",
        colors: {
          node_slot: {},
          litegraph_base: {},
          comfy_base: {}
        }
      };
      const defaultColorPalette = colorPalettes[defaultColorPaletteId];
      for (const key in defaultColorPalette.colors.litegraph_base) {
        if (!colorPalette.colors.litegraph_base[key]) {
          colorPalette.colors.litegraph_base[key] = "";
        }
      }
      for (const key in defaultColorPalette.colors.comfy_base) {
        if (!colorPalette.colors.comfy_base[key]) {
          colorPalette.colors.comfy_base[key] = "";
        }
      }
      return completeColorPalette(colorPalette);
    }, "getColorPaletteTemplate");
    const getCustomColorPalettes = /* @__PURE__ */ __name(() => {
      return app.ui.settings.getSettingValue(idCustomColorPalettes, {});
    }, "getCustomColorPalettes");
    const setCustomColorPalettes = /* @__PURE__ */ __name((customColorPalettes) => {
      return app.ui.settings.setSettingValue(
        idCustomColorPalettes,
        customColorPalettes
      );
    }, "setCustomColorPalettes");
    const addCustomColorPalette = /* @__PURE__ */ __name(async (colorPalette) => {
      if (typeof colorPalette !== "object") {
        alert("Invalid color palette.");
        return;
      }
      if (!colorPalette.id) {
        alert("Color palette missing id.");
        return;
      }
      if (!colorPalette.name) {
        alert("Color palette missing name.");
        return;
      }
      if (!colorPalette.colors) {
        alert("Color palette missing colors.");
        return;
      }
      if (colorPalette.colors.node_slot && typeof colorPalette.colors.node_slot !== "object") {
        alert("Invalid color palette colors.node_slot.");
        return;
      }
      const customColorPalettes = getCustomColorPalettes();
      customColorPalettes[colorPalette.id] = colorPalette;
      setCustomColorPalettes(customColorPalettes);
      for (const option of els.select.childNodes) {
        if (option.value === "custom_" + colorPalette.id) {
          els.select.removeChild(option);
        }
      }
      els.select.append(
        $el("option", {
          textContent: colorPalette.name + " (custom)",
          value: "custom_" + colorPalette.id,
          selected: true
        })
      );
      setColorPalette("custom_" + colorPalette.id);
      await loadColorPalette(colorPalette);
    }, "addCustomColorPalette");
    const deleteCustomColorPalette = /* @__PURE__ */ __name(async (colorPaletteId) => {
      const customColorPalettes = getCustomColorPalettes();
      delete customColorPalettes[colorPaletteId];
      setCustomColorPalettes(customColorPalettes);
      for (const opt of els.select.childNodes) {
        const option = opt;
        if (option.value === defaultColorPaletteId) {
          option.selected = true;
        }
        if (option.value === "custom_" + colorPaletteId) {
          els.select.removeChild(option);
        }
      }
      setColorPalette(defaultColorPaletteId);
      await loadColorPalette(getColorPalette());
    }, "deleteCustomColorPalette");
    const loadColorPalette = /* @__PURE__ */ __name(async (colorPalette) => {
      colorPalette = await completeColorPalette(colorPalette);
      if (colorPalette.colors) {
        if (colorPalette.colors.node_slot) {
          Object.assign(
            // @ts-expect-error
            app.canvas.default_connection_color_byType,
            colorPalette.colors.node_slot
          );
          Object.assign(
            LGraphCanvas.link_type_colors,
            colorPalette.colors.node_slot
          );
        }
        if (colorPalette.colors.litegraph_base) {
          app.canvas.node_title_color = colorPalette.colors.litegraph_base.NODE_TITLE_COLOR;
          app.canvas.default_link_color = colorPalette.colors.litegraph_base.LINK_COLOR;
          for (const key in colorPalette.colors.litegraph_base) {
            if (colorPalette.colors.litegraph_base.hasOwnProperty(key) && LiteGraph.hasOwnProperty(key)) {
              LiteGraph[key] = colorPalette.colors.litegraph_base[key];
            }
          }
        }
        if (colorPalette.colors.comfy_base) {
          const rootStyle = document.documentElement.style;
          for (const key in colorPalette.colors.comfy_base) {
            rootStyle.setProperty(
              "--" + key,
              colorPalette.colors.comfy_base[key]
            );
          }
        }
        app.canvas.draw(true, true);
      }
    }, "loadColorPalette");
    const getColorPalette = /* @__PURE__ */ __name((colorPaletteId) => {
      if (!colorPaletteId) {
        colorPaletteId = app.ui.settings.getSettingValue(
          id$4,
          defaultColorPaletteId
        );
      }
      if (colorPaletteId.startsWith("custom_")) {
        colorPaletteId = colorPaletteId.substr(7);
        let customColorPalettes = getCustomColorPalettes();
        if (customColorPalettes[colorPaletteId]) {
          return customColorPalettes[colorPaletteId];
        }
      }
      return colorPalettes[colorPaletteId];
    }, "getColorPalette");
    const setColorPalette = /* @__PURE__ */ __name((colorPaletteId) => {
      app.ui.settings.setSettingValue(id$4, colorPaletteId);
    }, "setColorPalette");
    const fileInput = $el("input", {
      type: "file",
      accept: ".json",
      style: { display: "none" },
      parent: document.body,
      onchange: /* @__PURE__ */ __name(() => {
        const file2 = fileInput.files[0];
        if (file2.type === "application/json" || file2.name.endsWith(".json")) {
          const reader = new FileReader();
          reader.onload = async () => {
            await addCustomColorPalette(JSON.parse(reader.result));
          };
          reader.readAsText(file2);
        }
      }, "onchange")
    });
    app.ui.settings.addSetting({
      id: id$4,
      category: ["Comfy", "ColorPalette"],
      name: "Color Palette",
      type: /* @__PURE__ */ __name((name, setter, value) => {
        const options = [
          ...Object.values(colorPalettes).map(
            (c) => $el("option", {
              textContent: c.name,
              value: c.id,
              selected: c.id === value
            })
          ),
          ...Object.values(getCustomColorPalettes()).map(
            (c) => $el("option", {
              textContent: `${c.name} (custom)`,
              value: `custom_${c.id}`,
              selected: `custom_${c.id}` === value
            })
          )
        ];
        els.select = $el(
          "select",
          {
            style: {
              marginBottom: "0.15rem",
              width: "100%"
            },
            onchange: /* @__PURE__ */ __name((e) => {
              setter(e.target.value);
            }, "onchange")
          },
          options
        );
        return $el("tr", [
          $el("td", [
            els.select,
            $el(
              "div",
              {
                style: {
                  display: "grid",
                  gap: "4px",
                  gridAutoFlow: "column"
                }
              },
              [
                $el("input", {
                  type: "button",
                  value: "Export",
                  onclick: /* @__PURE__ */ __name(async () => {
                    const colorPaletteId = app.ui.settings.getSettingValue(
                      id$4,
                      defaultColorPaletteId
                    );
                    const colorPalette = await completeColorPalette(
                      getColorPalette(colorPaletteId)
                    );
                    const json = JSON.stringify(colorPalette, null, 2);
                    const blob = new Blob([json], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = $el("a", {
                      href: url,
                      download: colorPaletteId + ".json",
                      style: { display: "none" },
                      parent: document.body
                    });
                    a.click();
                    setTimeout(function() {
                      a.remove();
                      window.URL.revokeObjectURL(url);
                    }, 0);
                  }, "onclick")
                }),
                $el("input", {
                  type: "button",
                  value: "Import",
                  onclick: /* @__PURE__ */ __name(() => {
                    fileInput.click();
                  }, "onclick")
                }),
                $el("input", {
                  type: "button",
                  value: "Template",
                  onclick: /* @__PURE__ */ __name(async () => {
                    const colorPalette = await getColorPaletteTemplate();
                    const json = JSON.stringify(colorPalette, null, 2);
                    const blob = new Blob([json], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = $el("a", {
                      href: url,
                      download: "color_palette.json",
                      style: { display: "none" },
                      parent: document.body
                    });
                    a.click();
                    setTimeout(function() {
                      a.remove();
                      window.URL.revokeObjectURL(url);
                    }, 0);
                  }, "onclick")
                }),
                $el("input", {
                  type: "button",
                  value: "Delete",
                  onclick: /* @__PURE__ */ __name(async () => {
                    let colorPaletteId = app.ui.settings.getSettingValue(
                      id$4,
                      defaultColorPaletteId
                    );
                    if (colorPalettes[colorPaletteId]) {
                      alert("You cannot delete a built-in color palette.");
                      return;
                    }
                    if (colorPaletteId.startsWith("custom_")) {
                      colorPaletteId = colorPaletteId.substr(7);
                    }
                    await deleteCustomColorPalette(colorPaletteId);
                  }, "onclick")
                })
              ]
            )
          ])
        ]);
      }, "type"),
      defaultValue: defaultColorPaletteId,
      async onChange(value) {
        if (!value) {
          return;
        }
        let palette = colorPalettes[value];
        if (palette) {
          await loadColorPalette(palette);
        } else if (value.startsWith("custom_")) {
          value = value.substr(7);
          let customColorPalettes = getCustomColorPalettes();
          if (customColorPalettes[value]) {
            palette = customColorPalettes[value];
            await loadColorPalette(customColorPalettes[value]);
          }
        }
        let { BACKGROUND_IMAGE, CLEAR_BACKGROUND_COLOR } = palette.colors.litegraph_base;
        if (BACKGROUND_IMAGE === void 0 || CLEAR_BACKGROUND_COLOR === void 0) {
          const base = colorPalettes["dark"].colors.litegraph_base;
          BACKGROUND_IMAGE = base.BACKGROUND_IMAGE;
          CLEAR_BACKGROUND_COLOR = base.CLEAR_BACKGROUND_COLOR;
        }
        app.canvas.updateBackground(BACKGROUND_IMAGE, CLEAR_BACKGROUND_COLOR);
      }
    });
  }
});
const ext$2 = {
  name: "Comfy.ContextMenuFilter",
  init() {
    const ctxMenu = LiteGraph.ContextMenu;
    LiteGraph.ContextMenu = function(values, options) {
      const ctx = new ctxMenu(values, options);
      if (options?.className === "dark" && values?.length > 4) {
        const filter = document.createElement("input");
        filter.classList.add("comfy-context-menu-filter");
        filter.placeholder = "Filter list";
        ctx.root.prepend(filter);
        const items = Array.from(
          ctx.root.querySelectorAll(".litemenu-entry")
        );
        let displayedItems = [...items];
        let itemCount = displayedItems.length;
        requestAnimationFrame(() => {
          const currentNode = LGraphCanvas.active_canvas.current_node;
          const clickedComboValue = currentNode.widgets?.filter(
            (w) => w.type === "combo" && w.options.values.length === values.length
          ).find(
            (w) => w.options.values.every((v, i) => v === values[i])
          )?.value;
          let selectedIndex = clickedComboValue ? values.findIndex((v) => v === clickedComboValue) : 0;
          if (selectedIndex < 0) {
            selectedIndex = 0;
          }
          let selectedItem = displayedItems[selectedIndex];
          updateSelected();
          function updateSelected() {
            selectedItem?.style.setProperty("background-color", "");
            selectedItem?.style.setProperty("color", "");
            selectedItem = displayedItems[selectedIndex];
            selectedItem?.style.setProperty(
              "background-color",
              "#ccc",
              "important"
            );
            selectedItem?.style.setProperty("color", "#000", "important");
          }
          __name(updateSelected, "updateSelected");
          const positionList = /* @__PURE__ */ __name(() => {
            const rect = ctx.root.getBoundingClientRect();
            if (rect.top < 0) {
              const scale = 1 - ctx.root.getBoundingClientRect().height / ctx.root.clientHeight;
              const shift = ctx.root.clientHeight * scale / 2;
              ctx.root.style.top = -shift + "px";
            }
          }, "positionList");
          filter.addEventListener("keydown", (event) => {
            switch (event.key) {
              case "ArrowUp":
                event.preventDefault();
                if (selectedIndex === 0) {
                  selectedIndex = itemCount - 1;
                } else {
                  selectedIndex--;
                }
                updateSelected();
                break;
              case "ArrowRight":
                event.preventDefault();
                selectedIndex = itemCount - 1;
                updateSelected();
                break;
              case "ArrowDown":
                event.preventDefault();
                if (selectedIndex === itemCount - 1) {
                  selectedIndex = 0;
                } else {
                  selectedIndex++;
                }
                updateSelected();
                break;
              case "ArrowLeft":
                event.preventDefault();
                selectedIndex = 0;
                updateSelected();
                break;
              case "Enter":
                selectedItem?.click();
                break;
              case "Escape":
                ctx.close();
                break;
            }
          });
          filter.addEventListener("input", () => {
            const term = filter.value.toLocaleLowerCase();
            displayedItems = items.filter((item) => {
              const isVisible = !term || item.textContent.toLocaleLowerCase().includes(term);
              item.style.display = isVisible ? "block" : "none";
              return isVisible;
            });
            selectedIndex = 0;
            if (displayedItems.includes(selectedItem)) {
              selectedIndex = displayedItems.findIndex(
                (d) => d === selectedItem
              );
            }
            itemCount = displayedItems.length;
            updateSelected();
            if (options.event) {
              let top = options.event.clientY - 10;
              const bodyRect = document.body.getBoundingClientRect();
              const rootRect = ctx.root.getBoundingClientRect();
              if (bodyRect.height && top > bodyRect.height - rootRect.height - 10) {
                top = Math.max(0, bodyRect.height - rootRect.height - 10);
              }
              ctx.root.style.top = top + "px";
              positionList();
            }
          });
          requestAnimationFrame(() => {
            filter.focus();
            positionList();
          });
        });
      }
      return ctx;
    };
    LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
  }
};
app.registerExtension(ext$2);
function stripComments(str) {
  return str.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g, "");
}
__name(stripComments, "stripComments");
app.registerExtension({
  name: "Comfy.DynamicPrompts",
  nodeCreated(node) {
    if (node.widgets) {
      const widgets = node.widgets.filter((n) => n.dynamicPrompts);
      for (const widget of widgets) {
        widget.serializeValue = (workflowNode, widgetIndex) => {
          let prompt2 = stripComments(widget.value);
          while (prompt2.replace("\\{", "").includes("{") && prompt2.replace("\\}", "").includes("}")) {
            const startIndex = prompt2.replace("\\{", "00").indexOf("{");
            const endIndex = prompt2.replace("\\}", "00").indexOf("}");
            const optionsString = prompt2.substring(startIndex + 1, endIndex);
            const options = optionsString.split("|");
            const randomIndex = Math.floor(Math.random() * options.length);
            const randomOption = options[randomIndex];
            prompt2 = prompt2.substring(0, startIndex) + randomOption + prompt2.substring(endIndex + 1);
          }
          if (workflowNode?.widgets_values)
            workflowNode.widgets_values[widgetIndex] = prompt2;
          return prompt2;
        };
      }
    }
  }
});
app.registerExtension({
  name: "Comfy.EditAttention",
  init() {
    const editAttentionDelta = app.ui.settings.addSetting({
      id: "Comfy.EditAttention.Delta",
      name: "Ctrl+up/down precision",
      type: "slider",
      attrs: {
        min: 0.01,
        max: 0.5,
        step: 0.01
      },
      defaultValue: 0.05
    });
    function incrementWeight(weight, delta) {
      const floatWeight = parseFloat(weight);
      if (isNaN(floatWeight)) return weight;
      const newWeight = floatWeight + delta;
      return String(Number(newWeight.toFixed(10)));
    }
    __name(incrementWeight, "incrementWeight");
    function findNearestEnclosure(text, cursorPos) {
      let start = cursorPos, end = cursorPos;
      let openCount = 0, closeCount = 0;
      while (start >= 0) {
        start--;
        if (text[start] === "(" && openCount === closeCount) break;
        if (text[start] === "(") openCount++;
        if (text[start] === ")") closeCount++;
      }
      if (start < 0) return false;
      openCount = 0;
      closeCount = 0;
      while (end < text.length) {
        if (text[end] === ")" && openCount === closeCount) break;
        if (text[end] === "(") openCount++;
        if (text[end] === ")") closeCount++;
        end++;
      }
      if (end === text.length) return false;
      return { start: start + 1, end };
    }
    __name(findNearestEnclosure, "findNearestEnclosure");
    function addWeightToParentheses(text) {
      const parenRegex = /^\((.*)\)$/;
      const parenMatch = text.match(parenRegex);
      const floatRegex = /:([+-]?(\d*\.)?\d+([eE][+-]?\d+)?)/;
      const floatMatch = text.match(floatRegex);
      if (parenMatch && !floatMatch) {
        return `(${parenMatch[1]}:1.0)`;
      } else {
        return text;
      }
    }
    __name(addWeightToParentheses, "addWeightToParentheses");
    function editAttention(event) {
      const inputField = event.composedPath()[0];
      const delta = parseFloat(editAttentionDelta.value);
      if (inputField.tagName !== "TEXTAREA") return;
      if (!(event.key === "ArrowUp" || event.key === "ArrowDown")) return;
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      let start = inputField.selectionStart;
      let end = inputField.selectionEnd;
      let selectedText = inputField.value.substring(start, end);
      if (!selectedText) {
        const nearestEnclosure = findNearestEnclosure(inputField.value, start);
        if (nearestEnclosure) {
          start = nearestEnclosure.start;
          end = nearestEnclosure.end;
          selectedText = inputField.value.substring(start, end);
        } else {
          const delimiters = " .,\\/!?%^*;:{}=-_`~()\r\n	";
          while (!delimiters.includes(inputField.value[start - 1]) && start > 0) {
            start--;
          }
          while (!delimiters.includes(inputField.value[end]) && end < inputField.value.length) {
            end++;
          }
          selectedText = inputField.value.substring(start, end);
          if (!selectedText) return;
        }
      }
      if (selectedText[selectedText.length - 1] === " ") {
        selectedText = selectedText.substring(0, selectedText.length - 1);
        end -= 1;
      }
      if (inputField.value[start - 1] === "(" && inputField.value[end] === ")") {
        start -= 1;
        end += 1;
        selectedText = inputField.value.substring(start, end);
      }
      if (selectedText[0] !== "(" || selectedText[selectedText.length - 1] !== ")") {
        selectedText = `(${selectedText})`;
      }
      selectedText = addWeightToParentheses(selectedText);
      const weightDelta = event.key === "ArrowUp" ? delta : -delta;
      const updatedText = selectedText.replace(
        /\((.*):([+-]?\d+(?:\.\d+)?)\)/,
        (match, text, weight) => {
          weight = incrementWeight(weight, weightDelta);
          if (weight == 1) {
            return text;
          } else {
            return `(${text}:${weight})`;
          }
        }
      );
      inputField.setSelectionRange(start, end);
      document.execCommand("insertText", false, updatedText);
      inputField.setSelectionRange(start, start + updatedText.length);
    }
    __name(editAttention, "editAttention");
    window.addEventListener("keydown", editAttention);
  }
});
const CONVERTED_TYPE = "converted-widget";
const VALID_TYPES = ["STRING", "combo", "number", "toggle", "BOOLEAN"];
const CONFIG = Symbol();
const GET_CONFIG = Symbol();
const TARGET = Symbol();
const replacePropertyName = "Run widget replace on values";
class PrimitiveNode extends LGraphNode {
  static {
    __name(this, "PrimitiveNode");
  }
  controlValues;
  lastType;
  static category;
  constructor(title) {
    super(title);
    this.addOutput("connect to widget input", "*");
    this.serialize_widgets = true;
    this.isVirtualNode = true;
    if (!this.properties || !(replacePropertyName in this.properties)) {
      this.addProperty(replacePropertyName, false, "boolean");
    }
  }
  applyToGraph(extraLinks = []) {
    if (!this.outputs[0].links?.length) return;
    function get_links(node) {
      let links2 = [];
      for (const l of node.outputs[0].links) {
        const linkInfo = app.graph.links[l];
        const n = node.graph.getNodeById(linkInfo.target_id);
        if (n.type == "Reroute") {
          links2 = links2.concat(get_links(n));
        } else {
          links2.push(l);
        }
      }
      return links2;
    }
    __name(get_links, "get_links");
    let links = [
      ...get_links(this).map((l) => app.graph.links[l]),
      ...extraLinks
    ];
    let v = this.widgets?.[0].value;
    if (v && this.properties[replacePropertyName]) {
      v = applyTextReplacements(app, v);
    }
    for (const linkInfo of links) {
      const node = this.graph.getNodeById(linkInfo.target_id);
      const input = node.inputs[linkInfo.target_slot];
      let widget;
      if (input.widget[TARGET]) {
        widget = input.widget[TARGET];
      } else {
        const widgetName = input.widget.name;
        if (widgetName) {
          widget = node.widgets.find((w) => w.name === widgetName);
        }
      }
      if (widget) {
        widget.value = v;
        if (widget.callback) {
          widget.callback(
            widget.value,
            app.canvas,
            node,
            app.canvas.graph_mouse,
            {}
          );
        }
      }
    }
  }
  refreshComboInNode() {
    const widget = this.widgets?.[0];
    if (widget?.type === "combo") {
      widget.options.values = this.outputs[0].widget[GET_CONFIG]()[0];
      if (!widget.options.values.includes(widget.value)) {
        widget.value = widget.options.values[0];
        widget.callback(widget.value);
      }
    }
  }
  onAfterGraphConfigured() {
    if (this.outputs[0].links?.length && !this.widgets?.length) {
      if (!this.#onFirstConnection()) return;
      if (this.widgets) {
        for (let i = 0; i < this.widgets_values.length; i++) {
          const w = this.widgets[i];
          if (w) {
            w.value = this.widgets_values[i];
          }
        }
      }
      this.#mergeWidgetConfig();
    }
  }
  onConnectionsChange(_, index, connected) {
    if (app.configuringGraph) {
      return;
    }
    const links = this.outputs[0].links;
    if (connected) {
      if (links?.length && !this.widgets?.length) {
        this.#onFirstConnection();
      }
    } else {
      this.#mergeWidgetConfig();
      if (!links?.length) {
        this.onLastDisconnect();
      }
    }
  }
  onConnectOutput(slot, type, input, target_node, target_slot) {
    if (!input.widget) {
      if (!(input.type in ComfyWidgets)) return false;
    }
    if (this.outputs[slot].links?.length) {
      const valid = this.#isValidConnection(input);
      if (valid) {
        this.applyToGraph([{ target_id: target_node.id, target_slot }]);
      }
      return valid;
    }
  }
  #onFirstConnection(recreating) {
    if (!this.outputs[0].links) {
      this.onLastDisconnect();
      return;
    }
    const linkId = this.outputs[0].links[0];
    const link = this.graph.links[linkId];
    if (!link) return;
    const theirNode = this.graph.getNodeById(link.target_id);
    if (!theirNode || !theirNode.inputs) return;
    const input = theirNode.inputs[link.target_slot];
    if (!input) return;
    let widget;
    if (!input.widget) {
      if (!(input.type in ComfyWidgets)) return;
      widget = { name: input.name, [GET_CONFIG]: () => [input.type, {}] };
    } else {
      widget = input.widget;
    }
    const config = widget[GET_CONFIG]?.();
    if (!config) return;
    const { type } = getWidgetType(config);
    this.outputs[0].type = type;
    this.outputs[0].name = type;
    this.outputs[0].widget = widget;
    this.#createWidget(
      widget[CONFIG] ?? config,
      theirNode,
      widget.name,
      recreating,
      widget[TARGET]
    );
  }
  #createWidget(inputData, node, widgetName, recreating, targetWidget) {
    let type = inputData[0];
    if (type instanceof Array) {
      type = "COMBO";
    }
    const size = this.size;
    let widget;
    if (type in ComfyWidgets) {
      widget = (ComfyWidgets[type](this, "value", inputData, app) || {}).widget;
    } else {
      widget = this.addWidget(type, "value", null, () => {
      }, {});
    }
    if (targetWidget) {
      widget.value = targetWidget.value;
    } else if (node?.widgets && widget) {
      const theirWidget = node.widgets.find((w) => w.name === widgetName);
      if (theirWidget) {
        widget.value = theirWidget.value;
      }
    }
    if (!inputData?.[1]?.control_after_generate && (widget.type === "number" || widget.type === "combo")) {
      let control_value = this.widgets_values?.[1];
      if (!control_value) {
        control_value = "fixed";
      }
      addValueControlWidgets(
        this,
        widget,
        control_value,
        void 0,
        inputData
      );
      let filter = this.widgets_values?.[2];
      if (filter && this.widgets.length === 3) {
        this.widgets[2].value = filter;
      }
    }
    const controlValues = this.controlValues;
    if (this.lastType === this.widgets[0].type && controlValues?.length === this.widgets.length - 1) {
      for (let i = 0; i < controlValues.length; i++) {
        this.widgets[i + 1].value = controlValues[i];
      }
    }
    const callback = widget.callback;
    const self = this;
    widget.callback = function() {
      const r = callback ? callback.apply(this, arguments) : void 0;
      self.applyToGraph();
      return r;
    };
    this.size = [
      Math.max(this.size[0], size[0]),
      Math.max(this.size[1], size[1])
    ];
    if (!recreating) {
      const sz = this.computeSize();
      if (this.size[0] < sz[0]) {
        this.size[0] = sz[0];
      }
      if (this.size[1] < sz[1]) {
        this.size[1] = sz[1];
      }
      requestAnimationFrame(() => {
        if (this.onResize) {
          this.onResize(this.size);
        }
      });
    }
  }
  recreateWidget() {
    const values = this.widgets?.map((w) => w.value);
    this.#removeWidgets();
    this.#onFirstConnection(true);
    if (values?.length) {
      for (let i = 0; i < this.widgets?.length; i++)
        this.widgets[i].value = values[i];
    }
    return this.widgets?.[0];
  }
  #mergeWidgetConfig() {
    const output = this.outputs[0];
    const links = output.links;
    const hasConfig = !!output.widget[CONFIG];
    if (hasConfig) {
      delete output.widget[CONFIG];
    }
    if (links?.length < 2 && hasConfig) {
      if (links.length) {
        this.recreateWidget();
      }
      return;
    }
    const config1 = output.widget[GET_CONFIG]();
    const isNumber = config1[0] === "INT" || config1[0] === "FLOAT";
    if (!isNumber) return;
    for (const linkId of links) {
      const link = app.graph.links[linkId];
      if (!link) continue;
      const theirNode = app.graph.getNodeById(link.target_id);
      const theirInput = theirNode.inputs[link.target_slot];
      this.#isValidConnection(theirInput, hasConfig);
    }
  }
  #isValidConnection(input, forceUpdate) {
    const output = this.outputs[0];
    const config2 = input.widget[GET_CONFIG]();
    return !!mergeIfValid.call(
      this,
      output,
      config2,
      forceUpdate,
      this.recreateWidget
    );
  }
  #removeWidgets() {
    if (this.widgets) {
      for (const w of this.widgets) {
        if (w.onRemove) {
          w.onRemove();
        }
      }
      this.controlValues = [];
      this.lastType = this.widgets[0]?.type;
      for (let i = 1; i < this.widgets.length; i++) {
        this.controlValues.push(this.widgets[i].value);
      }
      setTimeout(() => {
        delete this.lastType;
        delete this.controlValues;
      }, 15);
      this.widgets.length = 0;
    }
  }
  onLastDisconnect() {
    this.outputs[0].type = "*";
    this.outputs[0].name = "connect to widget input";
    delete this.outputs[0].widget;
    this.#removeWidgets();
  }
}
function getWidgetConfig(slot) {
  return slot.widget[CONFIG] ?? slot.widget[GET_CONFIG]();
}
__name(getWidgetConfig, "getWidgetConfig");
function getConfig(widgetName) {
  const { nodeData } = this.constructor;
  return nodeData?.input?.required?.[widgetName] ?? nodeData?.input?.optional?.[widgetName];
}
__name(getConfig, "getConfig");
function isConvertibleWidget(widget, config) {
  return (VALID_TYPES.includes(widget.type) || VALID_TYPES.includes(config[0])) && !widget.options?.forceInput;
}
__name(isConvertibleWidget, "isConvertibleWidget");
function hideWidget(node, widget, suffix = "") {
  if (widget.type?.startsWith(CONVERTED_TYPE)) return;
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.origSerializeValue = widget.serializeValue;
  widget.computeSize = () => [0, -4];
  widget.type = CONVERTED_TYPE + suffix;
  widget.serializeValue = () => {
    if (!node.inputs) {
      return void 0;
    }
    let node_input = node.inputs.find((i) => i.widget?.name === widget.name);
    if (!node_input || !node_input.link) {
      return void 0;
    }
    return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
  };
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidget(node, w, ":" + widget.name);
    }
  }
}
__name(hideWidget, "hideWidget");
function showWidget(widget) {
  widget.type = widget.origType;
  widget.computeSize = widget.origComputeSize;
  widget.serializeValue = widget.origSerializeValue;
  delete widget.origType;
  delete widget.origComputeSize;
  delete widget.origSerializeValue;
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      showWidget(w);
    }
  }
}
__name(showWidget, "showWidget");
function convertToInput(node, widget, config) {
  hideWidget(node, widget);
  const { type } = getWidgetType(config);
  const sz = node.size;
  node.addInput(widget.name, type, {
    widget: { name: widget.name, [GET_CONFIG]: () => config }
  });
  for (const widget2 of node.widgets) {
    widget2.last_y += LiteGraph.NODE_SLOT_HEIGHT;
  }
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}
__name(convertToInput, "convertToInput");
function convertToWidget(node, widget) {
  showWidget(widget);
  const sz = node.size;
  node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name));
  for (const widget2 of node.widgets) {
    widget2.last_y -= LiteGraph.NODE_SLOT_HEIGHT;
  }
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}
__name(convertToWidget, "convertToWidget");
function getWidgetType(config) {
  let type = config[0];
  if (type instanceof Array) {
    type = "COMBO";
  }
  return { type };
}
__name(getWidgetType, "getWidgetType");
function isValidCombo(combo, obj) {
  if (!(obj instanceof Array)) {
    console.log(`connection rejected: tried to connect combo to ${obj}`);
    return false;
  }
  if (combo.length !== obj.length) {
    console.log(`connection rejected: combo lists dont match`);
    return false;
  }
  if (combo.find((v, i) => obj[i] !== v)) {
    console.log(`connection rejected: combo lists dont match`);
    return false;
  }
  return true;
}
__name(isValidCombo, "isValidCombo");
function isPrimitiveNode(node) {
  return node.type === "PrimitiveNode";
}
__name(isPrimitiveNode, "isPrimitiveNode");
function setWidgetConfig(slot, config, target) {
  if (!slot.widget) return;
  if (config) {
    slot.widget[GET_CONFIG] = () => config;
    slot.widget[TARGET] = target;
  } else {
    delete slot.widget;
  }
  if (slot.link) {
    const link = app.graph.links[slot.link];
    if (link) {
      const originNode = app.graph.getNodeById(link.origin_id);
      if (isPrimitiveNode(originNode)) {
        if (config) {
          originNode.recreateWidget();
        } else if (!app.configuringGraph) {
          originNode.disconnectOutput(0);
          originNode.onLastDisconnect();
        }
      }
    }
  }
}
__name(setWidgetConfig, "setWidgetConfig");
function mergeIfValid(output, config2, forceUpdate, recreateWidget, config1) {
  if (!config1) {
    config1 = output.widget[CONFIG] ?? output.widget[GET_CONFIG]();
  }
  if (config1[0] instanceof Array) {
    if (!isValidCombo(config1[0], config2[0])) return;
  } else if (config1[0] !== config2[0]) {
    console.log(`connection rejected: types dont match`, config1[0], config2[0]);
    return;
  }
  const keys = /* @__PURE__ */ new Set([
    ...Object.keys(config1[1] ?? {}),
    ...Object.keys(config2[1] ?? {})
  ]);
  let customConfig;
  const getCustomConfig = /* @__PURE__ */ __name(() => {
    if (!customConfig) {
      if (typeof structuredClone === "undefined") {
        customConfig = JSON.parse(JSON.stringify(config1[1] ?? {}));
      } else {
        customConfig = structuredClone(config1[1] ?? {});
      }
    }
    return customConfig;
  }, "getCustomConfig");
  const isNumber = config1[0] === "INT" || config1[0] === "FLOAT";
  for (const k of keys.values()) {
    if (k !== "default" && k !== "forceInput" && k !== "defaultInput" && k !== "control_after_generate" && k !== "multiline" && k !== "tooltip") {
      let v1 = config1[1][k];
      let v2 = config2[1]?.[k];
      if (v1 === v2 || !v1 && !v2) continue;
      if (isNumber) {
        if (k === "min") {
          const theirMax = config2[1]?.["max"];
          if (theirMax != null && v1 > theirMax) {
            console.log("connection rejected: min > max", v1, theirMax);
            return;
          }
          getCustomConfig()[k] = v1 == null ? v2 : v2 == null ? v1 : Math.max(v1, v2);
          continue;
        } else if (k === "max") {
          const theirMin = config2[1]?.["min"];
          if (theirMin != null && v1 < theirMin) {
            console.log("connection rejected: max < min", v1, theirMin);
            return;
          }
          getCustomConfig()[k] = v1 == null ? v2 : v2 == null ? v1 : Math.min(v1, v2);
          continue;
        } else if (k === "step") {
          let step;
          if (v1 == null) {
            step = v2;
          } else if (v2 == null) {
            step = v1;
          } else {
            if (v1 < v2) {
              const a = v2;
              v2 = v1;
              v1 = a;
            }
            if (v1 % v2) {
              console.log(
                "connection rejected: steps not divisible",
                "current:",
                v1,
                "new:",
                v2
              );
              return;
            }
            step = v1;
          }
          getCustomConfig()[k] = step;
          continue;
        }
      }
      console.log(`connection rejected: config ${k} values dont match`, v1, v2);
      return;
    }
  }
  if (customConfig || forceUpdate) {
    if (customConfig) {
      output.widget[CONFIG] = [config1[0], customConfig];
    }
    const widget = recreateWidget?.call(this);
    if (widget) {
      const min = widget.options.min;
      const max = widget.options.max;
      if (min != null && widget.value < min) widget.value = min;
      if (max != null && widget.value > max) widget.value = max;
      widget.callback(widget.value);
    }
  }
  return { customConfig };
}
__name(mergeIfValid, "mergeIfValid");
let useConversionSubmenusSetting;
app.registerExtension({
  name: "Comfy.WidgetInputs",
  init() {
    useConversionSubmenusSetting = app.ui.settings.addSetting({
      id: "Comfy.NodeInputConversionSubmenus",
      name: "In the node context menu, place the entries that convert between input/widget in sub-menus.",
      type: "boolean",
      defaultValue: true
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app2) {
    const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.convertWidgetToInput = function(widget) {
      const config = getConfig.call(this, widget.name) ?? [
        widget.type,
        widget.options || {}
      ];
      if (!isConvertibleWidget(widget, config)) return false;
      convertToInput(this, widget, config);
      return true;
    };
    nodeType.prototype.getExtraMenuOptions = function(_, options) {
      const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : void 0;
      if (this.widgets) {
        let toInput = [];
        let toWidget = [];
        for (const w of this.widgets) {
          if (w.options?.forceInput) {
            continue;
          }
          if (w.type === CONVERTED_TYPE) {
            toWidget.push({
              content: `Convert ${w.name} to widget`,
              callback: /* @__PURE__ */ __name(() => convertToWidget(this, w), "callback")
            });
          } else {
            const config = getConfig.call(this, w.name) ?? [
              w.type,
              w.options || {}
            ];
            if (isConvertibleWidget(w, config)) {
              toInput.push({
                content: `Convert ${w.name} to input`,
                callback: /* @__PURE__ */ __name(() => convertToInput(this, w, config), "callback")
              });
            }
          }
        }
        if (toInput.length) {
          if (useConversionSubmenusSetting.value) {
            options.push({
              content: "Convert Widget to Input",
              submenu: {
                options: toInput
              }
            });
          } else {
            options.push(...toInput, null);
          }
        }
        if (toWidget.length) {
          if (useConversionSubmenusSetting.value) {
            options.push({
              content: "Convert Input to Widget",
              submenu: {
                options: toWidget
              }
            });
          } else {
            options.push(...toWidget, null);
          }
        }
      }
      return r;
    };
    nodeType.prototype.onGraphConfigured = function() {
      if (!this.inputs) return;
      for (const input of this.inputs) {
        if (input.widget) {
          if (!input.widget[GET_CONFIG]) {
            input.widget[GET_CONFIG] = () => getConfig.call(this, input.widget.name);
          }
          if (input.widget.config) {
            if (input.widget.config[0] instanceof Array) {
              input.type = "COMBO";
              const link = app2.graph.links[input.link];
              if (link) {
                link.type = input.type;
              }
            }
            delete input.widget.config;
          }
          const w = this.widgets.find((w2) => w2.name === input.widget.name);
          if (w) {
            hideWidget(this, w);
          } else {
            convertToWidget(this, input);
          }
        }
      }
    };
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this) : void 0;
      if (!app2.configuringGraph && this.widgets) {
        for (const w of this.widgets) {
          if (w?.options?.forceInput || w?.options?.defaultInput) {
            const config = getConfig.call(this, w.name) ?? [
              w.type,
              w.options || {}
            ];
            convertToInput(this, w, config);
          }
        }
      }
      return r;
    };
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function() {
      const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : void 0;
      if (!app2.configuringGraph && this.inputs) {
        for (const input of this.inputs) {
          if (input.widget && !input.widget[GET_CONFIG]) {
            input.widget[GET_CONFIG] = () => getConfig.call(this, input.widget.name);
            const w = this.widgets.find((w2) => w2.name === input.widget.name);
            if (w) {
              hideWidget(this, w);
            }
          }
        }
      }
      return r;
    };
    function isNodeAtPos(pos) {
      for (const n of app2.graph._nodes) {
        if (n.pos[0] === pos[0] && n.pos[1] === pos[1]) {
          return true;
        }
      }
      return false;
    }
    __name(isNodeAtPos, "isNodeAtPos");
    const origOnInputDblClick = nodeType.prototype.onInputDblClick;
    const ignoreDblClick = Symbol();
    nodeType.prototype.onInputDblClick = function(slot) {
      const r = origOnInputDblClick ? origOnInputDblClick.apply(this, arguments) : void 0;
      const input = this.inputs[slot];
      if (!input.widget || !input[ignoreDblClick]) {
        if (!(input.type in ComfyWidgets) && !(input.widget[GET_CONFIG]?.()?.[0] instanceof Array)) {
          return r;
        }
      }
      const node = LiteGraph.createNode("PrimitiveNode");
      app2.graph.add(node);
      const pos = [
        this.pos[0] - node.size[0] - 30,
        this.pos[1]
      ];
      while (isNodeAtPos(pos)) {
        pos[1] += LiteGraph.NODE_TITLE_HEIGHT;
      }
      node.pos = pos;
      node.connect(0, this, slot);
      node.title = input.name;
      input[ignoreDblClick] = true;
      setTimeout(() => {
        delete input[ignoreDblClick];
      }, 300);
      return r;
    };
    const onConnectInput = nodeType.prototype.onConnectInput;
    nodeType.prototype.onConnectInput = function(targetSlot, type, output, originNode, originSlot) {
      const v = onConnectInput?.(this, arguments);
      if (type !== "COMBO") return v;
      if (originNode.outputs[originSlot].widget) return v;
      const targetCombo = this.inputs[targetSlot].widget?.[GET_CONFIG]?.()?.[0];
      if (!targetCombo || !(targetCombo instanceof Array)) return v;
      const originConfig = originNode.constructor?.nodeData?.output?.[originSlot];
      if (!originConfig || !isValidCombo(targetCombo, originConfig)) {
        return false;
      }
      return v;
    };
  },
  registerCustomNodes() {
    LiteGraph.registerNodeType(
      "PrimitiveNode",
      Object.assign(PrimitiveNode, {
        title: "Primitive"
      })
    );
    PrimitiveNode.category = "utils";
  }
});
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.widgetInputs = window.comfyAPI.widgetInputs || {};
window.comfyAPI.widgetInputs.getWidgetConfig = getWidgetConfig;
window.comfyAPI.widgetInputs.setWidgetConfig = setWidgetConfig;
window.comfyAPI.widgetInputs.mergeIfValid = mergeIfValid;
const ORDER = Symbol();
function merge(target, source) {
  if (typeof target === "object" && typeof source === "object") {
    for (const key in source) {
      const sv = source[key];
      if (typeof sv === "object") {
        let tv = target[key];
        if (!tv) tv = target[key] = {};
        merge(tv, source[key]);
      } else {
        target[key] = sv;
      }
    }
  }
  return target;
}
__name(merge, "merge");
class ManageGroupDialog extends ComfyDialog {
  static {
    __name(this, "ManageGroupDialog");
  }
  tabs;
  selectedNodeIndex;
  selectedTab = "Inputs";
  selectedGroup;
  modifications = {};
  nodeItems;
  app;
  groupNodeType;
  groupNodeDef;
  groupData;
  innerNodesList;
  widgetsPage;
  inputsPage;
  outputsPage;
  draggable;
  get selectedNodeInnerIndex() {
    return +this.nodeItems[this.selectedNodeIndex].dataset.nodeindex;
  }
  constructor(app2) {
    super();
    this.app = app2;
    this.element = $el("dialog.comfy-group-manage", {
      parent: document.body
    });
  }
  changeTab(tab) {
    this.tabs[this.selectedTab].tab.classList.remove("active");
    this.tabs[this.selectedTab].page.classList.remove("active");
    this.tabs[tab].tab.classList.add("active");
    this.tabs[tab].page.classList.add("active");
    this.selectedTab = tab;
  }
  changeNode(index, force) {
    if (!force && this.selectedNodeIndex === index) return;
    if (this.selectedNodeIndex != null) {
      this.nodeItems[this.selectedNodeIndex].classList.remove("selected");
    }
    this.nodeItems[index].classList.add("selected");
    this.selectedNodeIndex = index;
    if (!this.buildInputsPage() && this.selectedTab === "Inputs") {
      this.changeTab("Widgets");
    }
    if (!this.buildWidgetsPage() && this.selectedTab === "Widgets") {
      this.changeTab("Outputs");
    }
    if (!this.buildOutputsPage() && this.selectedTab === "Outputs") {
      this.changeTab("Inputs");
    }
    this.changeTab(this.selectedTab);
  }
  getGroupData() {
    this.groupNodeType = LiteGraph.registered_node_types["workflow/" + this.selectedGroup];
    this.groupNodeDef = this.groupNodeType.nodeData;
    this.groupData = GroupNodeHandler.getGroupData(this.groupNodeType);
  }
  changeGroup(group, reset = true) {
    this.selectedGroup = group;
    this.getGroupData();
    const nodes = this.groupData.nodeData.nodes;
    this.nodeItems = nodes.map(
      (n, i) => $el(
        "li.draggable-item",
        {
          dataset: {
            nodeindex: n.index + ""
          },
          onclick: /* @__PURE__ */ __name(() => {
            this.changeNode(i);
          }, "onclick")
        },
        [
          $el("span.drag-handle"),
          $el(
            "div",
            {
              textContent: n.title ?? n.type
            },
            n.title ? $el("span", {
              textContent: n.type
            }) : []
          )
        ]
      )
    );
    this.innerNodesList.replaceChildren(...this.nodeItems);
    if (reset) {
      this.selectedNodeIndex = null;
      this.changeNode(0);
    } else {
      const items = this.draggable.getAllItems();
      let index = items.findIndex((item) => item.classList.contains("selected"));
      if (index === -1) index = this.selectedNodeIndex;
      this.changeNode(index, true);
    }
    const ordered = [...nodes];
    this.draggable?.dispose();
    this.draggable = new DraggableList(this.innerNodesList, "li");
    this.draggable.addEventListener(
      "dragend",
      ({ detail: { oldPosition, newPosition } }) => {
        if (oldPosition === newPosition) return;
        ordered.splice(newPosition, 0, ordered.splice(oldPosition, 1)[0]);
        for (let i = 0; i < ordered.length; i++) {
          this.storeModification({
            nodeIndex: ordered[i].index,
            section: ORDER,
            prop: "order",
            value: i
          });
        }
      }
    );
  }
  storeModification(props) {
    const { nodeIndex, section, prop, value } = props;
    const groupMod = this.modifications[this.selectedGroup] ??= {};
    const nodesMod = groupMod.nodes ??= {};
    const nodeMod = nodesMod[nodeIndex ?? this.selectedNodeInnerIndex] ??= {};
    const typeMod = nodeMod[section] ??= {};
    if (typeof value === "object") {
      const objMod = typeMod[prop] ??= {};
      Object.assign(objMod, value);
    } else {
      typeMod[prop] = value;
    }
  }
  getEditElement(section, prop, value, placeholder, checked, checkable = true) {
    if (value === placeholder) value = "";
    const mods = this.modifications[this.selectedGroup]?.nodes?.[this.selectedNodeInnerIndex]?.[section]?.[prop];
    if (mods) {
      if (mods.name != null) {
        value = mods.name;
      }
      if (mods.visible != null) {
        checked = mods.visible;
      }
    }
    return $el("div", [
      $el("input", {
        value,
        placeholder,
        type: "text",
        onchange: /* @__PURE__ */ __name((e) => {
          this.storeModification({
            section,
            prop,
            value: { name: e.target.value }
          });
        }, "onchange")
      }),
      $el("label", { textContent: "Visible" }, [
        $el("input", {
          type: "checkbox",
          checked,
          disabled: !checkable,
          onchange: /* @__PURE__ */ __name((e) => {
            this.storeModification({
              section,
              prop,
              value: { visible: !!e.target.checked }
            });
          }, "onchange")
        })
      ])
    ]);
  }
  buildWidgetsPage() {
    const widgets = this.groupData.oldToNewWidgetMap[this.selectedNodeInnerIndex];
    const items = Object.keys(widgets ?? {});
    const type = app.graph.extra.groupNodes[this.selectedGroup];
    const config = type.config?.[this.selectedNodeInnerIndex]?.input;
    this.widgetsPage.replaceChildren(
      ...items.map((oldName) => {
        return this.getEditElement(
          "input",
          oldName,
          widgets[oldName],
          oldName,
          config?.[oldName]?.visible !== false
        );
      })
    );
    return !!items.length;
  }
  buildInputsPage() {
    const inputs = this.groupData.nodeInputs[this.selectedNodeInnerIndex];
    const items = Object.keys(inputs ?? {});
    const type = app.graph.extra.groupNodes[this.selectedGroup];
    const config = type.config?.[this.selectedNodeInnerIndex]?.input;
    this.inputsPage.replaceChildren(
      ...items.map((oldName) => {
        let value = inputs[oldName];
        if (!value) {
          return;
        }
        return this.getEditElement(
          "input",
          oldName,
          value,
          oldName,
          config?.[oldName]?.visible !== false
        );
      }).filter(Boolean)
    );
    return !!items.length;
  }
  buildOutputsPage() {
    const nodes = this.groupData.nodeData.nodes;
    const innerNodeDef = this.groupData.getNodeDef(
      nodes[this.selectedNodeInnerIndex]
    );
    const outputs = innerNodeDef?.output ?? [];
    const groupOutputs = this.groupData.oldToNewOutputMap[this.selectedNodeInnerIndex];
    const type = app.graph.extra.groupNodes[this.selectedGroup];
    const config = type.config?.[this.selectedNodeInnerIndex]?.output;
    const node = this.groupData.nodeData.nodes[this.selectedNodeInnerIndex];
    const checkable = node.type !== "PrimitiveNode";
    this.outputsPage.replaceChildren(
      ...outputs.map((type2, slot) => {
        const groupOutputIndex = groupOutputs?.[slot];
        const oldName = innerNodeDef.output_name?.[slot] ?? type2;
        let value = config?.[slot]?.name;
        const visible = config?.[slot]?.visible || groupOutputIndex != null;
        if (!value || value === oldName) {
          value = "";
        }
        return this.getEditElement(
          "output",
          slot,
          value,
          oldName,
          visible,
          checkable
        );
      }).filter(Boolean)
    );
    return !!outputs.length;
  }
  show(type) {
    const groupNodes = Object.keys(app.graph.extra?.groupNodes ?? {}).sort(
      (a, b) => a.localeCompare(b)
    );
    this.innerNodesList = $el(
      "ul.comfy-group-manage-list-items"
    );
    this.widgetsPage = $el("section.comfy-group-manage-node-page");
    this.inputsPage = $el("section.comfy-group-manage-node-page");
    this.outputsPage = $el("section.comfy-group-manage-node-page");
    const pages = $el("div", [
      this.widgetsPage,
      this.inputsPage,
      this.outputsPage
    ]);
    this.tabs = [
      ["Inputs", this.inputsPage],
      ["Widgets", this.widgetsPage],
      ["Outputs", this.outputsPage]
    ].reduce((p, [name, page]) => {
      p[name] = {
        tab: $el("a", {
          onclick: /* @__PURE__ */ __name(() => {
            this.changeTab(name);
          }, "onclick"),
          textContent: name
        }),
        page
      };
      return p;
    }, {});
    const outer = $el("div.comfy-group-manage-outer", [
      $el("header", [
        $el("h2", "Group Nodes"),
        $el(
          "select",
          {
            onchange: /* @__PURE__ */ __name((e) => {
              this.changeGroup(e.target.value);
            }, "onchange")
          },
          groupNodes.map(
            (g) => $el("option", {
              textContent: g,
              selected: "workflow/" + g === type,
              value: g
            })
          )
        )
      ]),
      $el("main", [
        $el("section.comfy-group-manage-list", this.innerNodesList),
        $el("section.comfy-group-manage-node", [
          $el(
            "header",
            Object.values(this.tabs).map((t) => t.tab)
          ),
          pages
        ])
      ]),
      $el("footer", [
        $el(
          "button.comfy-btn",
          {
            onclick: /* @__PURE__ */ __name((e) => {
              const node = app.graph._nodes.find(
                (n) => n.type === "workflow/" + this.selectedGroup
              );
              if (node) {
                alert(
                  "This group node is in use in the current workflow, please first remove these."
                );
                return;
              }
              if (confirm(
                `Are you sure you want to remove the node: "${this.selectedGroup}"`
              )) {
                delete app.graph.extra.groupNodes[this.selectedGroup];
                LiteGraph.unregisterNodeType("workflow/" + this.selectedGroup);
              }
              this.show();
            }, "onclick")
          },
          "Delete Group Node"
        ),
        $el(
          "button.comfy-btn",
          {
            onclick: /* @__PURE__ */ __name(async () => {
              let nodesByType;
              let recreateNodes = [];
              const types = {};
              for (const g in this.modifications) {
                const type2 = app.graph.extra.groupNodes[g];
                let config = type2.config ??= {};
                let nodeMods = this.modifications[g]?.nodes;
                if (nodeMods) {
                  const keys = Object.keys(nodeMods);
                  if (nodeMods[keys[0]][ORDER]) {
                    const orderedNodes = [];
                    const orderedMods = {};
                    const orderedConfig = {};
                    for (const n of keys) {
                      const order = nodeMods[n][ORDER].order;
                      orderedNodes[order] = type2.nodes[+n];
                      orderedMods[order] = nodeMods[n];
                      orderedNodes[order].index = order;
                    }
                    for (const l of type2.links) {
                      if (l[0] != null) l[0] = type2.nodes[l[0]].index;
                      if (l[2] != null) l[2] = type2.nodes[l[2]].index;
                    }
                    if (type2.external) {
                      for (const ext2 of type2.external) {
                        ext2[0] = type2.nodes[ext2[0]];
                      }
                    }
                    for (const id2 of keys) {
                      if (config[id2]) {
                        orderedConfig[type2.nodes[id2].index] = config[id2];
                      }
                      delete config[id2];
                    }
                    type2.nodes = orderedNodes;
                    nodeMods = orderedMods;
                    type2.config = config = orderedConfig;
                  }
                  merge(config, nodeMods);
                }
                types[g] = type2;
                if (!nodesByType) {
                  nodesByType = app.graph._nodes.reduce((p, n) => {
                    p[n.type] ??= [];
                    p[n.type].push(n);
                    return p;
                  }, {});
                }
                const nodes = nodesByType["workflow/" + g];
                if (nodes) recreateNodes.push(...nodes);
              }
              await GroupNodeConfig.registerFromWorkflow(types, {});
              for (const node of recreateNodes) {
                node.recreate();
              }
              this.modifications = {};
              this.app.graph.setDirtyCanvas(true, true);
              this.changeGroup(this.selectedGroup, false);
            }, "onclick")
          },
          "Save"
        ),
        $el(
          "button.comfy-btn",
          { onclick: /* @__PURE__ */ __name(() => this.element.close(), "onclick") },
          "Close"
        )
      ])
    ]);
    this.element.replaceChildren(outer);
    this.changeGroup(
      type ? groupNodes.find((g) => "workflow/" + g === type) : groupNodes[0]
    );
    this.element.showModal();
    this.element.addEventListener("close", () => {
      this.draggable?.dispose();
    });
  }
}
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.groupNodeManage = window.comfyAPI.groupNodeManage || {};
window.comfyAPI.groupNodeManage.ManageGroupDialog = ManageGroupDialog;
const GROUP = Symbol();
const Workflow = {
  InUse: {
    Free: 0,
    Registered: 1,
    InWorkflow: 2
  },
  isInUseGroupNode(name) {
    const id2 = `workflow/${name}`;
    if (app.graph.extra?.groupNodes?.[name]) {
      if (app.graph._nodes.find((n) => n.type === id2)) {
        return Workflow.InUse.InWorkflow;
      } else {
        return Workflow.InUse.Registered;
      }
    }
    return Workflow.InUse.Free;
  },
  storeGroupNode(name, data) {
    let extra = app.graph.extra;
    if (!extra) app.graph.extra = extra = {};
    let groupNodes = extra.groupNodes;
    if (!groupNodes) extra.groupNodes = groupNodes = {};
    groupNodes[name] = data;
  }
};
class GroupNodeBuilder {
  static {
    __name(this, "GroupNodeBuilder");
  }
  nodes;
  nodeData;
  constructor(nodes) {
    this.nodes = nodes;
  }
  build() {
    const name = this.getName();
    if (!name) return;
    this.sortNodes();
    this.nodeData = this.getNodeData();
    Workflow.storeGroupNode(name, this.nodeData);
    return { name, nodeData: this.nodeData };
  }
  getName() {
    const name = prompt("Enter group name");
    if (!name) return;
    const used = Workflow.isInUseGroupNode(name);
    switch (used) {
      case Workflow.InUse.InWorkflow:
        alert(
          "An in use group node with this name already exists embedded in this workflow, please remove any instances or use a new name."
        );
        return;
      case Workflow.InUse.Registered:
        if (!confirm(
          "A group node with this name already exists embedded in this workflow, are you sure you want to overwrite it?"
        )) {
          return;
        }
        break;
    }
    return name;
  }
  sortNodes() {
    const nodesInOrder = app.graph.computeExecutionOrder(false);
    this.nodes = this.nodes.map((node) => ({ index: nodesInOrder.indexOf(node), node })).sort((a, b) => a.index - b.index || a.node.id - b.node.id).map(({ node }) => node);
  }
  getNodeData() {
    const storeLinkTypes = /* @__PURE__ */ __name((config) => {
      for (const link of config.links) {
        const origin = app.graph.getNodeById(link[4]);
        const type = origin.outputs[link[1]].type;
        link.push(type);
      }
    }, "storeLinkTypes");
    const storeExternalLinks = /* @__PURE__ */ __name((config) => {
      config.external = [];
      for (let i = 0; i < this.nodes.length; i++) {
        const node = this.nodes[i];
        if (!node.outputs?.length) continue;
        for (let slot = 0; slot < node.outputs.length; slot++) {
          let hasExternal = false;
          const output = node.outputs[slot];
          let type = output.type;
          if (!output.links?.length) continue;
          for (const l of output.links) {
            const link = app.graph.links[l];
            if (!link) continue;
            if (type === "*") type = link.type;
            if (!app.canvas.selected_nodes[link.target_id]) {
              hasExternal = true;
              break;
            }
          }
          if (hasExternal) {
            config.external.push([i, slot, type]);
          }
        }
      }
    }, "storeExternalLinks");
    const backup = localStorage.getItem("litegrapheditor_clipboard");
    try {
      app.canvas.copyToClipboard(this.nodes);
      const config = JSON.parse(
        localStorage.getItem("litegrapheditor_clipboard")
      );
      storeLinkTypes(config);
      storeExternalLinks(config);
      return config;
    } finally {
      localStorage.setItem("litegrapheditor_clipboard", backup);
    }
  }
}
class GroupNodeConfig {
  static {
    __name(this, "GroupNodeConfig");
  }
  name;
  nodeData;
  inputCount;
  oldToNewOutputMap;
  newToOldOutputMap;
  oldToNewInputMap;
  oldToNewWidgetMap;
  newToOldWidgetMap;
  primitiveDefs;
  widgetToPrimitive;
  primitiveToWidget;
  nodeInputs;
  outputVisibility;
  nodeDef;
  inputs;
  linksFrom;
  linksTo;
  externalFrom;
  constructor(name, nodeData) {
    this.name = name;
    this.nodeData = nodeData;
    this.getLinks();
    this.inputCount = 0;
    this.oldToNewOutputMap = {};
    this.newToOldOutputMap = {};
    this.oldToNewInputMap = {};
    this.oldToNewWidgetMap = {};
    this.newToOldWidgetMap = {};
    this.primitiveDefs = {};
    this.widgetToPrimitive = {};
    this.primitiveToWidget = {};
    this.nodeInputs = {};
    this.outputVisibility = [];
  }
  async registerType(source = "workflow") {
    this.nodeDef = {
      output: [],
      output_name: [],
      output_is_list: [],
      output_is_hidden: [],
      name: source + "/" + this.name,
      display_name: this.name,
      category: "group nodes" + ("/" + source),
      input: { required: {} },
      [GROUP]: this
    };
    this.inputs = [];
    const seenInputs = {};
    const seenOutputs = {};
    for (let i = 0; i < this.nodeData.nodes.length; i++) {
      const node = this.nodeData.nodes[i];
      node.index = i;
      this.processNode(node, seenInputs, seenOutputs);
    }
    for (const p of this.#convertedToProcess) {
      p();
    }
    this.#convertedToProcess = null;
    await app.registerNodeDef("workflow/" + this.name, this.nodeDef);
  }
  getLinks() {
    this.linksFrom = {};
    this.linksTo = {};
    this.externalFrom = {};
    for (const l of this.nodeData.links) {
      const [sourceNodeId, sourceNodeSlot, targetNodeId, targetNodeSlot] = l;
      if (sourceNodeId == null) continue;
      if (!this.linksFrom[sourceNodeId]) {
        this.linksFrom[sourceNodeId] = {};
      }
      if (!this.linksFrom[sourceNodeId][sourceNodeSlot]) {
        this.linksFrom[sourceNodeId][sourceNodeSlot] = [];
      }
      this.linksFrom[sourceNodeId][sourceNodeSlot].push(l);
      if (!this.linksTo[targetNodeId]) {
        this.linksTo[targetNodeId] = {};
      }
      this.linksTo[targetNodeId][targetNodeSlot] = l;
    }
    if (this.nodeData.external) {
      for (const ext2 of this.nodeData.external) {
        if (!this.externalFrom[ext2[0]]) {
          this.externalFrom[ext2[0]] = { [ext2[1]]: ext2[2] };
        } else {
          this.externalFrom[ext2[0]][ext2[1]] = ext2[2];
        }
      }
    }
  }
  processNode(node, seenInputs, seenOutputs) {
    const def = this.getNodeDef(node);
    if (!def) return;
    const inputs = { ...def.input?.required, ...def.input?.optional };
    this.inputs.push(this.processNodeInputs(node, seenInputs, inputs));
    if (def.output?.length) this.processNodeOutputs(node, seenOutputs, def);
  }
  getNodeDef(node) {
    const def = globalDefs[node.type];
    if (def) return def;
    const linksFrom = this.linksFrom[node.index];
    if (node.type === "PrimitiveNode") {
      if (!linksFrom) return;
      let type = linksFrom["0"][0][5];
      if (type === "COMBO") {
        const source = node.outputs[0].widget.name;
        const fromTypeName = this.nodeData.nodes[linksFrom["0"][0][2]].type;
        const fromType = globalDefs[fromTypeName];
        const input = fromType.input.required[source] ?? fromType.input.optional[source];
        type = input[0];
      }
      const def2 = this.primitiveDefs[node.index] = {
        input: {
          required: {
            value: [type, {}]
          }
        },
        output: [type],
        output_name: [],
        output_is_list: []
      };
      return def2;
    } else if (node.type === "Reroute") {
      const linksTo = this.linksTo[node.index];
      if (linksTo && linksFrom && !this.externalFrom[node.index]?.[0]) {
        return null;
      }
      let config = {};
      let rerouteType = "*";
      if (linksFrom) {
        for (const [, , id2, slot] of linksFrom["0"]) {
          const node2 = this.nodeData.nodes[id2];
          const input = node2.inputs[slot];
          if (rerouteType === "*") {
            rerouteType = input.type;
          }
          if (input.widget) {
            const targetDef = globalDefs[node2.type];
            const targetWidget = targetDef.input.required[input.widget.name] ?? targetDef.input.optional[input.widget.name];
            const widget = [targetWidget[0], config];
            const res = mergeIfValid(
              {
                widget
              },
              targetWidget,
              false,
              null,
              widget
            );
            config = res?.customConfig ?? config;
          }
        }
      } else if (linksTo) {
        const [id2, slot] = linksTo["0"];
        rerouteType = this.nodeData.nodes[id2].outputs[slot].type;
      } else {
        for (const l of this.nodeData.links) {
          if (l[2] === node.index) {
            rerouteType = l[5];
            break;
          }
        }
        if (rerouteType === "*") {
          const t = this.externalFrom[node.index]?.[0];
          if (t) {
            rerouteType = t;
          }
        }
      }
      config.forceInput = true;
      return {
        input: {
          required: {
            [rerouteType]: [rerouteType, config]
          }
        },
        output: [rerouteType],
        output_name: [],
        output_is_list: []
      };
    }
    console.warn(
      "Skipping virtual node " + node.type + " when building group node " + this.name
    );
  }
  getInputConfig(node, inputName, seenInputs, config, extra) {
    const customConfig = this.nodeData.config?.[node.index]?.input?.[inputName];
    let name = customConfig?.name ?? node.inputs?.find((inp) => inp.name === inputName)?.label ?? inputName;
    let key = name;
    let prefix = "";
    if (node.type === "PrimitiveNode" && node.title || name in seenInputs) {
      prefix = `${node.title ?? node.type} `;
      key = name = `${prefix}${inputName}`;
      if (name in seenInputs) {
        name = `${prefix}${seenInputs[name]} ${inputName}`;
      }
    }
    seenInputs[key] = (seenInputs[key] ?? 1) + 1;
    if (inputName === "seed" || inputName === "noise_seed") {
      if (!extra) extra = {};
      extra.control_after_generate = `${prefix}control_after_generate`;
    }
    if (config[0] === "IMAGEUPLOAD") {
      if (!extra) extra = {};
      extra.widget = this.oldToNewWidgetMap[node.index]?.[config[1]?.widget ?? "image"] ?? "image";
    }
    if (extra) {
      config = [config[0], { ...config[1], ...extra }];
    }
    return { name, config, customConfig };
  }
  processWidgetInputs(inputs, node, inputNames, seenInputs) {
    const slots = [];
    const converted = /* @__PURE__ */ new Map();
    const widgetMap = this.oldToNewWidgetMap[node.index] = {};
    for (const inputName of inputNames) {
      let widgetType = app.getWidgetType(inputs[inputName], inputName);
      if (widgetType) {
        const convertedIndex = node.inputs?.findIndex(
          (inp) => inp.name === inputName && inp.widget?.name === inputName
        );
        if (convertedIndex > -1) {
          converted.set(convertedIndex, inputName);
          widgetMap[inputName] = null;
        } else {
          const { name, config } = this.getInputConfig(
            node,
            inputName,
            seenInputs,
            inputs[inputName]
          );
          this.nodeDef.input.required[name] = config;
          widgetMap[inputName] = name;
          this.newToOldWidgetMap[name] = { node, inputName };
        }
      } else {
        slots.push(inputName);
      }
    }
    return { converted, slots };
  }
  checkPrimitiveConnection(link, inputName, inputs) {
    const sourceNode = this.nodeData.nodes[link[0]];
    if (sourceNode.type === "PrimitiveNode") {
      const [sourceNodeId, _, targetNodeId, __] = link;
      const primitiveDef = this.primitiveDefs[sourceNodeId];
      const targetWidget = inputs[inputName];
      const primitiveConfig = primitiveDef.input.required.value;
      const output = { widget: primitiveConfig };
      const config = mergeIfValid(
        output,
        targetWidget,
        false,
        null,
        primitiveConfig
      );
      primitiveConfig[1] = config?.customConfig ?? inputs[inputName][1] ? { ...inputs[inputName][1] } : {};
      let name = this.oldToNewWidgetMap[sourceNodeId]["value"];
      name = name.substr(0, name.length - 6);
      primitiveConfig[1].control_after_generate = true;
      primitiveConfig[1].control_prefix = name;
      let toPrimitive = this.widgetToPrimitive[targetNodeId];
      if (!toPrimitive) {
        toPrimitive = this.widgetToPrimitive[targetNodeId] = {};
      }
      if (toPrimitive[inputName]) {
        toPrimitive[inputName].push(sourceNodeId);
      }
      toPrimitive[inputName] = sourceNodeId;
      let toWidget = this.primitiveToWidget[sourceNodeId];
      if (!toWidget) {
        toWidget = this.primitiveToWidget[sourceNodeId] = [];
      }
      toWidget.push({ nodeId: targetNodeId, inputName });
    }
  }
  processInputSlots(inputs, node, slots, linksTo, inputMap, seenInputs) {
    this.nodeInputs[node.index] = {};
    for (let i = 0; i < slots.length; i++) {
      const inputName = slots[i];
      if (linksTo[i]) {
        this.checkPrimitiveConnection(linksTo[i], inputName, inputs);
        continue;
      }
      const { name, config, customConfig } = this.getInputConfig(
        node,
        inputName,
        seenInputs,
        inputs[inputName]
      );
      this.nodeInputs[node.index][inputName] = name;
      if (customConfig?.visible === false) continue;
      this.nodeDef.input.required[name] = config;
      inputMap[i] = this.inputCount++;
    }
  }
  processConvertedWidgets(inputs, node, slots, converted, linksTo, inputMap, seenInputs) {
    const convertedSlots = [...converted.keys()].sort().map((k) => converted.get(k));
    for (let i = 0; i < convertedSlots.length; i++) {
      const inputName = convertedSlots[i];
      if (linksTo[slots.length + i]) {
        this.checkPrimitiveConnection(
          linksTo[slots.length + i],
          inputName,
          inputs
        );
        continue;
      }
      const { name, config } = this.getInputConfig(
        node,
        inputName,
        seenInputs,
        inputs[inputName],
        {
          defaultInput: true
        }
      );
      this.nodeDef.input.required[name] = config;
      this.newToOldWidgetMap[name] = { node, inputName };
      if (!this.oldToNewWidgetMap[node.index]) {
        this.oldToNewWidgetMap[node.index] = {};
      }
      this.oldToNewWidgetMap[node.index][inputName] = name;
      inputMap[slots.length + i] = this.inputCount++;
    }
  }
  #convertedToProcess = [];
  processNodeInputs(node, seenInputs, inputs) {
    const inputMapping = [];
    const inputNames = Object.keys(inputs);
    if (!inputNames.length) return;
    const { converted, slots } = this.processWidgetInputs(
      inputs,
      node,
      inputNames,
      seenInputs
    );
    const linksTo = this.linksTo[node.index] ?? {};
    const inputMap = this.oldToNewInputMap[node.index] = {};
    this.processInputSlots(inputs, node, slots, linksTo, inputMap, seenInputs);
    this.#convertedToProcess.push(
      () => this.processConvertedWidgets(
        inputs,
        node,
        slots,
        converted,
        linksTo,
        inputMap,
        seenInputs
      )
    );
    return inputMapping;
  }
  processNodeOutputs(node, seenOutputs, def) {
    const oldToNew = this.oldToNewOutputMap[node.index] = {};
    for (let outputId = 0; outputId < def.output.length; outputId++) {
      const linksFrom = this.linksFrom[node.index];
      const hasLink = linksFrom?.[outputId] && !this.externalFrom[node.index]?.[outputId];
      const customConfig = this.nodeData.config?.[node.index]?.output?.[outputId];
      const visible = customConfig?.visible ?? !hasLink;
      this.outputVisibility.push(visible);
      if (!visible) {
        continue;
      }
      oldToNew[outputId] = this.nodeDef.output.length;
      this.newToOldOutputMap[this.nodeDef.output.length] = {
        node,
        slot: outputId
      };
      this.nodeDef.output.push(def.output[outputId]);
      this.nodeDef.output_is_list.push(def.output_is_list[outputId]);
      let label = customConfig?.name;
      if (!label) {
        label = def.output_name?.[outputId] ?? def.output[outputId];
        const output = node.outputs.find((o) => o.name === label);
        if (output?.label) {
          label = output.label;
        }
      }
      let name = label;
      if (name in seenOutputs) {
        const prefix = `${node.title ?? node.type} `;
        name = `${prefix}${label}`;
        if (name in seenOutputs) {
          name = `${prefix}${node.index} ${label}`;
        }
      }
      seenOutputs[name] = 1;
      this.nodeDef.output_name.push(name);
    }
  }
  static async registerFromWorkflow(groupNodes, missingNodeTypes) {
    const clean = app.clean;
    app.clean = function() {
      for (const g in groupNodes) {
        try {
          LiteGraph.unregisterNodeType("workflow/" + g);
        } catch (error) {
        }
      }
      app.clean = clean;
    };
    for (const g in groupNodes) {
      const groupData = groupNodes[g];
      let hasMissing = false;
      for (const n of groupData.nodes) {
        if (!(n.type in LiteGraph.registered_node_types)) {
          missingNodeTypes.push({
            type: n.type,
            hint: ` (In group node 'workflow/${g}')`
          });
          missingNodeTypes.push({
            type: "workflow/" + g,
            action: {
              text: "Remove from workflow",
              callback: /* @__PURE__ */ __name((e) => {
                delete groupNodes[g];
                e.target.textContent = "Removed";
                e.target.style.pointerEvents = "none";
                e.target.style.opacity = 0.7;
              }, "callback")
            }
          });
          hasMissing = true;
        }
      }
      if (hasMissing) continue;
      const config = new GroupNodeConfig(g, groupData);
      await config.registerType();
    }
  }
}
class GroupNodeHandler {
  static {
    __name(this, "GroupNodeHandler");
  }
  node;
  groupData;
  innerNodes;
  constructor(node) {
    this.node = node;
    this.groupData = node.constructor?.nodeData?.[GROUP];
    this.node.setInnerNodes = (innerNodes) => {
      this.innerNodes = innerNodes;
      for (let innerNodeIndex = 0; innerNodeIndex < this.innerNodes.length; innerNodeIndex++) {
        const innerNode = this.innerNodes[innerNodeIndex];
        for (const w of innerNode.widgets ?? []) {
          if (w.type === "converted-widget") {
            w.serializeValue = w.origSerializeValue;
          }
        }
        innerNode.index = innerNodeIndex;
        innerNode.getInputNode = (slot) => {
          const externalSlot = this.groupData.oldToNewInputMap[innerNode.index]?.[slot];
          if (externalSlot != null) {
            return this.node.getInputNode(externalSlot);
          }
          const innerLink = this.groupData.linksTo[innerNode.index]?.[slot];
          if (!innerLink) return null;
          const inputNode = innerNodes[innerLink[0]];
          if (inputNode.type === "PrimitiveNode") return null;
          return inputNode;
        };
        innerNode.getInputLink = (slot) => {
          const externalSlot = this.groupData.oldToNewInputMap[innerNode.index]?.[slot];
          if (externalSlot != null) {
            const linkId = this.node.inputs[externalSlot].link;
            let link2 = app.graph.links[linkId];
            link2 = {
              ...link2,
              target_id: innerNode.id,
              target_slot: +slot
            };
            return link2;
          }
          let link = this.groupData.linksTo[innerNode.index]?.[slot];
          if (!link) return null;
          link = {
            origin_id: innerNodes[link[0]].id,
            origin_slot: link[1],
            target_id: innerNode.id,
            target_slot: +slot
          };
          return link;
        };
      }
    };
    this.node.updateLink = (link) => {
      link = { ...link };
      const output = this.groupData.newToOldOutputMap[link.origin_slot];
      let innerNode = this.innerNodes[output.node.index];
      let l;
      while (innerNode?.type === "Reroute") {
        l = innerNode.getInputLink(0);
        innerNode = innerNode.getInputNode(0);
      }
      if (!innerNode) {
        return null;
      }
      if (l && GroupNodeHandler.isGroupNode(innerNode)) {
        return innerNode.updateLink(l);
      }
      link.origin_id = innerNode.id;
      link.origin_slot = l?.origin_slot ?? output.slot;
      return link;
    };
    this.node.getInnerNodes = () => {
      if (!this.innerNodes) {
        this.node.setInnerNodes(
          this.groupData.nodeData.nodes.map((n, i) => {
            const innerNode = LiteGraph.createNode(n.type);
            innerNode.configure(n);
            innerNode.id = `${this.node.id}:${i}`;
            return innerNode;
          })
        );
      }
      this.updateInnerWidgets();
      return this.innerNodes;
    };
    this.node.recreate = async () => {
      const id2 = this.node.id;
      const sz = this.node.size;
      const nodes = this.node.convertToNodes();
      const groupNode = LiteGraph.createNode(this.node.type);
      groupNode.id = id2;
      groupNode.setInnerNodes(nodes);
      groupNode[GROUP].populateWidgets();
      app.graph.add(groupNode);
      groupNode.size = [
        Math.max(groupNode.size[0], sz[0]),
        Math.max(groupNode.size[1], sz[1])
      ];
      groupNode[GROUP].replaceNodes(nodes);
      return groupNode;
    };
    this.node.convertToNodes = () => {
      const addInnerNodes = /* @__PURE__ */ __name(() => {
        const backup = localStorage.getItem("litegrapheditor_clipboard");
        const c = { ...this.groupData.nodeData };
        c.nodes = [...c.nodes];
        const innerNodes = this.node.getInnerNodes();
        let ids = [];
        for (let i = 0; i < c.nodes.length; i++) {
          let id2 = innerNodes?.[i]?.id;
          if (id2 == null || isNaN(id2)) {
            id2 = void 0;
          } else {
            ids.push(id2);
          }
          c.nodes[i] = { ...c.nodes[i], id: id2 };
        }
        localStorage.setItem("litegrapheditor_clipboard", JSON.stringify(c));
        app.canvas.pasteFromClipboard();
        localStorage.setItem("litegrapheditor_clipboard", backup);
        const [x, y] = this.node.pos;
        let top;
        let left;
        const selectedIds2 = ids.length ? ids : Object.keys(app.canvas.selected_nodes);
        const newNodes2 = [];
        for (let i = 0; i < selectedIds2.length; i++) {
          const id2 = selectedIds2[i];
          const newNode = app.graph.getNodeById(id2);
          const innerNode = innerNodes[i];
          newNodes2.push(newNode);
          if (left == null || newNode.pos[0] < left) {
            left = newNode.pos[0];
          }
          if (top == null || newNode.pos[1] < top) {
            top = newNode.pos[1];
          }
          if (!newNode.widgets) continue;
          const map = this.groupData.oldToNewWidgetMap[innerNode.index];
          if (map) {
            const widgets = Object.keys(map);
            for (const oldName of widgets) {
              const newName = map[oldName];
              if (!newName) continue;
              const widgetIndex = this.node.widgets.findIndex(
                (w) => w.name === newName
              );
              if (widgetIndex === -1) continue;
              if (innerNode.type === "PrimitiveNode") {
                for (let i2 = 0; i2 < newNode.widgets.length; i2++) {
                  newNode.widgets[i2].value = this.node.widgets[widgetIndex + i2].value;
                }
              } else {
                const outerWidget = this.node.widgets[widgetIndex];
                const newWidget = newNode.widgets.find(
                  (w) => w.name === oldName
                );
                if (!newWidget) continue;
                newWidget.value = outerWidget.value;
                for (let w = 0; w < outerWidget.linkedWidgets?.length; w++) {
                  newWidget.linkedWidgets[w].value = outerWidget.linkedWidgets[w].value;
                }
              }
            }
          }
        }
        for (const newNode of newNodes2) {
          newNode.pos = [
            newNode.pos[0] - (left - x),
            newNode.pos[1] - (top - y)
          ];
        }
        return { newNodes: newNodes2, selectedIds: selectedIds2 };
      }, "addInnerNodes");
      const reconnectInputs = /* @__PURE__ */ __name((selectedIds2) => {
        for (const innerNodeIndex in this.groupData.oldToNewInputMap) {
          const id2 = selectedIds2[innerNodeIndex];
          const newNode = app.graph.getNodeById(id2);
          const map = this.groupData.oldToNewInputMap[innerNodeIndex];
          for (const innerInputId in map) {
            const groupSlotId = map[innerInputId];
            if (groupSlotId == null) continue;
            const slot = node.inputs[groupSlotId];
            if (slot.link == null) continue;
            const link = app.graph.links[slot.link];
            if (!link) continue;
            const originNode = app.graph.getNodeById(link.origin_id);
            originNode.connect(link.origin_slot, newNode, +innerInputId);
          }
        }
      }, "reconnectInputs");
      const reconnectOutputs = /* @__PURE__ */ __name((selectedIds2) => {
        for (let groupOutputId = 0; groupOutputId < node.outputs?.length; groupOutputId++) {
          const output = node.outputs[groupOutputId];
          if (!output.links) continue;
          const links = [...output.links];
          for (const l of links) {
            const slot = this.groupData.newToOldOutputMap[groupOutputId];
            const link = app.graph.links[l];
            const targetNode = app.graph.getNodeById(link.target_id);
            const newNode = app.graph.getNodeById(selectedIds2[slot.node.index]);
            newNode.connect(slot.slot, targetNode, link.target_slot);
          }
        }
      }, "reconnectOutputs");
      const { newNodes, selectedIds } = addInnerNodes();
      reconnectInputs(selectedIds);
      reconnectOutputs(selectedIds);
      app.graph.remove(this.node);
      return newNodes;
    };
    const getExtraMenuOptions = this.node.getExtraMenuOptions;
    this.node.getExtraMenuOptions = function(_, options) {
      getExtraMenuOptions?.apply(this, arguments);
      let optionIndex = options.findIndex((o) => o.content === "Outputs");
      if (optionIndex === -1) optionIndex = options.length;
      else optionIndex++;
      options.splice(
        optionIndex,
        0,
        null,
        {
          content: "Convert to nodes",
          callback: /* @__PURE__ */ __name(() => {
            return this.convertToNodes();
          }, "callback")
        },
        {
          content: "Manage Group Node",
          callback: /* @__PURE__ */ __name(() => {
            new ManageGroupDialog(app).show(this.type);
          }, "callback")
        }
      );
    };
    const onDrawTitleBox = this.node.onDrawTitleBox;
    this.node.onDrawTitleBox = function(ctx, height, size, scale) {
      onDrawTitleBox?.apply(this, arguments);
      const fill = ctx.fillStyle;
      ctx.beginPath();
      ctx.rect(11, -height + 11, 2, 2);
      ctx.rect(14, -height + 11, 2, 2);
      ctx.rect(17, -height + 11, 2, 2);
      ctx.rect(11, -height + 14, 2, 2);
      ctx.rect(14, -height + 14, 2, 2);
      ctx.rect(17, -height + 14, 2, 2);
      ctx.rect(11, -height + 17, 2, 2);
      ctx.rect(14, -height + 17, 2, 2);
      ctx.rect(17, -height + 17, 2, 2);
      ctx.fillStyle = this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR;
      ctx.fill();
      ctx.fillStyle = fill;
    };
    const onDrawForeground = node.onDrawForeground;
    const groupData = this.groupData.nodeData;
    node.onDrawForeground = function(ctx) {
      const r = onDrawForeground?.apply?.(this, arguments);
      if (+app.runningNodeId === this.id && this.runningInternalNodeId !== null) {
        const n = groupData.nodes[this.runningInternalNodeId];
        if (!n) return;
        const message = `Running ${n.title || n.type} (${this.runningInternalNodeId}/${groupData.nodes.length})`;
        ctx.save();
        ctx.font = "12px sans-serif";
        const sz = ctx.measureText(message);
        ctx.fillStyle = node.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR;
        ctx.beginPath();
        ctx.roundRect(
          0,
          -LiteGraph.NODE_TITLE_HEIGHT - 20,
          sz.width + 12,
          20,
          5
        );
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.fillText(message, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
        ctx.restore();
      }
    };
    const onExecutionStart = this.node.onExecutionStart;
    this.node.onExecutionStart = function() {
      this.resetExecution = true;
      return onExecutionStart?.apply(this, arguments);
    };
    const self = this;
    const onNodeCreated = this.node.onNodeCreated;
    this.node.onNodeCreated = function() {
      if (!this.widgets) {
        return;
      }
      const config = self.groupData.nodeData.config;
      if (config) {
        for (const n in config) {
          const inputs = config[n]?.input;
          for (const w in inputs) {
            if (inputs[w].visible !== false) continue;
            const widgetName = self.groupData.oldToNewWidgetMap[n][w];
            const widget = this.widgets.find((w2) => w2.name === widgetName);
            if (widget) {
              widget.type = "hidden";
              widget.computeSize = () => [0, -4];
            }
          }
        }
      }
      return onNodeCreated?.apply(this, arguments);
    };
    function handleEvent(type, getId, getEvent) {
      const handler = /* @__PURE__ */ __name(({ detail }) => {
        const id2 = getId(detail);
        if (!id2) return;
        const node2 = app.graph.getNodeById(id2);
        if (node2) return;
        const innerNodeIndex = this.innerNodes?.findIndex((n) => n.id == id2);
        if (innerNodeIndex > -1) {
          this.node.runningInternalNodeId = innerNodeIndex;
          api.dispatchEvent(
            new CustomEvent(type, {
              detail: getEvent(detail, this.node.id + "", this.node)
            })
          );
        }
      }, "handler");
      api.addEventListener(type, handler);
      return handler;
    }
    __name(handleEvent, "handleEvent");
    const executing = handleEvent.call(
      this,
      "executing",
      (d) => d,
      (d, id2, node2) => id2
    );
    const executed = handleEvent.call(
      this,
      "executed",
      (d) => d?.display_node || d?.node,
      (d, id2, node2) => ({
        ...d,
        node: id2,
        display_node: id2,
        merge: !node2.resetExecution
      })
    );
    const onRemoved = node.onRemoved;
    this.node.onRemoved = function() {
      onRemoved?.apply(this, arguments);
      api.removeEventListener("executing", executing);
      api.removeEventListener("executed", executed);
    };
    this.node.refreshComboInNode = (defs) => {
      for (const widgetName in this.groupData.newToOldWidgetMap) {
        const widget = this.node.widgets.find((w) => w.name === widgetName);
        if (widget?.type === "combo") {
          const old = this.groupData.newToOldWidgetMap[widgetName];
          const def = defs[old.node.type];
          const input = def?.input?.required?.[old.inputName] ?? def?.input?.optional?.[old.inputName];
          if (!input) continue;
          widget.options.values = input[0];
          if (old.inputName !== "image" && !widget.options.values.includes(widget.value)) {
            widget.value = widget.options.values[0];
            widget.callback(widget.value);
          }
        }
      }
    };
  }
  updateInnerWidgets() {
    for (const newWidgetName in this.groupData.newToOldWidgetMap) {
      const newWidget = this.node.widgets.find((w) => w.name === newWidgetName);
      if (!newWidget) continue;
      const newValue = newWidget.value;
      const old = this.groupData.newToOldWidgetMap[newWidgetName];
      let innerNode = this.innerNodes[old.node.index];
      if (innerNode.type === "PrimitiveNode") {
        innerNode.primitiveValue = newValue;
        const primitiveLinked = this.groupData.primitiveToWidget[old.node.index];
        for (const linked of primitiveLinked ?? []) {
          const node = this.innerNodes[linked.nodeId];
          const widget2 = node.widgets.find((w) => w.name === linked.inputName);
          if (widget2) {
            widget2.value = newValue;
          }
        }
        continue;
      } else if (innerNode.type === "Reroute") {
        const rerouteLinks = this.groupData.linksFrom[old.node.index];
        if (rerouteLinks) {
          for (const [_, , targetNodeId, targetSlot] of rerouteLinks["0"]) {
            const node = this.innerNodes[targetNodeId];
            const input = node.inputs[targetSlot];
            if (input.widget) {
              const widget2 = node.widgets?.find(
                (w) => w.name === input.widget.name
              );
              if (widget2) {
                widget2.value = newValue;
              }
            }
          }
        }
      }
      const widget = innerNode.widgets?.find((w) => w.name === old.inputName);
      if (widget) {
        widget.value = newValue;
      }
    }
  }
  populatePrimitive(node, nodeId, oldName, i, linkedShift) {
    const primitiveId = this.groupData.widgetToPrimitive[nodeId]?.[oldName];
    if (primitiveId == null) return;
    const targetWidgetName = this.groupData.oldToNewWidgetMap[primitiveId]["value"];
    const targetWidgetIndex = this.node.widgets.findIndex(
      (w) => w.name === targetWidgetName
    );
    if (targetWidgetIndex > -1) {
      const primitiveNode = this.innerNodes[primitiveId];
      let len = primitiveNode.widgets.length;
      if (len - 1 !== this.node.widgets[targetWidgetIndex].linkedWidgets?.length) {
        len = 1;
      }
      for (let i2 = 0; i2 < len; i2++) {
        this.node.widgets[targetWidgetIndex + i2].value = primitiveNode.widgets[i2].value;
      }
    }
    return true;
  }
  populateReroute(node, nodeId, map) {
    if (node.type !== "Reroute") return;
    const link = this.groupData.linksFrom[nodeId]?.[0]?.[0];
    if (!link) return;
    const [, , targetNodeId, targetNodeSlot] = link;
    const targetNode = this.groupData.nodeData.nodes[targetNodeId];
    const inputs = targetNode.inputs;
    const targetWidget = inputs?.[targetNodeSlot]?.widget;
    if (!targetWidget) return;
    const offset = inputs.length - (targetNode.widgets_values?.length ?? 0);
    const v = targetNode.widgets_values?.[targetNodeSlot - offset];
    if (v == null) return;
    const widgetName = Object.values(map)[0];
    const widget = this.node.widgets.find((w) => w.name === widgetName);
    if (widget) {
      widget.value = v;
    }
  }
  populateWidgets() {
    if (!this.node.widgets) return;
    for (let nodeId = 0; nodeId < this.groupData.nodeData.nodes.length; nodeId++) {
      const node = this.groupData.nodeData.nodes[nodeId];
      const map = this.groupData.oldToNewWidgetMap[nodeId] ?? {};
      const widgets = Object.keys(map);
      if (!node.widgets_values?.length) {
        this.populateReroute(node, nodeId, map);
        continue;
      }
      let linkedShift = 0;
      for (let i = 0; i < widgets.length; i++) {
        const oldName = widgets[i];
        const newName = map[oldName];
        const widgetIndex = this.node.widgets.findIndex(
          (w) => w.name === newName
        );
        const mainWidget = this.node.widgets[widgetIndex];
        if (this.populatePrimitive(node, nodeId, oldName, i, linkedShift) || widgetIndex === -1) {
          const innerWidget = this.innerNodes[nodeId].widgets?.find(
            (w) => w.name === oldName
          );
          linkedShift += innerWidget?.linkedWidgets?.length ?? 0;
        }
        if (widgetIndex === -1) {
          continue;
        }
        mainWidget.value = node.widgets_values[i + linkedShift];
        for (let w = 0; w < mainWidget.linkedWidgets?.length; w++) {
          this.node.widgets[widgetIndex + w + 1].value = node.widgets_values[i + ++linkedShift];
        }
      }
    }
  }
  replaceNodes(nodes) {
    let top;
    let left;
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (left == null || node.pos[0] < left) {
        left = node.pos[0];
      }
      if (top == null || node.pos[1] < top) {
        top = node.pos[1];
      }
      this.linkOutputs(node, i);
      app.graph.remove(node);
    }
    this.linkInputs();
    this.node.pos = [left, top];
  }
  linkOutputs(originalNode, nodeId) {
    if (!originalNode.outputs) return;
    for (const output of originalNode.outputs) {
      if (!output.links) continue;
      const links = [...output.links];
      for (const l of links) {
        const link = app.graph.links[l];
        if (!link) continue;
        const targetNode = app.graph.getNodeById(link.target_id);
        const newSlot = this.groupData.oldToNewOutputMap[nodeId]?.[link.origin_slot];
        if (newSlot != null) {
          this.node.connect(newSlot, targetNode, link.target_slot);
        }
      }
    }
  }
  linkInputs() {
    for (const link of this.groupData.nodeData.links ?? []) {
      const [, originSlot, targetId, targetSlot, actualOriginId] = link;
      const originNode = app.graph.getNodeById(actualOriginId);
      if (!originNode) continue;
      originNode.connect(
        originSlot,
        this.node.id,
        this.groupData.oldToNewInputMap[targetId][targetSlot]
      );
    }
  }
  static getGroupData(node) {
    return (node.nodeData ?? node.constructor?.nodeData)?.[GROUP];
  }
  static isGroupNode(node) {
    return !!node.constructor?.nodeData?.[GROUP];
  }
  static async fromNodes(nodes) {
    const builder = new GroupNodeBuilder(nodes);
    const res = builder.build();
    if (!res) return;
    const { name, nodeData } = res;
    const config = new GroupNodeConfig(name, nodeData);
    await config.registerType();
    const groupNode = LiteGraph.createNode(`workflow/${name}`);
    groupNode.setInnerNodes(builder.nodes);
    groupNode[GROUP].populateWidgets();
    app.graph.add(groupNode);
    groupNode[GROUP].replaceNodes(builder.nodes);
    return groupNode;
  }
}
function addConvertToGroupOptions() {
  function addConvertOption(options, index) {
    const selected = Object.values(app.canvas.selected_nodes ?? {});
    const disabled = selected.length < 2 || selected.find((n) => GroupNodeHandler.isGroupNode(n));
    options.splice(index + 1, null, {
      content: `Convert to Group Node`,
      disabled,
      callback: /* @__PURE__ */ __name(async () => {
        return await GroupNodeHandler.fromNodes(selected);
      }, "callback")
    });
  }
  __name(addConvertOption, "addConvertOption");
  function addManageOption(options, index) {
    const groups = app.graph.extra?.groupNodes;
    const disabled = !groups || !Object.keys(groups).length;
    options.splice(index + 1, null, {
      content: `Manage Group Nodes`,
      disabled,
      callback: /* @__PURE__ */ __name(() => {
        new ManageGroupDialog(app).show();
      }, "callback")
    });
  }
  __name(addManageOption, "addManageOption");
  const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
  LGraphCanvas.prototype.getCanvasMenuOptions = function() {
    const options = getCanvasMenuOptions.apply(this, arguments);
    const index = options.findIndex((o) => o?.content === "Add Group") + 1 || options.length;
    addConvertOption(options, index);
    addManageOption(options, index + 1);
    return options;
  };
  const getNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
  LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
    const options = getNodeMenuOptions.apply(this, arguments);
    if (!GroupNodeHandler.isGroupNode(node)) {
      const index = options.findIndex((o) => o?.content === "Outputs") + 1 || options.length - 1;
      addConvertOption(options, index);
    }
    return options;
  };
}
__name(addConvertToGroupOptions, "addConvertToGroupOptions");
const id$3 = "Comfy.GroupNode";
let globalDefs;
const ext$1 = {
  name: id$3,
  setup() {
    addConvertToGroupOptions();
  },
  async beforeConfigureGraph(graphData, missingNodeTypes) {
    const nodes = graphData?.extra?.groupNodes;
    if (nodes) {
      await GroupNodeConfig.registerFromWorkflow(nodes, missingNodeTypes);
    }
  },
  addCustomNodeDefs(defs) {
    globalDefs = defs;
  },
  nodeCreated(node) {
    if (GroupNodeHandler.isGroupNode(node)) {
      node[GROUP] = new GroupNodeHandler(node);
    }
  },
  async refreshComboInNodes(defs) {
    Object.assign(globalDefs, defs);
    const nodes = app.graph.extra?.groupNodes;
    if (nodes) {
      await GroupNodeConfig.registerFromWorkflow(nodes, {});
    }
  }
};
app.registerExtension(ext$1);
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.groupNode = window.comfyAPI.groupNode || {};
window.comfyAPI.groupNode.GroupNodeConfig = GroupNodeConfig;
window.comfyAPI.groupNode.GroupNodeHandler = GroupNodeHandler;
function setNodeMode(node, mode) {
  node.mode = mode;
  node.graph.change();
}
__name(setNodeMode, "setNodeMode");
function addNodesToGroup(group, nodes = []) {
  var x1, y1, x2, y2;
  var nx1, ny1, nx2, ny2;
  var node;
  x1 = y1 = x2 = y2 = -1;
  nx1 = ny1 = nx2 = ny2 = -1;
  for (var n of [group._nodes, nodes]) {
    for (var i in n) {
      node = n[i];
      nx1 = node.pos[0];
      ny1 = node.pos[1];
      nx2 = node.pos[0] + node.size[0];
      ny2 = node.pos[1] + node.size[1];
      if (node.type != "Reroute") {
        ny1 -= LiteGraph.NODE_TITLE_HEIGHT;
      }
      if (node.flags?.collapsed) {
        ny2 = ny1 + LiteGraph.NODE_TITLE_HEIGHT;
        if (node?._collapsed_width) {
          nx2 = nx1 + Math.round(node._collapsed_width);
        }
      }
      if (x1 == -1 || nx1 < x1) {
        x1 = nx1;
      }
      if (y1 == -1 || ny1 < y1) {
        y1 = ny1;
      }
      if (x2 == -1 || nx2 > x2) {
        x2 = nx2;
      }
      if (y2 == -1 || ny2 > y2) {
        y2 = ny2;
      }
    }
  }
  var padding = 10;
  y1 = y1 - Math.round(group.font_size * 1.4);
  group.pos = [x1 - padding, y1 - padding];
  group.size = [x2 - x1 + padding * 2, y2 - y1 + padding * 2];
}
__name(addNodesToGroup, "addNodesToGroup");
app.registerExtension({
  name: "Comfy.GroupOptions",
  setup() {
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function() {
      const options = orig.apply(this, arguments);
      const group = this.graph.getGroupOnPos(
        this.graph_mouse[0],
        this.graph_mouse[1]
      );
      if (!group) {
        options.push({
          content: "Add Group For Selected Nodes",
          disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
          callback: /* @__PURE__ */ __name(() => {
            const group2 = new LGraphGroup();
            addNodesToGroup(group2, this.selected_nodes);
            app.canvas.graph.add(group2);
            this.graph.change();
          }, "callback")
        });
        return options;
      }
      group.recomputeInsideNodes();
      const nodesInGroup = group._nodes;
      options.push({
        content: "Add Selected Nodes To Group",
        disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
        callback: /* @__PURE__ */ __name(() => {
          addNodesToGroup(group, this.selected_nodes);
          this.graph.change();
        }, "callback")
      });
      if (nodesInGroup.length === 0) {
        return options;
      } else {
        options.push(null);
      }
      let allNodesAreSameMode = true;
      for (let i = 1; i < nodesInGroup.length; i++) {
        if (nodesInGroup[i].mode !== nodesInGroup[0].mode) {
          allNodesAreSameMode = false;
          break;
        }
      }
      options.push({
        content: "Fit Group To Nodes",
        callback: /* @__PURE__ */ __name(() => {
          addNodesToGroup(group);
          this.graph.change();
        }, "callback")
      });
      options.push({
        content: "Select Nodes",
        callback: /* @__PURE__ */ __name(() => {
          this.selectNodes(nodesInGroup);
          this.graph.change();
          this.canvas.focus();
        }, "callback")
      });
      if (allNodesAreSameMode) {
        const mode = nodesInGroup[0].mode;
        switch (mode) {
          case 0:
            options.push({
              content: "Set Group Nodes to Never",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 2);
                }
              }, "callback")
            });
            options.push({
              content: "Bypass Group Nodes",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 4);
                }
              }, "callback")
            });
            break;
          case 2:
            options.push({
              content: "Set Group Nodes to Always",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 0);
                }
              }, "callback")
            });
            options.push({
              content: "Bypass Group Nodes",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 4);
                }
              }, "callback")
            });
            break;
          case 4:
            options.push({
              content: "Set Group Nodes to Always",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 0);
                }
              }, "callback")
            });
            options.push({
              content: "Set Group Nodes to Never",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 2);
                }
              }, "callback")
            });
            break;
          default:
            options.push({
              content: "Set Group Nodes to Always",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 0);
                }
              }, "callback")
            });
            options.push({
              content: "Set Group Nodes to Never",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 2);
                }
              }, "callback")
            });
            options.push({
              content: "Bypass Group Nodes",
              callback: /* @__PURE__ */ __name(() => {
                for (const node of nodesInGroup) {
                  setNodeMode(node, 4);
                }
              }, "callback")
            });
            break;
        }
      } else {
        options.push({
          content: "Set Group Nodes to Always",
          callback: /* @__PURE__ */ __name(() => {
            for (const node of nodesInGroup) {
              setNodeMode(node, 0);
            }
          }, "callback")
        });
        options.push({
          content: "Set Group Nodes to Never",
          callback: /* @__PURE__ */ __name(() => {
            for (const node of nodesInGroup) {
              setNodeMode(node, 2);
            }
          }, "callback")
        });
        options.push({
          content: "Bypass Group Nodes",
          callback: /* @__PURE__ */ __name(() => {
            for (const node of nodesInGroup) {
              setNodeMode(node, 4);
            }
          }, "callback")
        });
      }
      return options;
    };
  }
});
const id$2 = "Comfy.InvertMenuScrolling";
app.registerExtension({
  name: id$2,
  init() {
    const ctxMenu = LiteGraph.ContextMenu;
    const replace = /* @__PURE__ */ __name(() => {
      LiteGraph.ContextMenu = function(values, options) {
        options = options || {};
        if (options.scroll_speed) {
          options.scroll_speed *= -1;
        } else {
          options.scroll_speed = -0.1;
        }
        return ctxMenu.call(this, values, options);
      };
      LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
    }, "replace");
    app.ui.settings.addSetting({
      id: id$2,
      category: ["Comfy", "Graph", "InvertMenuScrolling"],
      name: "Invert Context Menu Scrolling",
      type: "boolean",
      defaultValue: false,
      onChange(value) {
        if (value) {
          replace();
        } else {
          LiteGraph.ContextMenu = ctxMenu;
        }
      }
    });
  }
});
app.registerExtension({
  name: "Comfy.Keybinds",
  init() {
    const keybindListener = /* @__PURE__ */ __name(async function(event) {
      const modifierPressed = event.ctrlKey || event.metaKey;
      if (modifierPressed && event.key === "Enter") {
        if (event.altKey) {
          await api.interrupt();
          useToastStore().add({
            severity: "info",
            summary: "Interrupted",
            detail: "Execution has been interrupted",
            life: 1e3
          });
          return;
        }
        app.queuePrompt(event.shiftKey ? -1 : 0).then();
        return;
      }
      const target = event.composedPath()[0];
      if (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.tagName === "SPAN" && target.classList.contains("property_value")) {
        return;
      }
      const modifierKeyIdMap = {
        s: "#comfy-save-button",
        o: "#comfy-file-input",
        Backspace: "#comfy-clear-button",
        d: "#comfy-load-default-button",
        g: "#comfy-group-selected-nodes-button"
      };
      const modifierKeybindId = modifierKeyIdMap[event.key];
      if (modifierPressed && modifierKeybindId) {
        event.preventDefault();
        const elem = document.querySelector(modifierKeybindId);
        elem.click();
        return;
      }
      if (event.ctrlKey || event.altKey || event.metaKey) {
        return;
      }
      if (event.key === "Escape") {
        const modals = document.querySelectorAll(".comfy-modal");
        const modal = Array.from(modals).find(
          (modal2) => window.getComputedStyle(modal2).getPropertyValue("display") !== "none"
        );
        if (modal) {
          modal.style.display = "none";
        }
        ;
        [...document.querySelectorAll("dialog")].forEach((d) => {
          d.close();
        });
      }
      const keyIdMap = {
        q: ".queue-tab-button.side-bar-button",
        h: ".queue-tab-button.side-bar-button",
        r: "#comfy-refresh-button"
      };
      const buttonId = keyIdMap[event.key];
      if (buttonId) {
        const button = document.querySelector(buttonId);
        button.click();
      }
    }, "keybindListener");
    window.addEventListener("keydown", keybindListener, true);
  }
});
const id$1 = "Comfy.LinkRenderMode";
const ext = {
  name: id$1,
  async setup(app2) {
    app2.ui.settings.addSetting({
      id: id$1,
      category: ["Comfy", "Graph", "LinkRenderMode"],
      name: "Link Render Mode",
      defaultValue: 2,
      type: "combo",
      // @ts-expect-error
      options: [...LiteGraph.LINK_RENDER_MODES, "Hidden"].map((m, i) => ({
        value: i,
        text: m,
        selected: i == app2.canvas.links_render_mode
      })),
      onChange(value) {
        app2.canvas.links_render_mode = +value;
        app2.graph.setDirtyCanvas(true);
      }
    });
  }
};
app.registerExtension(ext);
function dataURLToBlob(dataURL) {
  const parts = dataURL.split(";base64,");
  const contentType = parts[0].split(":")[1];
  const byteString = atob(parts[1]);
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);
  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }
  return new Blob([arrayBuffer], { type: contentType });
}
__name(dataURLToBlob, "dataURLToBlob");
function loadedImageToBlob(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const dataURL = canvas.toDataURL("image/png", 1);
  const blob = dataURLToBlob(dataURL);
  return blob;
}
__name(loadedImageToBlob, "loadedImageToBlob");
function loadImage(imagePath) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = function() {
      resolve(image);
    };
    image.src = imagePath;
  });
}
__name(loadImage, "loadImage");
async function uploadMask(filepath, formData) {
  await api.fetchApi("/upload/mask", {
    method: "POST",
    body: formData
  }).then((response) => {
  }).catch((error) => {
    console.error("Error:", error);
  });
  ComfyApp.clipspace.imgs[ComfyApp.clipspace["selectedIndex"]] = new Image();
  ComfyApp.clipspace.imgs[ComfyApp.clipspace["selectedIndex"]].src = api.apiURL(
    "/view?" + new URLSearchParams(filepath).toString() + app.getPreviewFormatParam() + app.getRandParam()
  );
  if (ComfyApp.clipspace.images)
    ComfyApp.clipspace.images[ComfyApp.clipspace["selectedIndex"]] = filepath;
  ClipspaceDialog.invalidatePreview();
}
__name(uploadMask, "uploadMask");
function prepare_mask(image, maskCanvas, maskCtx, maskColor) {
  maskCtx.drawImage(image, 0, 0, maskCanvas.width, maskCanvas.height);
  const maskData = maskCtx.getImageData(
    0,
    0,
    maskCanvas.width,
    maskCanvas.height
  );
  for (let i = 0; i < maskData.data.length; i += 4) {
    if (maskData.data[i + 3] == 255) maskData.data[i + 3] = 0;
    else maskData.data[i + 3] = 255;
    maskData.data[i] = maskColor.r;
    maskData.data[i + 1] = maskColor.g;
    maskData.data[i + 2] = maskColor.b;
  }
  maskCtx.globalCompositeOperation = "source-over";
  maskCtx.putImageData(maskData, 0, 0);
}
__name(prepare_mask, "prepare_mask");
class MaskEditorDialog extends ComfyDialog {
  static {
    __name(this, "MaskEditorDialog");
  }
  static instance = null;
  static mousedown_x = null;
  static mousedown_y = null;
  brush;
  maskCtx;
  maskCanvas;
  brush_size_slider;
  brush_opacity_slider;
  colorButton;
  saveButton;
  zoom_ratio;
  pan_x;
  pan_y;
  imgCanvas;
  last_display_style;
  is_visible;
  image;
  handler_registered;
  brush_slider_input;
  cursorX;
  cursorY;
  mousedown_pan_x;
  mousedown_pan_y;
  last_pressure;
  static getInstance() {
    if (!MaskEditorDialog.instance) {
      MaskEditorDialog.instance = new MaskEditorDialog();
    }
    return MaskEditorDialog.instance;
  }
  is_layout_created = false;
  constructor() {
    super();
    this.element = $el("div.comfy-modal", { parent: document.body }, [
      $el("div.comfy-modal-content", [...this.createButtons()])
    ]);
  }
  createButtons() {
    return [];
  }
  createButton(name, callback) {
    var button = document.createElement("button");
    button.style.pointerEvents = "auto";
    button.innerText = name;
    button.addEventListener("click", callback);
    return button;
  }
  createLeftButton(name, callback) {
    var button = this.createButton(name, callback);
    button.style.cssFloat = "left";
    button.style.marginRight = "4px";
    return button;
  }
  createRightButton(name, callback) {
    var button = this.createButton(name, callback);
    button.style.cssFloat = "right";
    button.style.marginLeft = "4px";
    return button;
  }
  createLeftSlider(self, name, callback) {
    const divElement = document.createElement("div");
    divElement.id = "maskeditor-slider";
    divElement.style.cssFloat = "left";
    divElement.style.fontFamily = "sans-serif";
    divElement.style.marginRight = "4px";
    divElement.style.color = "var(--input-text)";
    divElement.style.backgroundColor = "var(--comfy-input-bg)";
    divElement.style.borderRadius = "8px";
    divElement.style.borderColor = "var(--border-color)";
    divElement.style.borderStyle = "solid";
    divElement.style.fontSize = "15px";
    divElement.style.height = "21px";
    divElement.style.padding = "1px 6px";
    divElement.style.display = "flex";
    divElement.style.position = "relative";
    divElement.style.top = "2px";
    divElement.style.pointerEvents = "auto";
    self.brush_slider_input = document.createElement("input");
    self.brush_slider_input.setAttribute("type", "range");
    self.brush_slider_input.setAttribute("min", "1");
    self.brush_slider_input.setAttribute("max", "100");
    self.brush_slider_input.setAttribute("value", "10");
    const labelElement = document.createElement("label");
    labelElement.textContent = name;
    divElement.appendChild(labelElement);
    divElement.appendChild(self.brush_slider_input);
    self.brush_slider_input.addEventListener("change", callback);
    return divElement;
  }
  createOpacitySlider(self, name, callback) {
    const divElement = document.createElement("div");
    divElement.id = "maskeditor-opacity-slider";
    divElement.style.cssFloat = "left";
    divElement.style.fontFamily = "sans-serif";
    divElement.style.marginRight = "4px";
    divElement.style.color = "var(--input-text)";
    divElement.style.backgroundColor = "var(--comfy-input-bg)";
    divElement.style.borderRadius = "8px";
    divElement.style.borderColor = "var(--border-color)";
    divElement.style.borderStyle = "solid";
    divElement.style.fontSize = "15px";
    divElement.style.height = "21px";
    divElement.style.padding = "1px 6px";
    divElement.style.display = "flex";
    divElement.style.position = "relative";
    divElement.style.top = "2px";
    divElement.style.pointerEvents = "auto";
    self.opacity_slider_input = document.createElement("input");
    self.opacity_slider_input.setAttribute("type", "range");
    self.opacity_slider_input.setAttribute("min", "0.1");
    self.opacity_slider_input.setAttribute("max", "1.0");
    self.opacity_slider_input.setAttribute("step", "0.01");
    self.opacity_slider_input.setAttribute("value", "0.7");
    const labelElement = document.createElement("label");
    labelElement.textContent = name;
    divElement.appendChild(labelElement);
    divElement.appendChild(self.opacity_slider_input);
    self.opacity_slider_input.addEventListener("input", callback);
    return divElement;
  }
  setlayout(imgCanvas, maskCanvas) {
    const self = this;
    var bottom_panel = document.createElement("div");
    bottom_panel.style.position = "absolute";
    bottom_panel.style.bottom = "0px";
    bottom_panel.style.left = "20px";
    bottom_panel.style.right = "20px";
    bottom_panel.style.height = "50px";
    bottom_panel.style.pointerEvents = "none";
    var brush = document.createElement("div");
    brush.id = "brush";
    brush.style.backgroundColor = "transparent";
    brush.style.outline = "1px dashed black";
    brush.style.boxShadow = "0 0 0 1px white";
    brush.style.borderRadius = "50%";
    brush.style.MozBorderRadius = "50%";
    brush.style.WebkitBorderRadius = "50%";
    brush.style.position = "absolute";
    brush.style.zIndex = "8889";
    brush.style.pointerEvents = "none";
    this.brush = brush;
    this.element.appendChild(imgCanvas);
    this.element.appendChild(maskCanvas);
    this.element.appendChild(bottom_panel);
    document.body.appendChild(brush);
    var clearButton = this.createLeftButton("Clear", () => {
      self.maskCtx.clearRect(
        0,
        0,
        self.maskCanvas.width,
        self.maskCanvas.height
      );
    });
    this.brush_size_slider = this.createLeftSlider(
      self,
      "Thickness",
      (event) => {
        self.brush_size = event.target.value;
        self.updateBrushPreview(self);
      }
    );
    this.brush_opacity_slider = this.createOpacitySlider(
      self,
      "Opacity",
      (event) => {
        self.brush_opacity = event.target.value;
        if (self.brush_color_mode !== "negative") {
          self.maskCanvas.style.opacity = self.brush_opacity.toString();
        }
      }
    );
    this.colorButton = this.createLeftButton(this.getColorButtonText(), () => {
      if (self.brush_color_mode === "black") {
        self.brush_color_mode = "white";
      } else if (self.brush_color_mode === "white") {
        self.brush_color_mode = "negative";
      } else {
        self.brush_color_mode = "black";
      }
      self.updateWhenBrushColorModeChanged();
    });
    var cancelButton = this.createRightButton("Cancel", () => {
      document.removeEventListener("keydown", MaskEditorDialog.handleKeyDown);
      self.close();
    });
    this.saveButton = this.createRightButton("Save", () => {
      document.removeEventListener("keydown", MaskEditorDialog.handleKeyDown);
      self.save();
    });
    this.element.appendChild(imgCanvas);
    this.element.appendChild(maskCanvas);
    this.element.appendChild(bottom_panel);
    bottom_panel.appendChild(clearButton);
    bottom_panel.appendChild(this.saveButton);
    bottom_panel.appendChild(cancelButton);
    bottom_panel.appendChild(this.brush_size_slider);
    bottom_panel.appendChild(this.brush_opacity_slider);
    bottom_panel.appendChild(this.colorButton);
    imgCanvas.style.position = "absolute";
    maskCanvas.style.position = "absolute";
    imgCanvas.style.top = "200";
    imgCanvas.style.left = "0";
    maskCanvas.style.top = imgCanvas.style.top;
    maskCanvas.style.left = imgCanvas.style.left;
    const maskCanvasStyle = this.getMaskCanvasStyle();
    maskCanvas.style.mixBlendMode = maskCanvasStyle.mixBlendMode;
    maskCanvas.style.opacity = maskCanvasStyle.opacity.toString();
  }
  async show() {
    this.zoom_ratio = 1;
    this.pan_x = 0;
    this.pan_y = 0;
    if (!this.is_layout_created) {
      const imgCanvas = document.createElement("canvas");
      const maskCanvas = document.createElement("canvas");
      imgCanvas.id = "imageCanvas";
      maskCanvas.id = "maskCanvas";
      this.setlayout(imgCanvas, maskCanvas);
      this.imgCanvas = imgCanvas;
      this.maskCanvas = maskCanvas;
      this.maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
      this.setEventHandler(maskCanvas);
      this.is_layout_created = true;
      const self = this;
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          if (mutation.type === "attributes" && mutation.attributeName === "style") {
            if (self.last_display_style && self.last_display_style != "none" && self.element.style.display == "none") {
              self.brush.style.display = "none";
              ComfyApp.onClipspaceEditorClosed();
            }
            self.last_display_style = self.element.style.display;
          }
        });
      });
      const config = { attributes: true };
      observer.observe(this.element, config);
    }
    document.addEventListener("keydown", MaskEditorDialog.handleKeyDown);
    if (ComfyApp.clipspace_return_node) {
      this.saveButton.innerText = "Save to node";
    } else {
      this.saveButton.innerText = "Save";
    }
    this.saveButton.disabled = false;
    this.element.style.display = "block";
    this.element.style.width = "85%";
    this.element.style.margin = "0 7.5%";
    this.element.style.height = "100vh";
    this.element.style.top = "50%";
    this.element.style.left = "42%";
    this.element.style.zIndex = "8888";
    await this.setImages(this.imgCanvas);
    this.is_visible = true;
  }
  isOpened() {
    return this.element.style.display == "block";
  }
  invalidateCanvas(orig_image, mask_image) {
    this.imgCanvas.width = orig_image.width;
    this.imgCanvas.height = orig_image.height;
    this.maskCanvas.width = orig_image.width;
    this.maskCanvas.height = orig_image.height;
    let imgCtx = this.imgCanvas.getContext("2d", { willReadFrequently: true });
    let maskCtx = this.maskCanvas.getContext("2d", {
      willReadFrequently: true
    });
    imgCtx.drawImage(orig_image, 0, 0, orig_image.width, orig_image.height);
    prepare_mask(mask_image, this.maskCanvas, maskCtx, this.getMaskColor());
  }
  async setImages(imgCanvas) {
    let self = this;
    const imgCtx = imgCanvas.getContext("2d", { willReadFrequently: true });
    const maskCtx = this.maskCtx;
    const maskCanvas = this.maskCanvas;
    imgCtx.clearRect(0, 0, this.imgCanvas.width, this.imgCanvas.height);
    maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    const filepath = ComfyApp.clipspace.images;
    const alpha_url = new URL(
      ComfyApp.clipspace.imgs[ComfyApp.clipspace["selectedIndex"]].src
    );
    alpha_url.searchParams.delete("channel");
    alpha_url.searchParams.delete("preview");
    alpha_url.searchParams.set("channel", "a");
    let mask_image = await loadImage(alpha_url);
    const rgb_url = new URL(
      ComfyApp.clipspace.imgs[ComfyApp.clipspace["selectedIndex"]].src
    );
    rgb_url.searchParams.delete("channel");
    rgb_url.searchParams.set("channel", "rgb");
    this.image = new Image();
    this.image.onload = function() {
      maskCanvas.width = self.image.width;
      maskCanvas.height = self.image.height;
      self.invalidateCanvas(self.image, mask_image);
      self.initializeCanvasPanZoom();
    };
    this.image.src = rgb_url.toString();
  }
  initializeCanvasPanZoom() {
    let drawWidth = this.image.width;
    let drawHeight = this.image.height;
    let width = this.element.clientWidth;
    let height = this.element.clientHeight;
    if (this.image.width > width) {
      drawWidth = width;
      drawHeight = drawWidth / this.image.width * this.image.height;
    }
    if (drawHeight > height) {
      drawHeight = height;
      drawWidth = drawHeight / this.image.height * this.image.width;
    }
    this.zoom_ratio = drawWidth / this.image.width;
    const canvasX = (width - drawWidth) / 2;
    const canvasY = (height - drawHeight) / 2;
    this.pan_x = canvasX;
    this.pan_y = canvasY;
    this.invalidatePanZoom();
  }
  invalidatePanZoom() {
    let raw_width = this.image.width * this.zoom_ratio;
    let raw_height = this.image.height * this.zoom_ratio;
    if (this.pan_x + raw_width < 10) {
      this.pan_x = 10 - raw_width;
    }
    if (this.pan_y + raw_height < 10) {
      this.pan_y = 10 - raw_height;
    }
    let width = `${raw_width}px`;
    let height = `${raw_height}px`;
    let left = `${this.pan_x}px`;
    let top = `${this.pan_y}px`;
    this.maskCanvas.style.width = width;
    this.maskCanvas.style.height = height;
    this.maskCanvas.style.left = left;
    this.maskCanvas.style.top = top;
    this.imgCanvas.style.width = width;
    this.imgCanvas.style.height = height;
    this.imgCanvas.style.left = left;
    this.imgCanvas.style.top = top;
  }
  setEventHandler(maskCanvas) {
    const self = this;
    if (!this.handler_registered) {
      maskCanvas.addEventListener("contextmenu", (event) => {
        event.preventDefault();
      });
      this.element.addEventListener(
        "wheel",
        (event) => this.handleWheelEvent(self, event)
      );
      this.element.addEventListener(
        "pointermove",
        (event) => this.pointMoveEvent(self, event)
      );
      this.element.addEventListener(
        "touchmove",
        (event) => this.pointMoveEvent(self, event)
      );
      this.element.addEventListener("dragstart", (event) => {
        if (event.ctrlKey) {
          event.preventDefault();
        }
      });
      maskCanvas.addEventListener(
        "pointerdown",
        (event) => this.handlePointerDown(self, event)
      );
      maskCanvas.addEventListener(
        "pointermove",
        (event) => this.draw_move(self, event)
      );
      maskCanvas.addEventListener(
        "touchmove",
        (event) => this.draw_move(self, event)
      );
      maskCanvas.addEventListener("pointerover", (event) => {
        this.brush.style.display = "block";
      });
      maskCanvas.addEventListener("pointerleave", (event) => {
        this.brush.style.display = "none";
      });
      document.addEventListener("pointerup", MaskEditorDialog.handlePointerUp);
      this.handler_registered = true;
    }
  }
  getMaskCanvasStyle() {
    if (this.brush_color_mode === "negative") {
      return {
        mixBlendMode: "difference",
        opacity: "1"
      };
    } else {
      return {
        mixBlendMode: "initial",
        opacity: this.brush_opacity
      };
    }
  }
  getMaskColor() {
    if (this.brush_color_mode === "black") {
      return { r: 0, g: 0, b: 0 };
    }
    if (this.brush_color_mode === "white") {
      return { r: 255, g: 255, b: 255 };
    }
    if (this.brush_color_mode === "negative") {
      return { r: 255, g: 255, b: 255 };
    }
    return { r: 0, g: 0, b: 0 };
  }
  getMaskFillStyle() {
    const maskColor = this.getMaskColor();
    return "rgb(" + maskColor.r + "," + maskColor.g + "," + maskColor.b + ")";
  }
  getColorButtonText() {
    let colorCaption = "unknown";
    if (this.brush_color_mode === "black") {
      colorCaption = "black";
    } else if (this.brush_color_mode === "white") {
      colorCaption = "white";
    } else if (this.brush_color_mode === "negative") {
      colorCaption = "negative";
    }
    return "Color: " + colorCaption;
  }
  updateWhenBrushColorModeChanged() {
    this.colorButton.innerText = this.getColorButtonText();
    const maskCanvasStyle = this.getMaskCanvasStyle();
    this.maskCanvas.style.mixBlendMode = maskCanvasStyle.mixBlendMode;
    this.maskCanvas.style.opacity = maskCanvasStyle.opacity.toString();
    const maskColor = this.getMaskColor();
    const maskData = this.maskCtx.getImageData(
      0,
      0,
      this.maskCanvas.width,
      this.maskCanvas.height
    );
    for (let i = 0; i < maskData.data.length; i += 4) {
      maskData.data[i] = maskColor.r;
      maskData.data[i + 1] = maskColor.g;
      maskData.data[i + 2] = maskColor.b;
    }
    this.maskCtx.putImageData(maskData, 0, 0);
  }
  brush_opacity = 0.7;
  brush_size = 10;
  brush_color_mode = "black";
  drawing_mode = false;
  lastx = -1;
  lasty = -1;
  lasttime = 0;
  static handleKeyDown(event) {
    const self = MaskEditorDialog.instance;
    if (event.key === "]") {
      self.brush_size = Math.min(self.brush_size + 2, 100);
      self.brush_slider_input.value = self.brush_size;
    } else if (event.key === "[") {
      self.brush_size = Math.max(self.brush_size - 2, 1);
      self.brush_slider_input.value = self.brush_size;
    } else if (event.key === "Enter") {
      self.save();
    }
    self.updateBrushPreview(self);
  }
  static handlePointerUp(event) {
    event.preventDefault();
    this.mousedown_x = null;
    this.mousedown_y = null;
    MaskEditorDialog.instance.drawing_mode = false;
  }
  updateBrushPreview(self) {
    const brush = self.brush;
    var centerX = self.cursorX;
    var centerY = self.cursorY;
    brush.style.width = self.brush_size * 2 * this.zoom_ratio + "px";
    brush.style.height = self.brush_size * 2 * this.zoom_ratio + "px";
    brush.style.left = centerX - self.brush_size * this.zoom_ratio + "px";
    brush.style.top = centerY - self.brush_size * this.zoom_ratio + "px";
  }
  handleWheelEvent(self, event) {
    event.preventDefault();
    if (event.ctrlKey) {
      if (event.deltaY < 0) {
        this.zoom_ratio = Math.min(10, this.zoom_ratio + 0.2);
      } else {
        this.zoom_ratio = Math.max(0.2, this.zoom_ratio - 0.2);
      }
      this.invalidatePanZoom();
    } else {
      if (event.deltaY < 0) this.brush_size = Math.min(this.brush_size + 2, 100);
      else this.brush_size = Math.max(this.brush_size - 2, 1);
      this.brush_slider_input.value = this.brush_size.toString();
      this.updateBrushPreview(this);
    }
  }
  pointMoveEvent(self, event) {
    this.cursorX = event.pageX;
    this.cursorY = event.pageY;
    self.updateBrushPreview(self);
    if (event.ctrlKey) {
      event.preventDefault();
      self.pan_move(self, event);
    }
    let left_button_down = window.TouchEvent && event instanceof TouchEvent || event.buttons == 1;
    if (event.shiftKey && left_button_down) {
      self.drawing_mode = false;
      const y = event.clientY;
      let delta = (self.zoom_lasty - y) * 5e-3;
      self.zoom_ratio = Math.max(
        Math.min(10, self.last_zoom_ratio - delta),
        0.2
      );
      this.invalidatePanZoom();
      return;
    }
  }
  pan_move(self, event) {
    if (event.buttons == 1) {
      if (MaskEditorDialog.mousedown_x) {
        let deltaX = MaskEditorDialog.mousedown_x - event.clientX;
        let deltaY = MaskEditorDialog.mousedown_y - event.clientY;
        self.pan_x = this.mousedown_pan_x - deltaX;
        self.pan_y = this.mousedown_pan_y - deltaY;
        self.invalidatePanZoom();
      }
    }
  }
  draw_move(self, event) {
    if (event.ctrlKey || event.shiftKey) {
      return;
    }
    event.preventDefault();
    this.cursorX = event.pageX;
    this.cursorY = event.pageY;
    self.updateBrushPreview(self);
    let left_button_down = window.TouchEvent && event instanceof TouchEvent || event.buttons == 1;
    let right_button_down = [2, 5, 32].includes(event.buttons);
    if (!event.altKey && left_button_down) {
      var diff = performance.now() - self.lasttime;
      const maskRect = self.maskCanvas.getBoundingClientRect();
      var x = event.offsetX;
      var y = event.offsetY;
      if (event.offsetX == null) {
        x = event.targetTouches[0].clientX - maskRect.left;
      }
      if (event.offsetY == null) {
        y = event.targetTouches[0].clientY - maskRect.top;
      }
      x /= self.zoom_ratio;
      y /= self.zoom_ratio;
      var brush_size = this.brush_size;
      if (event instanceof PointerEvent && event.pointerType == "pen") {
        brush_size *= event.pressure;
        this.last_pressure = event.pressure;
      } else if (window.TouchEvent && event instanceof TouchEvent && diff < 20) {
        brush_size *= this.last_pressure;
      } else {
        brush_size = this.brush_size;
      }
      if (diff > 20 && !this.drawing_mode)
        requestAnimationFrame(() => {
          self.maskCtx.beginPath();
          self.maskCtx.fillStyle = this.getMaskFillStyle();
          self.maskCtx.globalCompositeOperation = "source-over";
          self.maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
          self.maskCtx.fill();
          self.lastx = x;
          self.lasty = y;
        });
      else
        requestAnimationFrame(() => {
          self.maskCtx.beginPath();
          self.maskCtx.fillStyle = this.getMaskFillStyle();
          self.maskCtx.globalCompositeOperation = "source-over";
          var dx = x - self.lastx;
          var dy = y - self.lasty;
          var distance = Math.sqrt(dx * dx + dy * dy);
          var directionX = dx / distance;
          var directionY = dy / distance;
          for (var i = 0; i < distance; i += 5) {
            var px = self.lastx + directionX * i;
            var py = self.lasty + directionY * i;
            self.maskCtx.arc(px, py, brush_size, 0, Math.PI * 2, false);
            self.maskCtx.fill();
          }
          self.lastx = x;
          self.lasty = y;
        });
      self.lasttime = performance.now();
    } else if (event.altKey && left_button_down || right_button_down) {
      const maskRect = self.maskCanvas.getBoundingClientRect();
      const x2 = (event.offsetX || event.targetTouches[0].clientX - maskRect.left) / self.zoom_ratio;
      const y2 = (event.offsetY || event.targetTouches[0].clientY - maskRect.top) / self.zoom_ratio;
      var brush_size = this.brush_size;
      if (event instanceof PointerEvent && event.pointerType == "pen") {
        brush_size *= event.pressure;
        this.last_pressure = event.pressure;
      } else if (window.TouchEvent && event instanceof TouchEvent && diff < 20) {
        brush_size *= this.last_pressure;
      } else {
        brush_size = this.brush_size;
      }
      if (diff > 20 && !this.drawing_mode)
        requestAnimationFrame(() => {
          self.maskCtx.beginPath();
          self.maskCtx.globalCompositeOperation = "destination-out";
          self.maskCtx.arc(x2, y2, brush_size, 0, Math.PI * 2, false);
          self.maskCtx.fill();
          self.lastx = x2;
          self.lasty = y2;
        });
      else
        requestAnimationFrame(() => {
          self.maskCtx.beginPath();
          self.maskCtx.globalCompositeOperation = "destination-out";
          var dx = x2 - self.lastx;
          var dy = y2 - self.lasty;
          var distance = Math.sqrt(dx * dx + dy * dy);
          var directionX = dx / distance;
          var directionY = dy / distance;
          for (var i = 0; i < distance; i += 5) {
            var px = self.lastx + directionX * i;
            var py = self.lasty + directionY * i;
            self.maskCtx.arc(px, py, brush_size, 0, Math.PI * 2, false);
            self.maskCtx.fill();
          }
          self.lastx = x2;
          self.lasty = y2;
        });
      self.lasttime = performance.now();
    }
  }
  handlePointerDown(self, event) {
    if (event.ctrlKey) {
      if (event.buttons == 1) {
        MaskEditorDialog.mousedown_x = event.clientX;
        MaskEditorDialog.mousedown_y = event.clientY;
        this.mousedown_pan_x = this.pan_x;
        this.mousedown_pan_y = this.pan_y;
      }
      return;
    }
    var brush_size = this.brush_size;
    if (event instanceof PointerEvent && event.pointerType == "pen") {
      brush_size *= event.pressure;
      this.last_pressure = event.pressure;
    }
    if ([0, 2, 5].includes(event.button)) {
      self.drawing_mode = true;
      event.preventDefault();
      if (event.shiftKey) {
        self.zoom_lasty = event.clientY;
        self.last_zoom_ratio = self.zoom_ratio;
        return;
      }
      const maskRect = self.maskCanvas.getBoundingClientRect();
      const x = (event.offsetX || event.targetTouches[0].clientX - maskRect.left) / self.zoom_ratio;
      const y = (event.offsetY || event.targetTouches[0].clientY - maskRect.top) / self.zoom_ratio;
      self.maskCtx.beginPath();
      if (!event.altKey && event.button == 0) {
        self.maskCtx.fillStyle = this.getMaskFillStyle();
        self.maskCtx.globalCompositeOperation = "source-over";
      } else {
        self.maskCtx.globalCompositeOperation = "destination-out";
      }
      self.maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
      self.maskCtx.fill();
      self.lastx = x;
      self.lasty = y;
      self.lasttime = performance.now();
    }
  }
  async save() {
    const backupCanvas = document.createElement("canvas");
    const backupCtx = backupCanvas.getContext("2d", {
      willReadFrequently: true
    });
    backupCanvas.width = this.image.width;
    backupCanvas.height = this.image.height;
    backupCtx.clearRect(0, 0, backupCanvas.width, backupCanvas.height);
    backupCtx.drawImage(
      this.maskCanvas,
      0,
      0,
      this.maskCanvas.width,
      this.maskCanvas.height,
      0,
      0,
      backupCanvas.width,
      backupCanvas.height
    );
    const backupData = backupCtx.getImageData(
      0,
      0,
      backupCanvas.width,
      backupCanvas.height
    );
    for (let i = 0; i < backupData.data.length; i += 4) {
      if (backupData.data[i + 3] == 255) backupData.data[i + 3] = 0;
      else backupData.data[i + 3] = 255;
      backupData.data[i] = 0;
      backupData.data[i + 1] = 0;
      backupData.data[i + 2] = 0;
    }
    backupCtx.globalCompositeOperation = "source-over";
    backupCtx.putImageData(backupData, 0, 0);
    const formData = new FormData();
    const filename = "clipspace-mask-" + performance.now() + ".png";
    const item = {
      filename,
      subfolder: "clipspace",
      type: "input"
    };
    if (ComfyApp.clipspace.images) ComfyApp.clipspace.images[0] = item;
    if (ComfyApp.clipspace.widgets) {
      const index = ComfyApp.clipspace.widgets.findIndex(
        (obj) => obj.name === "image"
      );
      if (index >= 0) ComfyApp.clipspace.widgets[index].value = item;
    }
    const dataURL = backupCanvas.toDataURL();
    const blob = dataURLToBlob(dataURL);
    let original_url = new URL(this.image.src);
    const original_ref = {
      filename: original_url.searchParams.get("filename")
    };
    let original_subfolder = original_url.searchParams.get("subfolder");
    if (original_subfolder) original_ref.subfolder = original_subfolder;
    let original_type = original_url.searchParams.get("type");
    if (original_type) original_ref.type = original_type;
    formData.append("image", blob, filename);
    formData.append("original_ref", JSON.stringify(original_ref));
    formData.append("type", "input");
    formData.append("subfolder", "clipspace");
    this.saveButton.innerText = "Saving...";
    this.saveButton.disabled = true;
    await uploadMask(item, formData);
    ComfyApp.onClipspaceEditorSave();
    this.close();
  }
}
app.registerExtension({
  name: "Comfy.MaskEditor",
  init(app2) {
    ComfyApp.open_maskeditor = function() {
      const dlg = MaskEditorDialog.getInstance();
      if (!dlg.isOpened()) {
        dlg.show();
      }
    };
    const context_predicate = /* @__PURE__ */ __name(() => ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0, "context_predicate");
    ClipspaceDialog.registerButton(
      "MaskEditor",
      context_predicate,
      ComfyApp.open_maskeditor
    );
  }
});
const id = "Comfy.NodeTemplates";
const file = "comfy.templates.json";
class ManageTemplates extends ComfyDialog {
  static {
    __name(this, "ManageTemplates");
  }
  templates;
  draggedEl;
  saveVisualCue;
  emptyImg;
  importInput;
  constructor() {
    super();
    this.load().then((v) => {
      this.templates = v;
    });
    this.element.classList.add("comfy-manage-templates");
    this.draggedEl = null;
    this.saveVisualCue = null;
    this.emptyImg = new Image();
    this.emptyImg.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=";
    this.importInput = $el("input", {
      type: "file",
      accept: ".json",
      multiple: true,
      style: { display: "none" },
      parent: document.body,
      onchange: /* @__PURE__ */ __name(() => this.importAll(), "onchange")
    });
  }
  createButtons() {
    const btns = super.createButtons();
    btns[0].textContent = "Close";
    btns[0].onclick = (e) => {
      clearTimeout(this.saveVisualCue);
      this.close();
    };
    btns.unshift(
      $el("button", {
        type: "button",
        textContent: "Export",
        onclick: /* @__PURE__ */ __name(() => this.exportAll(), "onclick")
      })
    );
    btns.unshift(
      $el("button", {
        type: "button",
        textContent: "Import",
        onclick: /* @__PURE__ */ __name(() => {
          this.importInput.click();
        }, "onclick")
      })
    );
    return btns;
  }
  async load() {
    let templates = [];
    if (app.storageLocation === "server") {
      if (app.isNewUserSession) {
        const json = localStorage.getItem(id);
        if (json) {
          templates = JSON.parse(json);
        }
        await api.storeUserData(file, json, { stringify: false });
      } else {
        const res = await api.getUserData(file);
        if (res.status === 200) {
          try {
            templates = await res.json();
          } catch (error) {
          }
        } else if (res.status !== 404) {
          console.error(res.status + " " + res.statusText);
        }
      }
    } else {
      const json = localStorage.getItem(id);
      if (json) {
        templates = JSON.parse(json);
      }
    }
    return templates ?? [];
  }
  async store() {
    if (app.storageLocation === "server") {
      const templates = JSON.stringify(this.templates, void 0, 4);
      localStorage.setItem(id, templates);
      try {
        await api.storeUserData(file, templates, { stringify: false });
      } catch (error) {
        console.error(error);
        alert(error.message);
      }
    } else {
      localStorage.setItem(id, JSON.stringify(this.templates));
    }
  }
  async importAll() {
    for (const file2 of this.importInput.files) {
      if (file2.type === "application/json" || file2.name.endsWith(".json")) {
        const reader = new FileReader();
        reader.onload = async () => {
          const importFile = JSON.parse(reader.result);
          if (importFile?.templates) {
            for (const template of importFile.templates) {
              if (template?.name && template?.data) {
                this.templates.push(template);
              }
            }
            await this.store();
          }
        };
        await reader.readAsText(file2);
      }
    }
    this.importInput.value = null;
    this.close();
  }
  exportAll() {
    if (this.templates.length == 0) {
      alert("No templates to export.");
      return;
    }
    const json = JSON.stringify({ templates: this.templates }, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = $el("a", {
      href: url,
      download: "node_templates.json",
      style: { display: "none" },
      parent: document.body
    });
    a.click();
    setTimeout(function() {
      a.remove();
      window.URL.revokeObjectURL(url);
    }, 0);
  }
  show() {
    super.show(
      $el(
        "div",
        {},
        this.templates.flatMap((t, i) => {
          let nameInput;
          return [
            $el(
              "div",
              {
                dataset: { id: i.toString() },
                className: "templateManagerRow",
                style: {
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  border: "1px dashed transparent",
                  gap: "5px",
                  backgroundColor: "var(--comfy-menu-bg)"
                },
                ondragstart: /* @__PURE__ */ __name((e) => {
                  this.draggedEl = e.currentTarget;
                  e.currentTarget.style.opacity = "0.6";
                  e.currentTarget.style.border = "1px dashed yellow";
                  e.dataTransfer.effectAllowed = "move";
                  e.dataTransfer.setDragImage(this.emptyImg, 0, 0);
                }, "ondragstart"),
                ondragend: /* @__PURE__ */ __name((e) => {
                  e.target.style.opacity = "1";
                  e.currentTarget.style.border = "1px dashed transparent";
                  e.currentTarget.removeAttribute("draggable");
                  this.element.querySelectorAll(".templateManagerRow").forEach((el, i2) => {
                    var prev_i = Number.parseInt(el.dataset.id);
                    if (el == this.draggedEl && prev_i != i2) {
                      this.templates.splice(
                        i2,
                        0,
                        this.templates.splice(prev_i, 1)[0]
                      );
                    }
                    el.dataset.id = i2.toString();
                  });
                  this.store();
                }, "ondragend"),
                ondragover: /* @__PURE__ */ __name((e) => {
                  e.preventDefault();
                  if (e.currentTarget == this.draggedEl) return;
                  let rect = e.currentTarget.getBoundingClientRect();
                  if (e.clientY > rect.top + rect.height / 2) {
                    e.currentTarget.parentNode.insertBefore(
                      this.draggedEl,
                      e.currentTarget.nextSibling
                    );
                  } else {
                    e.currentTarget.parentNode.insertBefore(
                      this.draggedEl,
                      e.currentTarget
                    );
                  }
                }, "ondragover")
              },
              [
                $el(
                  "label",
                  {
                    textContent: "Name: ",
                    style: {
                      cursor: "grab"
                    },
                    onmousedown: /* @__PURE__ */ __name((e) => {
                      if (e.target.localName == "label")
                        e.currentTarget.parentNode.draggable = "true";
                    }, "onmousedown")
                  },
                  [
                    $el("input", {
                      value: t.name,
                      dataset: { name: t.name },
                      style: {
                        transitionProperty: "background-color",
                        transitionDuration: "0s"
                      },
                      onchange: /* @__PURE__ */ __name((e) => {
                        clearTimeout(this.saveVisualCue);
                        var el = e.target;
                        var row = el.parentNode.parentNode;
                        this.templates[row.dataset.id].name = el.value.trim() || "untitled";
                        this.store();
                        el.style.backgroundColor = "rgb(40, 95, 40)";
                        el.style.transitionDuration = "0s";
                        this.saveVisualCue = setTimeout(function() {
                          el.style.transitionDuration = ".7s";
                          el.style.backgroundColor = "var(--comfy-input-bg)";
                        }, 15);
                      }, "onchange"),
                      onkeypress: /* @__PURE__ */ __name((e) => {
                        var el = e.target;
                        clearTimeout(this.saveVisualCue);
                        el.style.transitionDuration = "0s";
                        el.style.backgroundColor = "var(--comfy-input-bg)";
                      }, "onkeypress"),
                      $: /* @__PURE__ */ __name((el) => nameInput = el, "$")
                    })
                  ]
                ),
                $el("div", {}, [
                  $el("button", {
                    textContent: "Export",
                    style: {
                      fontSize: "12px",
                      fontWeight: "normal"
                    },
                    onclick: /* @__PURE__ */ __name((e) => {
                      const json = JSON.stringify({ templates: [t] }, null, 2);
                      const blob = new Blob([json], {
                        type: "application/json"
                      });
                      const url = URL.createObjectURL(blob);
                      const a = $el("a", {
                        href: url,
                        download: (nameInput.value || t.name) + ".json",
                        style: { display: "none" },
                        parent: document.body
                      });
                      a.click();
                      setTimeout(function() {
                        a.remove();
                        window.URL.revokeObjectURL(url);
                      }, 0);
                    }, "onclick")
                  }),
                  $el("button", {
                    textContent: "Delete",
                    style: {
                      fontSize: "12px",
                      color: "red",
                      fontWeight: "normal"
                    },
                    onclick: /* @__PURE__ */ __name((e) => {
                      const item = e.target.parentNode.parentNode;
                      item.parentNode.removeChild(item);
                      this.templates.splice(item.dataset.id * 1, 1);
                      this.store();
                      var that = this;
                      setTimeout(function() {
                        that.element.querySelectorAll(".templateManagerRow").forEach((el, i2) => {
                          el.dataset.id = i2.toString();
                        });
                      }, 0);
                    }, "onclick")
                  })
                ])
              ]
            )
          ];
        })
      )
    );
  }
}
app.registerExtension({
  name: id,
  setup() {
    const manage = new ManageTemplates();
    const clipboardAction = /* @__PURE__ */ __name(async (cb) => {
      const old = localStorage.getItem("litegrapheditor_clipboard");
      await cb();
      localStorage.setItem("litegrapheditor_clipboard", old);
    }, "clipboardAction");
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function() {
      const options = orig.apply(this, arguments);
      options.push(null);
      options.push({
        content: `Save Selected as Template`,
        disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
        callback: /* @__PURE__ */ __name(() => {
          const name = prompt("Enter name");
          if (!name?.trim()) return;
          clipboardAction(() => {
            app.canvas.copyToClipboard();
            let data = localStorage.getItem("litegrapheditor_clipboard");
            data = JSON.parse(data);
            const nodeIds = Object.keys(app.canvas.selected_nodes);
            for (let i = 0; i < nodeIds.length; i++) {
              const node = app.graph.getNodeById(nodeIds[i]);
              const nodeData = node?.constructor.nodeData;
              let groupData = GroupNodeHandler.getGroupData(node);
              if (groupData) {
                groupData = groupData.nodeData;
                if (!data.groupNodes) {
                  data.groupNodes = {};
                }
                data.groupNodes[nodeData.name] = groupData;
                data.nodes[i].type = nodeData.name;
              }
            }
            manage.templates.push({
              name,
              data: JSON.stringify(data)
            });
            manage.store();
          });
        }, "callback")
      });
      const subItems = manage.templates.map((t) => {
        return {
          content: t.name,
          callback: /* @__PURE__ */ __name(() => {
            clipboardAction(async () => {
              const data = JSON.parse(t.data);
              await GroupNodeConfig.registerFromWorkflow(data.groupNodes, {});
              localStorage.setItem("litegrapheditor_clipboard", t.data);
              app.canvas.pasteFromClipboard();
            });
          }, "callback")
        };
      });
      subItems.push(null, {
        content: "Manage",
        callback: /* @__PURE__ */ __name(() => manage.show(), "callback")
      });
      options.push({
        content: "Node Templates",
        submenu: {
          options: subItems
        }
      });
      return options;
    };
  }
});
app.registerExtension({
  name: "Comfy.NoteNode",
  registerCustomNodes() {
    class NoteNode extends LGraphNode {
      static {
        __name(this, "NoteNode");
      }
      static category;
      color = LGraphCanvas.node_colors.yellow.color;
      bgcolor = LGraphCanvas.node_colors.yellow.bgcolor;
      groupcolor = LGraphCanvas.node_colors.yellow.groupcolor;
      isVirtualNode;
      collapsable;
      title_mode;
      constructor(title) {
        super(title);
        if (!this.properties) {
          this.properties = { text: "" };
        }
        ComfyWidgets.STRING(
          // Should we extends LGraphNode?  Yesss
          this,
          "",
          // @ts-expect-error
          ["", { default: this.properties.text, multiline: true }],
          app
        );
        this.serialize_widgets = true;
        this.isVirtualNode = true;
      }
    }
    LiteGraph.registerNodeType(
      "Note",
      Object.assign(NoteNode, {
        title_mode: LiteGraph.NORMAL_TITLE,
        title: "Note",
        collapsable: true
      })
    );
    NoteNode.category = "utils";
  }
});
app.registerExtension({
  name: "Comfy.RerouteNode",
  registerCustomNodes(app2) {
    class RerouteNode extends LGraphNode {
      static {
        __name(this, "RerouteNode");
      }
      static category;
      static defaultVisibility = false;
      constructor(title) {
        super(title);
        if (!this.properties) {
          this.properties = {};
        }
        this.properties.showOutputText = RerouteNode.defaultVisibility;
        this.properties.horizontal = false;
        this.addInput("", "*");
        this.addOutput(this.properties.showOutputText ? "*" : "", "*");
        this.onAfterGraphConfigured = function() {
          requestAnimationFrame(() => {
            this.onConnectionsChange(LiteGraph.INPUT, null, true, null);
          });
        };
        this.onConnectionsChange = function(type, index, connected, link_info) {
          this.applyOrientation();
          if (connected && type === LiteGraph.OUTPUT) {
            const types = new Set(
              this.outputs[0].links.map((l) => app2.graph.links[l].type).filter((t) => t !== "*")
            );
            if (types.size > 1) {
              const linksToDisconnect = [];
              for (let i = 0; i < this.outputs[0].links.length - 1; i++) {
                const linkId = this.outputs[0].links[i];
                const link = app2.graph.links[linkId];
                linksToDisconnect.push(link);
              }
              for (const link of linksToDisconnect) {
                const node = app2.graph.getNodeById(link.target_id);
                node.disconnectInput(link.target_slot);
              }
            }
          }
          let currentNode = this;
          let updateNodes = [];
          let inputType = null;
          let inputNode = null;
          while (currentNode) {
            updateNodes.unshift(currentNode);
            const linkId = currentNode.inputs[0].link;
            if (linkId !== null) {
              const link = app2.graph.links[linkId];
              if (!link) return;
              const node = app2.graph.getNodeById(link.origin_id);
              const type2 = node.constructor.type;
              if (type2 === "Reroute") {
                if (node === this) {
                  currentNode.disconnectInput(link.target_slot);
                  currentNode = null;
                } else {
                  currentNode = node;
                }
              } else {
                inputNode = currentNode;
                inputType = node.outputs[link.origin_slot]?.type ?? null;
                break;
              }
            } else {
              currentNode = null;
              break;
            }
          }
          const nodes = [this];
          let outputType = null;
          while (nodes.length) {
            currentNode = nodes.pop();
            const outputs = (currentNode.outputs ? currentNode.outputs[0].links : []) || [];
            if (outputs.length) {
              for (const linkId of outputs) {
                const link = app2.graph.links[linkId];
                if (!link) continue;
                const node = app2.graph.getNodeById(link.target_id);
                const type2 = node.constructor.type;
                if (type2 === "Reroute") {
                  nodes.push(node);
                  updateNodes.push(node);
                } else {
                  const nodeOutType = node.inputs && node.inputs[link?.target_slot] && node.inputs[link.target_slot].type ? node.inputs[link.target_slot].type : null;
                  if (inputType && inputType !== "*" && nodeOutType !== inputType) {
                    node.disconnectInput(link.target_slot);
                  } else {
                    outputType = nodeOutType;
                  }
                }
              }
            } else {
            }
          }
          const displayType = inputType || outputType || "*";
          const color = LGraphCanvas.link_type_colors[displayType];
          let widgetConfig;
          let targetWidget;
          let widgetType;
          for (const node of updateNodes) {
            node.outputs[0].type = inputType || "*";
            node.__outputType = displayType;
            node.outputs[0].name = node.properties.showOutputText ? displayType : "";
            node.size = node.computeSize();
            node.applyOrientation();
            for (const l of node.outputs[0].links || []) {
              const link = app2.graph.links[l];
              if (link) {
                link.color = color;
                if (app2.configuringGraph) continue;
                const targetNode = app2.graph.getNodeById(link.target_id);
                const targetInput = targetNode.inputs?.[link.target_slot];
                if (targetInput?.widget) {
                  const config = getWidgetConfig(targetInput);
                  if (!widgetConfig) {
                    widgetConfig = config[1] ?? {};
                    widgetType = config[0];
                  }
                  if (!targetWidget) {
                    targetWidget = targetNode.widgets?.find(
                      (w) => w.name === targetInput.widget.name
                    );
                  }
                  const merged = mergeIfValid(targetInput, [
                    config[0],
                    widgetConfig
                  ]);
                  if (merged.customConfig) {
                    widgetConfig = merged.customConfig;
                  }
                }
              }
            }
          }
          for (const node of updateNodes) {
            if (widgetConfig && outputType) {
              node.inputs[0].widget = { name: "value" };
              setWidgetConfig(
                node.inputs[0],
                [widgetType ?? displayType, widgetConfig],
                targetWidget
              );
            } else {
              setWidgetConfig(node.inputs[0], null);
            }
          }
          if (inputNode) {
            const link = app2.graph.links[inputNode.inputs[0].link];
            if (link) {
              link.color = color;
            }
          }
        };
        this.clone = function() {
          const cloned = RerouteNode.prototype.clone.apply(this);
          cloned.removeOutput(0);
          cloned.addOutput(this.properties.showOutputText ? "*" : "", "*");
          cloned.size = cloned.computeSize();
          return cloned;
        };
        this.isVirtualNode = true;
      }
      getExtraMenuOptions(_, options) {
        options.unshift(
          {
            content: (this.properties.showOutputText ? "Hide" : "Show") + " Type",
            callback: /* @__PURE__ */ __name(() => {
              this.properties.showOutputText = !this.properties.showOutputText;
              if (this.properties.showOutputText) {
                this.outputs[0].name = this.__outputType || this.outputs[0].type;
              } else {
                this.outputs[0].name = "";
              }
              this.size = this.computeSize();
              this.applyOrientation();
              app2.graph.setDirtyCanvas(true, true);
            }, "callback")
          },
          {
            content: (RerouteNode.defaultVisibility ? "Hide" : "Show") + " Type By Default",
            callback: /* @__PURE__ */ __name(() => {
              RerouteNode.setDefaultTextVisibility(
                !RerouteNode.defaultVisibility
              );
            }, "callback")
          },
          {
            // naming is inverted with respect to LiteGraphNode.horizontal
            // LiteGraphNode.horizontal == true means that
            // each slot in the inputs and outputs are laid out horizontally,
            // which is the opposite of the visual orientation of the inputs and outputs as a node
            content: "Set " + (this.properties.horizontal ? "Horizontal" : "Vertical"),
            callback: /* @__PURE__ */ __name(() => {
              this.properties.horizontal = !this.properties.horizontal;
              this.applyOrientation();
            }, "callback")
          }
        );
      }
      applyOrientation() {
        this.horizontal = this.properties.horizontal;
        if (this.horizontal) {
          this.inputs[0].pos = [this.size[0] / 2, 0];
        } else {
          delete this.inputs[0].pos;
        }
        app2.graph.setDirtyCanvas(true, true);
      }
      computeSize() {
        return [
          this.properties.showOutputText && this.outputs && this.outputs.length ? Math.max(
            75,
            LiteGraph.NODE_TEXT_SIZE * this.outputs[0].name.length * 0.6 + 40
          ) : 75,
          26
        ];
      }
      static setDefaultTextVisibility(visible) {
        RerouteNode.defaultVisibility = visible;
        if (visible) {
          localStorage["Comfy.RerouteNode.DefaultVisibility"] = "true";
        } else {
          delete localStorage["Comfy.RerouteNode.DefaultVisibility"];
        }
      }
    }
    RerouteNode.setDefaultTextVisibility(
      !!localStorage["Comfy.RerouteNode.DefaultVisibility"]
    );
    LiteGraph.registerNodeType(
      "Reroute",
      Object.assign(RerouteNode, {
        title_mode: LiteGraph.NO_TITLE,
        title: "Reroute",
        collapsable: false
      })
    );
    RerouteNode.category = "utils";
  }
});
app.registerExtension({
  name: "Comfy.SaveImageExtraOutput",
  async beforeRegisterNodeDef(nodeType, nodeData, app2) {
    if (nodeData.name === "SaveImage" || nodeData.name === "SaveAnimatedWEBP") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : void 0;
        const widget = this.widgets.find((w) => w.name === "filename_prefix");
        widget.serializeValue = () => {
          return applyTextReplacements(app2, widget.value);
        };
        return r;
      };
    } else {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : void 0;
        if (!this.properties || !("Node name for S&R" in this.properties)) {
          this.addProperty("Node name for S&R", this.constructor.type, "string");
        }
        return r;
      };
    }
  }
});
let touchZooming;
let touchCount = 0;
app.registerExtension({
  name: "Comfy.SimpleTouchSupport",
  setup() {
    let zoomPos;
    let touchTime;
    let lastTouch;
    function getMultiTouchPos(e) {
      return Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
    }
    __name(getMultiTouchPos, "getMultiTouchPos");
    app.canvasEl.addEventListener(
      "touchstart",
      (e) => {
        touchCount++;
        lastTouch = null;
        if (e.touches?.length === 1) {
          touchTime = /* @__PURE__ */ new Date();
          lastTouch = e.touches[0];
        } else {
          touchTime = null;
          if (e.touches?.length === 2) {
            zoomPos = getMultiTouchPos(e);
            app.canvas.pointer_is_down = false;
          }
        }
      },
      true
    );
    app.canvasEl.addEventListener("touchend", (e) => {
      touchZooming = false;
      touchCount = e.touches?.length ?? touchCount - 1;
      if (touchTime && !e.touches?.length) {
        if ((/* @__PURE__ */ new Date()).getTime() - touchTime > 600) {
          try {
            e.constructor = CustomEvent;
          } catch (error) {
          }
          e.clientX = lastTouch.clientX;
          e.clientY = lastTouch.clientY;
          app.canvas.pointer_is_down = true;
          app.canvas._mousedown_callback(e);
        }
        touchTime = null;
      }
    });
    app.canvasEl.addEventListener(
      "touchmove",
      (e) => {
        touchTime = null;
        if (e.touches?.length === 2) {
          app.canvas.pointer_is_down = false;
          touchZooming = true;
          LiteGraph.closeAllContextMenus();
          app.canvas.search_box?.close();
          const newZoomPos = getMultiTouchPos(e);
          const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
          const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
          let scale = app.canvas.ds.scale;
          const diff = zoomPos - newZoomPos;
          if (diff > 0.5) {
            scale *= 1 / 1.07;
          } else if (diff < -0.5) {
            scale *= 1.07;
          }
          app.canvas.ds.changeScale(scale, [midX, midY]);
          app.canvas.setDirty(true, true);
          zoomPos = newZoomPos;
        }
      },
      true
    );
  }
});
const processMouseDown = LGraphCanvas.prototype.processMouseDown;
LGraphCanvas.prototype.processMouseDown = function(e) {
  if (touchZooming || touchCount) {
    return;
  }
  return processMouseDown.apply(this, arguments);
};
const processMouseMove = LGraphCanvas.prototype.processMouseMove;
LGraphCanvas.prototype.processMouseMove = function(e) {
  if (touchZooming || touchCount > 1) {
    return;
  }
  return processMouseMove.apply(this, arguments);
};
app.registerExtension({
  name: "Comfy.SlotDefaults",
  suggestionsNumber: null,
  init() {
    LiteGraph.search_filter_enabled = true;
    LiteGraph.middle_click_slot_add_default_node = true;
    this.suggestionsNumber = app.ui.settings.addSetting({
      id: "Comfy.NodeSuggestions.number",
      category: ["Comfy", "Node Search Box", "NodeSuggestions"],
      name: "Number of nodes suggestions",
      tooltip: "Only for litegraph searchbox/context menu",
      type: "slider",
      attrs: {
        min: 1,
        max: 100,
        step: 1
      },
      defaultValue: 5,
      onChange: /* @__PURE__ */ __name((newVal, oldVal) => {
        this.setDefaults(newVal);
      }, "onChange")
    });
  },
  slot_types_default_out: {},
  slot_types_default_in: {},
  async beforeRegisterNodeDef(nodeType, nodeData, app2) {
    var nodeId = nodeData.name;
    var inputs = [];
    inputs = nodeData["input"]["required"];
    for (const inputKey in inputs) {
      var input = inputs[inputKey];
      if (typeof input[0] !== "string") continue;
      var type = input[0];
      if (type in ComfyWidgets) {
        var customProperties = input[1];
        if (!customProperties?.forceInput) continue;
      }
      if (!(type in this.slot_types_default_out)) {
        this.slot_types_default_out[type] = ["Reroute"];
      }
      if (this.slot_types_default_out[type].includes(nodeId)) continue;
      this.slot_types_default_out[type].push(nodeId);
      const lowerType = type.toLocaleLowerCase();
      if (!(lowerType in LiteGraph.registered_slot_in_types)) {
        LiteGraph.registered_slot_in_types[lowerType] = { nodes: [] };
      }
      LiteGraph.registered_slot_in_types[lowerType].nodes.push(
        nodeType.comfyClass
      );
    }
    var outputs = nodeData["output"];
    for (const key in outputs) {
      var type = outputs[key];
      if (!(type in this.slot_types_default_in)) {
        this.slot_types_default_in[type] = ["Reroute"];
      }
      this.slot_types_default_in[type].push(nodeId);
      if (!(type in LiteGraph.registered_slot_out_types)) {
        LiteGraph.registered_slot_out_types[type] = { nodes: [] };
      }
      LiteGraph.registered_slot_out_types[type].nodes.push(nodeType.comfyClass);
      if (!LiteGraph.slot_types_out.includes(type)) {
        LiteGraph.slot_types_out.push(type);
      }
    }
    var maxNum = this.suggestionsNumber.value;
    this.setDefaults(maxNum);
  },
  setDefaults(maxNum) {
    LiteGraph.slot_types_default_out = {};
    LiteGraph.slot_types_default_in = {};
    for (const type in this.slot_types_default_out) {
      LiteGraph.slot_types_default_out[type] = this.slot_types_default_out[type].slice(0, maxNum);
    }
    for (const type in this.slot_types_default_in) {
      LiteGraph.slot_types_default_in[type] = this.slot_types_default_in[type].slice(0, maxNum);
    }
  }
});
function roundVectorToGrid(vec) {
  vec[0] = LiteGraph.CANVAS_GRID_SIZE * Math.round(vec[0] / LiteGraph.CANVAS_GRID_SIZE);
  vec[1] = LiteGraph.CANVAS_GRID_SIZE * Math.round(vec[1] / LiteGraph.CANVAS_GRID_SIZE);
  return vec;
}
__name(roundVectorToGrid, "roundVectorToGrid");
app.registerExtension({
  name: "Comfy.SnapToGrid",
  init() {
    app.ui.settings.addSetting({
      id: "Comfy.SnapToGrid.GridSize",
      category: ["Comfy", "Graph", "GridSize"],
      name: "Snap to grid size",
      type: "slider",
      attrs: {
        min: 1,
        max: 500
      },
      tooltip: "When dragging and resizing nodes while holding shift they will be aligned to the grid, this controls the size of that grid.",
      defaultValue: LiteGraph.CANVAS_GRID_SIZE,
      onChange(value) {
        LiteGraph.CANVAS_GRID_SIZE = +value;
      }
    });
    const onNodeMoved = app.canvas.onNodeMoved;
    app.canvas.onNodeMoved = function(node) {
      const r = onNodeMoved?.apply(this, arguments);
      if (app.shiftDown) {
        for (const id2 in this.selected_nodes) {
          this.selected_nodes[id2].alignToGrid();
        }
      }
      return r;
    };
    const onNodeAdded = app.graph.onNodeAdded;
    app.graph.onNodeAdded = function(node) {
      const onResize = node.onResize;
      node.onResize = function() {
        if (app.shiftDown) {
          roundVectorToGrid(node.size);
        }
        return onResize?.apply(this, arguments);
      };
      return onNodeAdded?.apply(this, arguments);
    };
    const origDrawNode = LGraphCanvas.prototype.drawNode;
    LGraphCanvas.prototype.drawNode = function(node, ctx) {
      if (app.shiftDown && this.node_dragged && node.id in this.selected_nodes) {
        const [x, y] = roundVectorToGrid([...node.pos]);
        const shiftX = x - node.pos[0];
        let shiftY = y - node.pos[1];
        let w, h;
        if (node.flags.collapsed) {
          w = node._collapsed_width;
          h = LiteGraph.NODE_TITLE_HEIGHT;
          shiftY -= LiteGraph.NODE_TITLE_HEIGHT;
        } else {
          w = node.size[0];
          h = node.size[1];
          let titleMode = node.constructor.title_mode;
          if (titleMode !== LiteGraph.TRANSPARENT_TITLE && titleMode !== LiteGraph.NO_TITLE) {
            h += LiteGraph.NODE_TITLE_HEIGHT;
            shiftY -= LiteGraph.NODE_TITLE_HEIGHT;
          }
        }
        const f = ctx.fillStyle;
        ctx.fillStyle = "rgba(100, 100, 100, 0.5)";
        ctx.fillRect(shiftX, shiftY, w, h);
        ctx.fillStyle = f;
      }
      return origDrawNode.apply(this, arguments);
    };
    let selectedAndMovingGroup = null;
    const groupMove = LGraphGroup.prototype.move;
    LGraphGroup.prototype.move = function(deltax, deltay, ignore_nodes) {
      const v = groupMove.apply(this, arguments);
      if (!selectedAndMovingGroup && app.canvas.selected_group === this && (deltax || deltay)) {
        selectedAndMovingGroup = this;
      }
      if (app.canvas.last_mouse_dragging === false && app.shiftDown) {
        this.recomputeInsideNodes();
        for (const node of this._nodes) {
          node.alignToGrid();
        }
        LGraphNode.prototype.alignToGrid.apply(this);
      }
      return v;
    };
    const drawGroups = LGraphCanvas.prototype.drawGroups;
    LGraphCanvas.prototype.drawGroups = function(canvas, ctx) {
      if (this.selected_group && app.shiftDown) {
        if (this.selected_group_resizing) {
          roundVectorToGrid(this.selected_group.size);
        } else if (selectedAndMovingGroup) {
          const [x, y] = roundVectorToGrid([...selectedAndMovingGroup.pos]);
          const f = ctx.fillStyle;
          const s = ctx.strokeStyle;
          ctx.fillStyle = "rgba(100, 100, 100, 0.33)";
          ctx.strokeStyle = "rgba(100, 100, 100, 0.66)";
          ctx.rect(x, y, ...selectedAndMovingGroup.size);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = f;
          ctx.strokeStyle = s;
        }
      } else if (!this.selected_group) {
        selectedAndMovingGroup = null;
      }
      return drawGroups.apply(this, arguments);
    };
    const onGroupAdd = LGraphCanvas.onGroupAdd;
    LGraphCanvas.onGroupAdd = function() {
      const v = onGroupAdd.apply(app.canvas, arguments);
      if (app.shiftDown) {
        const lastGroup = app.graph._groups[app.graph._groups.length - 1];
        if (lastGroup) {
          roundVectorToGrid(lastGroup.pos);
          roundVectorToGrid(lastGroup.size);
        }
      }
      return v;
    };
  }
});
app.registerExtension({
  name: "Comfy.UploadImage",
  async beforeRegisterNodeDef(nodeType, nodeData, app2) {
    if (nodeData?.input?.required?.image?.[1]?.image_upload === true) {
      nodeData.input.required.upload = ["IMAGEUPLOAD"];
    }
  }
});
const WEBCAM_READY = Symbol();
app.registerExtension({
  name: "Comfy.WebcamCapture",
  getCustomWidgets(app2) {
    return {
      WEBCAM(node, inputName) {
        let res;
        node[WEBCAM_READY] = new Promise((resolve) => res = resolve);
        const container = document.createElement("div");
        container.style.background = "rgba(0,0,0,0.25)";
        container.style.textAlign = "center";
        const video = document.createElement("video");
        video.style.height = video.style.width = "100%";
        const loadVideo = /* @__PURE__ */ __name(async () => {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false
            });
            container.replaceChildren(video);
            setTimeout(() => res(video), 500);
            video.addEventListener("loadedmetadata", () => res(video), false);
            video.srcObject = stream;
            video.play();
          } catch (error) {
            const label = document.createElement("div");
            label.style.color = "red";
            label.style.overflow = "auto";
            label.style.maxHeight = "100%";
            label.style.whiteSpace = "pre-wrap";
            if (window.isSecureContext) {
              label.textContent = "Unable to load webcam, please ensure access is granted:\n" + error.message;
            } else {
              label.textContent = "Unable to load webcam. A secure context is required, if you are not accessing ComfyUI on localhost (127.0.0.1) you will have to enable TLS (https)\n\n" + error.message;
            }
            container.replaceChildren(label);
          }
        }, "loadVideo");
        loadVideo();
        return { widget: node.addDOMWidget(inputName, "WEBCAM", container) };
      }
    };
  },
  nodeCreated(node) {
    if (node.type, node.constructor.comfyClass !== "WebcamCapture") return;
    let video;
    const camera = node.widgets.find((w2) => w2.name === "image");
    const w = node.widgets.find((w2) => w2.name === "width");
    const h = node.widgets.find((w2) => w2.name === "height");
    const captureOnQueue = node.widgets.find(
      (w2) => w2.name === "capture_on_queue"
    );
    const canvas = document.createElement("canvas");
    const capture = /* @__PURE__ */ __name(() => {
      canvas.width = w.value;
      canvas.height = h.value;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, w.value, h.value);
      const data = canvas.toDataURL("image/png");
      const img = new Image();
      img.onload = () => {
        node.imgs = [img];
        app.graph.setDirtyCanvas(true);
        requestAnimationFrame(() => {
          node.setSizeForImage?.();
        });
      };
      img.src = data;
    }, "capture");
    const btn = node.addWidget(
      "button",
      "waiting for camera...",
      "capture",
      capture
    );
    btn.disabled = true;
    btn.serializeValue = () => void 0;
    camera.serializeValue = async () => {
      if (captureOnQueue.value) {
        capture();
      } else if (!node.imgs?.length) {
        const err = `No webcam image captured`;
        alert(err);
        throw new Error(err);
      }
      const blob = await new Promise((r) => canvas.toBlob(r));
      const name = `${+/* @__PURE__ */ new Date()}.png`;
      const file2 = new File([blob], name);
      const body = new FormData();
      body.append("image", file2);
      body.append("subfolder", "webcam");
      body.append("type", "temp");
      const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body
      });
      if (resp.status !== 200) {
        const err = `Error uploading camera image: ${resp.status} - ${resp.statusText}`;
        alert(err);
        throw new Error(err);
      }
      return `webcam/${name} [temp]`;
    };
    node[WEBCAM_READY].then((v) => {
      video = v;
      if (!w.value) {
        w.value = video.videoWidth || 640;
        h.value = video.videoHeight || 480;
      }
      btn.disabled = false;
      btn.label = "capture";
    });
  }
});
function splitFilePath(path) {
  const folder_separator = path.lastIndexOf("/");
  if (folder_separator === -1) {
    return ["", path];
  }
  return [
    path.substring(0, folder_separator),
    path.substring(folder_separator + 1)
  ];
}
__name(splitFilePath, "splitFilePath");
function getResourceURL(subfolder, filename, type = "input") {
  const params = [
    "filename=" + encodeURIComponent(filename),
    "type=" + type,
    "subfolder=" + subfolder,
    app.getRandParam().substring(1)
  ].join("&");
  return `/view?${params}`;
}
__name(getResourceURL, "getResourceURL");
async function uploadFile(audioWidget, audioUIWidget, file2, updateNode, pasted = false) {
  try {
    const body = new FormData();
    body.append("image", file2);
    if (pasted) body.append("subfolder", "pasted");
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body
    });
    if (resp.status === 200) {
      const data = await resp.json();
      let path = data.name;
      if (data.subfolder) path = data.subfolder + "/" + path;
      if (!audioWidget.options.values.includes(path)) {
        audioWidget.options.values.push(path);
      }
      if (updateNode) {
        audioUIWidget.element.src = api.apiURL(
          getResourceURL(...splitFilePath(path))
        );
        audioWidget.value = path;
      }
    } else {
      alert(resp.status + " - " + resp.statusText);
    }
  } catch (error) {
    alert(error);
  }
}
__name(uploadFile, "uploadFile");
app.registerExtension({
  name: "Comfy.AudioWidget",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (["LoadAudio", "SaveAudio", "PreviewAudio"].includes(nodeType.comfyClass)) {
      nodeData.input.required.audioUI = ["AUDIO_UI"];
    }
  },
  getCustomWidgets() {
    return {
      AUDIO_UI(node, inputName) {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.classList.add("comfy-audio");
        audio.setAttribute("name", "media");
        const audioUIWidget = node.addDOMWidget(
          inputName,
          /* name=*/
          "audioUI",
          audio
        );
        audioUIWidget.serialize = false;
        const isOutputNode = node.constructor.nodeData.output_node;
        if (isOutputNode) {
          audioUIWidget.element.classList.add("empty-audio-widget");
          const onExecuted = node.onExecuted;
          node.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);
            const audios = message.audio;
            if (!audios) return;
            const audio2 = audios[0];
            audioUIWidget.element.src = api.apiURL(
              getResourceURL(audio2.subfolder, audio2.filename, audio2.type)
            );
            audioUIWidget.element.classList.remove("empty-audio-widget");
          };
        }
        return { widget: audioUIWidget };
      }
    };
  },
  onNodeOutputsUpdated(nodeOutputs) {
    for (const [nodeId, output] of Object.entries(nodeOutputs)) {
      const node = app.graph.getNodeById(nodeId);
      if ("audio" in output) {
        const audioUIWidget = node.widgets.find(
          (w) => w.name === "audioUI"
        );
        const audio = output.audio[0];
        audioUIWidget.element.src = api.apiURL(
          getResourceURL(audio.subfolder, audio.filename, audio.type)
        );
        audioUIWidget.element.classList.remove("empty-audio-widget");
      }
    }
  }
});
app.registerExtension({
  name: "Comfy.UploadAudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.input?.required?.audio?.[1]?.audio_upload === true) {
      nodeData.input.required.upload = ["AUDIOUPLOAD"];
    }
  },
  getCustomWidgets() {
    return {
      AUDIOUPLOAD(node, inputName) {
        const audioWidget = node.widgets.find(
          (w) => w.name === "audio"
        );
        const audioUIWidget = node.widgets.find(
          (w) => w.name === "audioUI"
        );
        const onAudioWidgetUpdate = /* @__PURE__ */ __name(() => {
          audioUIWidget.element.src = api.apiURL(
            getResourceURL(...splitFilePath(audioWidget.value))
          );
        }, "onAudioWidgetUpdate");
        if (audioWidget.value) {
          onAudioWidgetUpdate();
        }
        audioWidget.callback = onAudioWidgetUpdate;
        const onGraphConfigured = node.onGraphConfigured;
        node.onGraphConfigured = function() {
          onGraphConfigured?.apply(this, arguments);
          if (audioWidget.value) {
            onAudioWidgetUpdate();
          }
        };
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "audio/*";
        fileInput.style.display = "none";
        fileInput.onchange = () => {
          if (fileInput.files.length) {
            uploadFile(audioWidget, audioUIWidget, fileInput.files[0], true);
          }
        };
        const uploadWidget = node.addWidget(
          "button",
          inputName,
          /* value=*/
          "",
          () => {
            fileInput.click();
          }
        );
        uploadWidget.label = "choose file to upload";
        uploadWidget.serialize = false;
        return { widget: uploadWidget };
      }
    };
  }
});
//# sourceMappingURL=index-CrROdkG4.js.map
