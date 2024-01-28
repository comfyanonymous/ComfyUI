import {app} from "../../scripts/app.js";
import {$el} from "../../scripts/ui.js";

// Manage color palettes

const colorPalettes = {
	"dark": {
		"id": "dark",
		"name": "Dark (Default)",
		"colors": {
			"node_slot": {
				"CLIP": "#FFD500", // bright yellow
				"CLIP_VISION": "#A8DADC", // light blue-gray
				"CLIP_VISION_OUTPUT": "#ad7452", // rusty brown-orange
				"CONDITIONING": "#FFA931", // vibrant orange-yellow
				"CONTROL_NET": "#6EE7B7", // soft mint green
				"IMAGE": "#64B5F6", // bright sky blue
				"LATENT": "#FF9CF9", // light pink-purple
				"MASK": "#81C784", // muted green
				"MODEL": "#B39DDB", // light lavender-purple
				"STYLE_MODEL": "#C2FFAE", // light green-yellow
				"VAE": "#FF6E6E", // bright red
				"TAESD": "#DCC274", // cheesecake
			},
			"litegraph_base": {
				"BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII=",
				"CLEAR_BACKGROUND_COLOR": "#222",
				"NODE_TITLE_COLOR": "#999",
				"NODE_SELECTED_TITLE_COLOR": "#FFF",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#AAA",
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#333",
				"NODE_DEFAULT_BGCOLOR": "#353535",
				"NODE_DEFAULT_BOXCOLOR": "#666",
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#FFF",
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#222",
				"WIDGET_OUTLINE_COLOR": "#666",
				"WIDGET_TEXT_COLOR": "#DDD",
				"WIDGET_SECONDARY_TEXT_COLOR": "#999",

				"LINK_COLOR": "#9A9",
				"EVENT_LINK_COLOR": "#A86",
				"CONNECTING_LINK_COLOR": "#AFA",
			},
			"comfy_base": {
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
			}
		},
	},
	"light": {
		"id": "light",
		"name": "Light",
		"colors": {
			"node_slot": {
				"CLIP": "#FFA726", // orange
				"CLIP_VISION": "#5C6BC0", // indigo
				"CLIP_VISION_OUTPUT": "#8D6E63", // brown
				"CONDITIONING": "#EF5350", // red
				"CONTROL_NET": "#66BB6A", // green
				"IMAGE": "#42A5F5", // blue
				"LATENT": "#AB47BC", // purple
				"MASK": "#9CCC65", // light green
				"MODEL": "#7E57C2", // deep purple
				"STYLE_MODEL": "#D4E157", // lime
				"VAE": "#FF7043", // deep orange
			},
			"litegraph_base": {
				"BACKGROUND_IMAGE": "data:image/gif;base64,R0lGODlhZABkALMAAAAAAP///+vr6+rq6ujo6Ofn5+bm5uXl5d3d3f///wAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAAkALAAAAABkAGQAAAT/UMhJq7046827HkcoHkYxjgZhnGG6si5LqnIM0/fL4qwwIMAg0CAsEovBIxKhRDaNy2GUOX0KfVFrssrNdpdaqTeKBX+dZ+jYvEaTf+y4W66mC8PUdrE879f9d2mBeoNLfH+IhYBbhIx2jkiHiomQlGKPl4uZe3CaeZifnnijgkESBqipqqusra6vsLGys62SlZO4t7qbuby7CLa+wqGWxL3Gv3jByMOkjc2lw8vOoNSi0czAncXW3Njdx9Pf48/Z4Kbbx+fQ5evZ4u3k1fKR6cn03vHlp7T9/v8A/8Gbp4+gwXoFryXMB2qgwoMMHyKEqA5fxX322FG8tzBcRnMW/zlulPbRncmQGidKjMjyYsOSKEF2FBlJQMCbOHP6c9iSZs+UnGYCdbnSo1CZI5F64kn0p1KnTH02nSoV3dGTV7FFHVqVq1dtWcMmVQZTbNGu72zqXMuW7danVL+6e4t1bEy6MeueBYLXrNO5Ze36jQtWsOG97wIj1vt3St/DjTEORss4nNq2mDP3e7w4r1bFkSET5hy6s2TRlD2/mSxXtSHQhCunXo26NevCpmvD/UU6tuullzULH76q92zdZG/Ltv1a+W+osI/nRmyc+fRi1Xdbh+68+0vv10dH3+77KD/i6IdnX669/frn5Zsjh4/2PXju8+8bzc9/6fj27LFnX11/+IUnXWl7BJfegm79FyB9JOl3oHgSklefgxAC+FmFGpqHIYcCfkhgfCohSKKJVo044YUMttggiBkmp6KFXw1oII24oYhjiDByaKOOHcp3Y5BD/njikSkO+eBREQAAOw==",
				"CLEAR_BACKGROUND_COLOR": "lightgray",
				"NODE_TITLE_COLOR": "#222",
				"NODE_SELECTED_TITLE_COLOR": "#000",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#444",
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#F7F7F7",
				"NODE_DEFAULT_BGCOLOR": "#F5F5F5",
				"NODE_DEFAULT_BOXCOLOR": "#CCC",
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#000",
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.1)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#D4D4D4",
				"WIDGET_OUTLINE_COLOR": "#999",
				"WIDGET_TEXT_COLOR": "#222",
				"WIDGET_SECONDARY_TEXT_COLOR": "#555",

				"LINK_COLOR": "#4CAF50",
				"EVENT_LINK_COLOR": "#FF9800",
				"CONNECTING_LINK_COLOR": "#2196F3",
			},
			"comfy_base": {
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
			}
		},
	},
	"solarized": {
		"id": "solarized",
		"name": "Solarized",
		"colors": {
			"node_slot": {
				"CLIP": "#2AB7CA", // light blue
				"CLIP_VISION": "#6c71c4", // blue violet
				"CLIP_VISION_OUTPUT": "#859900", // olive green
				"CONDITIONING": "#d33682", // magenta
				"CONTROL_NET": "#d1ffd7", // light mint green
				"IMAGE": "#5940bb", // deep blue violet
				"LATENT": "#268bd2", // blue
				"MASK": "#CCC9E7", // light purple-gray
				"MODEL": "#dc322f", // red
				"STYLE_MODEL": "#1a998a", // teal
				"UPSCALE_MODEL": "#054A29", // dark green
				"VAE": "#facfad", // light pink-orange
			},
			"litegraph_base": {
				"NODE_TITLE_COLOR": "#fdf6e3", // Base3
				"NODE_SELECTED_TITLE_COLOR": "#A9D400",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#657b83", // Base00
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#094656",
				"NODE_DEFAULT_BGCOLOR": "#073642", // Base02
				"NODE_DEFAULT_BOXCOLOR": "#839496", // Base0
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#fdf6e3", // Base3
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#002b36", // Base03
				"WIDGET_OUTLINE_COLOR": "#839496", // Base0
				"WIDGET_TEXT_COLOR": "#fdf6e3", // Base3
				"WIDGET_SECONDARY_TEXT_COLOR": "#93a1a1", // Base1

				"LINK_COLOR": "#2aa198", // Solarized Cyan
				"EVENT_LINK_COLOR": "#268bd2", // Solarized Blue
				"CONNECTING_LINK_COLOR": "#859900", // Solarized Green
			},
			"comfy_base": {
				"fg-color": "#fdf6e3", // Base3
				"bg-color": "#002b36", // Base03
				"comfy-menu-bg": "#073642", // Base02
				"comfy-input-bg": "#002b36", // Base03
				"input-text": "#93a1a1", // Base1
				"descrip-text": "#586e75", // Base01
				"drag-text": "#839496", // Base0
				"error-text": "#dc322f", // Solarized Red
				"border-color": "#657b83", // Base00
				"tr-even-bg-color": "#002b36",
				"tr-odd-bg-color": "#073642",
			}
		},
	},
	"arc": {
		"id": "arc",
		"name": "Arc",
		"colors": {
			"node_slot": {
			  "BOOLEAN": "",
			  "CLIP": "#eacb8b",
			  "CLIP_VISION": "#A8DADC",
			  "CLIP_VISION_OUTPUT": "#ad7452",
			  "CONDITIONING": "#cf876f",
			  "CONTROL_NET": "#00d78d",
			  "CONTROL_NET_WEIGHTS": "",
			  "FLOAT": "",
			  "GLIGEN": "",
			  "IMAGE": "#80a1c0",
			  "IMAGEUPLOAD": "",
			  "INT": "",
			  "LATENT": "#b38ead",
			  "LATENT_KEYFRAME": "",
			  "MASK": "#a3bd8d",
			  "MODEL": "#8978a7",
			  "SAMPLER": "",
			  "SIGMAS": "",
			  "STRING": "",
			  "STYLE_MODEL": "#C2FFAE",
			  "T2I_ADAPTER_WEIGHTS": "",
			  "TAESD": "#DCC274",
			  "TIMESTEP_KEYFRAME": "",
			  "UPSCALE_MODEL": "",
			  "VAE": "#be616b"
			},
			"litegraph_base": {
			  "BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAACXBIWXMAAAsTAAALEwEAmpwYAAABcklEQVR4nO3YMUoDARgF4RfxBqZI6/0vZqFn0MYtrLIQMFN8U6V4LAtD+Jm9XG/v30OGl2e/AP7yevz4+vx45nvgF/+QGITEICQGITEIiUFIjNNC3q43u3/YnRJyPOzeQ+0e220nhRzReC8e7R7bbdvl+Jal1Bs46jEIiUFIDEJiEBKDkBhKPbZT6qHdptRTu02p53DUYxASg5AYhMQgJAYhMZR6bKfUQ7tNqad2m1LP4ajHICQGITEIiUFIDEJiKPXYTqmHdptST+02pZ7DUY9BSAxCYhASg5AYhMRQ6rGdUg/tNqWe2m1KPYejHoOQGITEICQGITEIiaHUYzulHtptSj2125R6Dkc9BiExCIlBSAxCYhASQ6nHdko9tNuUemq3KfUcjnoMQmIQEoOQGITEICSGUo/tlHpotyn11G5T6jkc9RiExCAkBiExCIlBSAylHtsp9dBuU+qp3abUczjqMQiJQUgMQmIQEoOQGITE+AHFISNQrFTGuwAAAABJRU5ErkJggg==",
			  "CLEAR_BACKGROUND_COLOR": "#2b2f38",
			  "NODE_TITLE_COLOR": "#b2b7bd",
			  "NODE_SELECTED_TITLE_COLOR": "#FFF",
			  "NODE_TEXT_SIZE": 14,
			  "NODE_TEXT_COLOR": "#AAA",
			  "NODE_SUBTEXT_SIZE": 12,
			  "NODE_DEFAULT_COLOR": "#2b2f38",
			  "NODE_DEFAULT_BGCOLOR": "#242730",
			  "NODE_DEFAULT_BOXCOLOR": "#6e7581",
			  "NODE_DEFAULT_SHAPE": "box",
			  "NODE_BOX_OUTLINE_COLOR": "#FFF",
			  "DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
			  "DEFAULT_GROUP_FONT": 22,
			  "WIDGET_BGCOLOR": "#2b2f38",
			  "WIDGET_OUTLINE_COLOR": "#6e7581",
			  "WIDGET_TEXT_COLOR": "#DDD",
			  "WIDGET_SECONDARY_TEXT_COLOR": "#b2b7bd",
			  "LINK_COLOR": "#9A9",
			  "EVENT_LINK_COLOR": "#A86",
			  "CONNECTING_LINK_COLOR": "#AFA"
			},
			"comfy_base": {
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
			  "tr-odd-bg-color": "#242730"
			}
		},
	},
	"nord": {
		"id": "nord",
		"name": "Nord",
		"colors": {
			"node_slot": {
			  "BOOLEAN": "",
			  "CLIP": "#eacb8b",
			  "CLIP_VISION": "#A8DADC",
			  "CLIP_VISION_OUTPUT": "#ad7452",
			  "CONDITIONING": "#cf876f",
			  "CONTROL_NET": "#00d78d",
			  "CONTROL_NET_WEIGHTS": "",
			  "FLOAT": "",
			  "GLIGEN": "",
			  "IMAGE": "#80a1c0",
			  "IMAGEUPLOAD": "",
			  "INT": "",
			  "LATENT": "#b38ead",
			  "LATENT_KEYFRAME": "",
			  "MASK": "#a3bd8d",
			  "MODEL": "#8978a7",
			  "SAMPLER": "",
			  "SIGMAS": "",
			  "STRING": "",
			  "STYLE_MODEL": "#C2FFAE",
			  "T2I_ADAPTER_WEIGHTS": "",
			  "TAESD": "#DCC274",
			  "TIMESTEP_KEYFRAME": "",
			  "UPSCALE_MODEL": "",
			  "VAE": "#be616b"
			},
			"litegraph_base": {
			  "BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAFu2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOjUwNDFhMmZjLTEzNzQtMTk0ZC1hZWY4LTYxMzM1MTVmNjUwMCIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDoyMzFiMTBiMC1iNGZiLTAyNGUtYjEyZS0zMDUzMDNjZDA3YzgiIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDoyMzFiMTBiMC1iNGZiLTAyNGUtYjEyZS0zMDUzMDNjZDA3YzgiPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJjcmVhdGVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDo1MDQxYTJmYy0xMzc0LTE5NGQtYWVmOC02MTMzNTE1ZjY1MDAiIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDE6MjA6NDUrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz73jWg/AAAAyUlEQVR42u3WKwoAIBRFQRdiMb1idv9Lsxn9gEFw4Dbb8JCTojbbXEJwjJVL2HKwYMGCBQuWLbDmjr+9zrBGjHl1WVcvy2DBggULFizTWQpewSt4HzwsgwULFiwFr7MUvMtS8D54WLBgGSxYCl7BK3iXZbBgwYIFC5bpLAWv4BW8Dx6WwYIFC5aC11kK3mUpeB88LFiwDBYsBa/gFbzLMliwYMGCBct0loJX8AreBw/LYMGCBUvB6ywF77IUvA8eFixYBgsWrNfWAZPltufdad+1AAAAAElFTkSuQmCC",
			  "CLEAR_BACKGROUND_COLOR": "#212732",
			  "NODE_TITLE_COLOR": "#999",
			  "NODE_SELECTED_TITLE_COLOR": "#e5eaf0",
			  "NODE_TEXT_SIZE": 14,
			  "NODE_TEXT_COLOR": "#bcc2c8",
			  "NODE_SUBTEXT_SIZE": 12,
			  "NODE_DEFAULT_COLOR": "#2e3440",
			  "NODE_DEFAULT_BGCOLOR": "#161b22",
			  "NODE_DEFAULT_BOXCOLOR": "#545d70",
			  "NODE_DEFAULT_SHAPE": "box",
			  "NODE_BOX_OUTLINE_COLOR": "#e5eaf0",
			  "DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
			  "DEFAULT_GROUP_FONT": 24,
			  "WIDGET_BGCOLOR": "#2e3440",
			  "WIDGET_OUTLINE_COLOR": "#545d70",
			  "WIDGET_TEXT_COLOR": "#bcc2c8",
			  "WIDGET_SECONDARY_TEXT_COLOR": "#999",
			  "LINK_COLOR": "#9A9",
			  "EVENT_LINK_COLOR": "#A86",
			  "CONNECTING_LINK_COLOR": "#AFA"
			},
			"comfy_base": {
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
			  "tr-odd-bg-color": "#161b22"
			}
		},
	},
	"github": {
		"id": "github",
		"name": "Github",
		"colors": {
			"node_slot": {
			  "BOOLEAN": "",
			  "CLIP": "#eacb8b",
			  "CLIP_VISION": "#A8DADC",
			  "CLIP_VISION_OUTPUT": "#ad7452",
			  "CONDITIONING": "#cf876f",
			  "CONTROL_NET": "#00d78d",
			  "CONTROL_NET_WEIGHTS": "",
			  "FLOAT": "",
			  "GLIGEN": "",
			  "IMAGE": "#80a1c0",
			  "IMAGEUPLOAD": "",
			  "INT": "",
			  "LATENT": "#b38ead",
			  "LATENT_KEYFRAME": "",
			  "MASK": "#a3bd8d",
			  "MODEL": "#8978a7",
			  "SAMPLER": "",
			  "SIGMAS": "",
			  "STRING": "",
			  "STYLE_MODEL": "#C2FFAE",
			  "T2I_ADAPTER_WEIGHTS": "",
			  "TAESD": "#DCC274",
			  "TIMESTEP_KEYFRAME": "",
			  "UPSCALE_MODEL": "",
			  "VAE": "#be616b"
			},
			"litegraph_base": {
			  "BACKGROUND_IMAGE": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGlmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOmIyYzRhNjA5LWJmYTctYTg0MC1iOGFlLTk3MzE2ZjM1ZGIyNyIgeG1wTU06RG9jdW1lbnRJRD0iYWRvYmU6ZG9jaWQ6cGhvdG9zaG9wOjk0ZmNlZGU4LTE1MTctZmQ0MC04ZGU3LWYzOTgxM2E3ODk5ZiIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6MjMxYjEwYjAtYjRmYi0wMjRlLWIxMmUtMzA1MzAzY2QwN2M4IiBzdEV2dDp3aGVuPSIyMDIzLTExLTEzVDAwOjE4OjAyKzAxOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjUuMSAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjQ4OWY1NzlmLTJkNjUtZWQ0Zi04OTg0LTA4NGE2MGE1ZTMzNSIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xNVQwMjowNDo1OSswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDpiMmM0YTYwOS1iZmE3LWE4NDAtYjhhZS05NzMxNmYzNWRiMjciIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz4OTe6GAAAAx0lEQVR42u3WMQoAIQxFwRzJys77X8vSLiRgITif7bYbgrwYc/mKXyBoY4VVBgsWLFiwYFmOlTv+9jfDOjHmr8u6eVkGCxYsWLBgmc5S8ApewXvgYRksWLBgKXidpeBdloL3wMOCBctgwVLwCl7BuyyDBQsWLFiwTGcpeAWv4D3wsAwWLFiwFLzOUvAuS8F74GHBgmWwYCl4Ba/gXZbBggULFixYprMUvIJX8B54WAYLFixYCl5nKXiXpeA98LBgwTJYsGC9tg1o8f4TTtqzNQAAAABJRU5ErkJggg==",
			  "CLEAR_BACKGROUND_COLOR": "#040506",
			  "NODE_TITLE_COLOR": "#999",
			  "NODE_SELECTED_TITLE_COLOR": "#e5eaf0",
			  "NODE_TEXT_SIZE": 14,
			  "NODE_TEXT_COLOR": "#bcc2c8",
			  "NODE_SUBTEXT_SIZE": 12,
			  "NODE_DEFAULT_COLOR": "#161b22",
			  "NODE_DEFAULT_BGCOLOR": "#13171d",
			  "NODE_DEFAULT_BOXCOLOR": "#30363d",
			  "NODE_DEFAULT_SHAPE": "box",
			  "NODE_BOX_OUTLINE_COLOR": "#e5eaf0",
			  "DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
			  "DEFAULT_GROUP_FONT": 24,
			  "WIDGET_BGCOLOR": "#161b22",
			  "WIDGET_OUTLINE_COLOR": "#30363d",
			  "WIDGET_TEXT_COLOR": "#bcc2c8",
			  "WIDGET_SECONDARY_TEXT_COLOR": "#999",
			  "LINK_COLOR": "#9A9",
			  "EVENT_LINK_COLOR": "#A86",
			  "CONNECTING_LINK_COLOR": "#AFA"
			},
			"comfy_base": {
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
			  "tr-odd-bg-color": "#13171d"
			}
		},
	}
};

const id = "Comfy.ColorPalette";
const idCustomColorPalettes = "Comfy.CustomColorPalettes";
const defaultColorPaletteId = "dark";
const els = {}
// const ctxMenu = LiteGraph.ContextMenu;
app.registerExtension({
	name: id,
	addCustomNodeDefs(node_defs) {
		const sortObjectKeys = (unordered) => {
			return Object.keys(unordered).sort().reduce((obj, key) => {
				obj[key] = unordered[key];
				return obj;
			}, {});
		};

		function getSlotTypes() {
			var types = [];

			const defs = node_defs;
			for (const nodeId in defs) {
				const nodeData = defs[nodeId];

				var inputs = nodeData["input"]["required"];
				if (nodeData["input"]["optional"] !== undefined) {
					inputs = Object.assign({}, nodeData["input"]["required"], nodeData["input"]["optional"])
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

		function completeColorPalette(colorPalette) {
			var types = getSlotTypes();

			for (const type of types) {
				if (!colorPalette.colors.node_slot[type]) {
					colorPalette.colors.node_slot[type] = "";
				}
			}

			colorPalette.colors.node_slot = sortObjectKeys(colorPalette.colors.node_slot);

			return colorPalette;
		}

		const getColorPaletteTemplate = async () => {
			let colorPalette = {
				"id": "my_color_palette_unique_id",
				"name": "My Color Palette",
				"colors": {
					"node_slot": {},
					"litegraph_base": {},
					"comfy_base": {}
				}
			};

			// Copy over missing keys from default color palette
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
		};

		const getCustomColorPalettes = () => {
			return app.ui.settings.getSettingValue(idCustomColorPalettes, {});
		};

		const setCustomColorPalettes = (customColorPalettes) => {
			return app.ui.settings.setSettingValue(idCustomColorPalettes, customColorPalettes);
		};

		const addCustomColorPalette = async (colorPalette) => {
			if (typeof (colorPalette) !== "object") {
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

			if (colorPalette.colors.node_slot && typeof (colorPalette.colors.node_slot) !== "object") {
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

			els.select.append($el("option", {
				textContent: colorPalette.name + " (custom)",
				value: "custom_" + colorPalette.id,
				selected: true
			}));

			setColorPalette("custom_" + colorPalette.id);
			await loadColorPalette(colorPalette);
		};

		const deleteCustomColorPalette = async (colorPaletteId) => {
			const customColorPalettes = getCustomColorPalettes();
			delete customColorPalettes[colorPaletteId];
			setCustomColorPalettes(customColorPalettes);

			for (const option of els.select.childNodes) {
				if (option.value === defaultColorPaletteId) {
					option.selected = true;
				}

				if (option.value === "custom_" + colorPaletteId) {
					els.select.removeChild(option);
				}
			}

			setColorPalette(defaultColorPaletteId);
			await loadColorPalette(getColorPalette());
		};

		const loadColorPalette = async (colorPalette) => {
			colorPalette = await completeColorPalette(colorPalette);
			if (colorPalette.colors) {
				// Sets the colors of node slots and links
				if (colorPalette.colors.node_slot) {
					Object.assign(app.canvas.default_connection_color_byType, colorPalette.colors.node_slot);
					Object.assign(LGraphCanvas.link_type_colors, colorPalette.colors.node_slot);
				}
				// Sets the colors of the LiteGraph objects
				if (colorPalette.colors.litegraph_base) {
					// Everything updates correctly in the loop, except the Node Title and Link Color for some reason
					app.canvas.node_title_color = colorPalette.colors.litegraph_base.NODE_TITLE_COLOR;
					app.canvas.default_link_color = colorPalette.colors.litegraph_base.LINK_COLOR;

					for (const key in colorPalette.colors.litegraph_base) {
						if (colorPalette.colors.litegraph_base.hasOwnProperty(key) && LiteGraph.hasOwnProperty(key)) {
							LiteGraph[key] = colorPalette.colors.litegraph_base[key];
						}
					}
				}
				// Sets the color of ComfyUI elements
				if (colorPalette.colors.comfy_base) {
					const rootStyle = document.documentElement.style;
					for (const key in colorPalette.colors.comfy_base) {
						rootStyle.setProperty('--' + key, colorPalette.colors.comfy_base[key]);
					}
				}
				app.canvas.draw(true, true);
			}
		};

		const getColorPalette = (colorPaletteId) => {
			if (!colorPaletteId) {
				colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);
			}

			if (colorPaletteId.startsWith("custom_")) {
				colorPaletteId = colorPaletteId.substr(7);
				let customColorPalettes = getCustomColorPalettes();
				if (customColorPalettes[colorPaletteId]) {
					return customColorPalettes[colorPaletteId];
				}
			}

			return colorPalettes[colorPaletteId];
		};

		const setColorPalette = (colorPaletteId) => {
			app.ui.settings.setSettingValue(id, colorPaletteId);
		};

		const fileInput = $el("input", {
			type: "file",
			accept: ".json",
			style: {display: "none"},
			parent: document.body,
			onchange: () => {
				const file = fileInput.files[0];
				if (file.type === "application/json" || file.name.endsWith(".json")) {
					const reader = new FileReader();
					reader.onload = async () => {
						await addCustomColorPalette(JSON.parse(reader.result));
					};
					reader.readAsText(file);
				}
			},
		});

		app.ui.settings.addSetting({
			id,
			name: "Color Palette",
			type: (name, setter, value) => {
				const options = [
					...Object.values(colorPalettes).map(c=> $el("option", {
						textContent: c.name,
						value: c.id,
						selected: c.id === value
					})),
					...Object.values(getCustomColorPalettes()).map(c=>$el("option", {
						textContent: `${c.name} (custom)`,
						value: `custom_${c.id}`,
						selected: `custom_${c.id}` === value
					}))	,
				];

				els.select = $el("select", {
					style: {
						marginBottom: "0.15rem",
						width: "100%",
					},
					onchange: (e) => {
						setter(e.target.value);
					}
				}, options)

				return $el("tr", [
					$el("td", [
						$el("label", {
							for: id.replaceAll(".", "-"),
							textContent: "Color palette",
						}),
					]),
					$el("td", [
						els.select,
						$el("div", {
							style: {
								display: "grid",
								gap: "4px",
								gridAutoFlow: "column",
							},
						}, [
							$el("input", {
								type: "button",
								value: "Export",
								onclick: async () => {
									const colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);
									const colorPalette = await completeColorPalette(getColorPalette(colorPaletteId));
									const json = JSON.stringify(colorPalette, null, 2); // convert the data to a JSON string
									const blob = new Blob([json], {type: "application/json"});
									const url = URL.createObjectURL(blob);
									const a = $el("a", {
										href: url,
										download: colorPaletteId + ".json",
										style: {display: "none"},
										parent: document.body,
									});
									a.click();
									setTimeout(function () {
										a.remove();
										window.URL.revokeObjectURL(url);
									}, 0);
								},
							}),
							$el("input", {
								type: "button",
								value: "Import",
								onclick: () => {
									fileInput.click();
								}
							}),
							$el("input", {
								type: "button",
								value: "Template",
								onclick: async () => {
									const colorPalette = await getColorPaletteTemplate();
									const json = JSON.stringify(colorPalette, null, 2); // convert the data to a JSON string
									const blob = new Blob([json], {type: "application/json"});
									const url = URL.createObjectURL(blob);
									const a = $el("a", {
										href: url,
										download: "color_palette.json",
										style: {display: "none"},
										parent: document.body,
									});
									a.click();
									setTimeout(function () {
										a.remove();
										window.URL.revokeObjectURL(url);
									}, 0);
								}
							}),
							$el("input", {
								type: "button",
								value: "Delete",
								onclick: async () => {
									let colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);

									if (colorPalettes[colorPaletteId]) {
										alert("You cannot delete a built-in color palette.");
										return;
									}

									if (colorPaletteId.startsWith("custom_")) {
										colorPaletteId = colorPaletteId.substr(7);
									}

									await deleteCustomColorPalette(colorPaletteId);
								}
							}),
						]),
					]),
				])
			},
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

				let {BACKGROUND_IMAGE, CLEAR_BACKGROUND_COLOR} = palette.colors.litegraph_base;
				if (BACKGROUND_IMAGE === undefined || CLEAR_BACKGROUND_COLOR === undefined) {
					const base = colorPalettes["dark"].colors.litegraph_base;
					BACKGROUND_IMAGE = base.BACKGROUND_IMAGE;
					CLEAR_BACKGROUND_COLOR = base.CLEAR_BACKGROUND_COLOR;
				}
				app.canvas.updateBackground(BACKGROUND_IMAGE, CLEAR_BACKGROUND_COLOR);
			},
		});
	},
});
