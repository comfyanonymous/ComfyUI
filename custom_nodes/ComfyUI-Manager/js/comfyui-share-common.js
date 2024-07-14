import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";
import { CopusShareDialog } from "./comfyui-share-copus.js";
import { OpenArtShareDialog } from "./comfyui-share-openart.js";
import { YouMLShareDialog } from "./comfyui-share-youml.js";

export const SUPPORTED_OUTPUT_NODE_TYPES = [
	"PreviewImage",
	"SaveImage",
	"VHS_VideoCombine",
	"ADE_AnimateDiffCombine",
	"SaveAnimatedWEBP",
	"CR Image Output"
]

var docStyle = document.createElement('style');
docStyle.innerHTML = `
.cm-menu-container {
  column-gap: 20px;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
}

.cm-menu-column {
  display: flex;
  flex-direction: column;
}

.cm-title {
	padding: 10px 10px 0 10p;
	background-color: black;
	text-align: center;
	height: 45px;
}
`;
document.head.appendChild(docStyle);

export function getPotentialOutputsAndOutputNodes(nodes) {
	const potential_outputs = [];
	const potential_output_nodes = [];

	// iterate over the array of nodes to find the ones that are marked as SaveImage
	// TODO: Add support for AnimateDiffCombine, etc. nodes that save videos/gifs, etc.
	for (let i = 0; i < nodes.length; i++) {
		const node = nodes[i];
		if (!SUPPORTED_OUTPUT_NODE_TYPES.includes(node.type)) {
			continue;
		}

		if (node.type === "SaveImage" || node.type === "CR Image Output") {
			// check if node has an 'images' array property
			if (node.hasOwnProperty("images") && Array.isArray(node.images)) {
				// iterate over the images array and add each image to the potential_outputs array
				for (let j = 0; j < node.images.length; j++) {
				    potential_output_nodes.push(node);
					potential_outputs.push({ "type": "image", "image": node.images[j], "title": node.title, "node_id": node.id });
				}
			}
		}
		else if (node.type === "PreviewImage") {
			// check if node has an 'images' array property
			if (node.hasOwnProperty("images") && Array.isArray(node.images)) {
				// iterate over the images array and add each image to the potential_outputs array
				for (let j = 0; j < node.images.length; j++) {
				    potential_output_nodes.push(node);
					potential_outputs.push({ "type": "image", "image": node.images[j], "title": node.title, "node_id": node.id });
				}
			}
		}
		else if (node.type === "VHS_VideoCombine") {
			// check if node has a 'widgets' array property, with type 'image'
			if (node.hasOwnProperty("widgets") && Array.isArray(node.widgets)) {
				// iterate over the widgets array and add each image to the potential_outputs array
				for (let j = 0; j < node.widgets.length; j++) {
					if (node.widgets[j].type === "image") {
						const widgetValue = node.widgets[j].value;
						const parsedURLVals = parseURLPath(widgetValue);

						// ensure that the parsedURLVals have 'filename', 'subfolder', 'type', and 'format' properties
						if (parsedURLVals.hasOwnProperty("filename") && parsedURLVals.hasOwnProperty("subfolder") && parsedURLVals.hasOwnProperty("type") && parsedURLVals.hasOwnProperty("format")) {
							if (parsedURLVals.type !== "output") {
								// TODO
							}
							potential_output_nodes.push(node);
							potential_outputs.push({ "type": "output", 'title': node.title, "node_id": node.id , "output": { "filename": parsedURLVals.filename, "subfolder": parsedURLVals.subfolder, "value": widgetValue, "format": parsedURLVals.format } });
						}
					} else if (node.widgets[j].type === "preview") {
						const widgetValue = node.widgets[j].value;
						const parsedURLVals = widgetValue.params;

						if(!parsedURLVals.format?.startsWith('image')) {
							// video isn't supported format
							continue;
						}

						// ensure that the parsedURLVals have 'filename', 'subfolder', 'type', and 'format' properties
						if (parsedURLVals.hasOwnProperty("filename") && parsedURLVals.hasOwnProperty("subfolder") && parsedURLVals.hasOwnProperty("type") && parsedURLVals.hasOwnProperty("format")) {
							if (parsedURLVals.type !== "output") {
								// TODO
							}
							potential_output_nodes.push(node);
							potential_outputs.push({ "type": "output", 'title': node.title, "node_id": node.id , "output": { "filename": parsedURLVals.filename, "subfolder": parsedURLVals.subfolder, "value": `/view?filename=${parsedURLVals.filename}&subfolder=${parsedURLVals.subfolder}&type=${parsedURLVals.type}&format=${parsedURLVals.format}`, "format": parsedURLVals.format } });
						}
					}
				}
			}
		}
		else if (node.type === "ADE_AnimateDiffCombine") {
			// check if node has a 'widgets' array property, with type 'image'
			if (node.hasOwnProperty("widgets") && Array.isArray(node.widgets)) {
				// iterate over the widgets array and add each image to the potential_outputs array
				for (let j = 0; j < node.widgets.length; j++) {
					if (node.widgets[j].type === "image") {
						const widgetValue = node.widgets[j].value;
						const parsedURLVals = parseURLPath(widgetValue);
						// ensure that the parsedURLVals have 'filename', 'subfolder', 'type', and 'format' properties
						if (parsedURLVals.hasOwnProperty("filename") && parsedURLVals.hasOwnProperty("subfolder") && parsedURLVals.hasOwnProperty("type") && parsedURLVals.hasOwnProperty("format")) {
							if (parsedURLVals.type !== "output") {
								// TODO
								continue;
							}
							potential_output_nodes.push(node);
							potential_outputs.push({ "type": "output", 'title': node.title, "output": { "filename": parsedURLVals.filename, "subfolder": parsedURLVals.subfolder, "type": parsedURLVals.type, "value": widgetValue, "format": parsedURLVals.format } });
						}
					}
				}
			}
		}
		else if (node.type === "SaveAnimatedWEBP") {
			// check if node has an 'images' array property
			if (node.hasOwnProperty("images") && Array.isArray(node.images)) {
				// iterate over the images array and add each image to the potential_outputs array
				for (let j = 0; j < node.images.length; j++) {
				    potential_output_nodes.push(node);
					potential_outputs.push({ "type": "image", "image": node.images[j], "title": node.title });
				}
			}
		}
	}

	// Note: make sure that two arrays are the same length
	return { potential_outputs, potential_output_nodes };
}


export function parseURLPath(urlPath) {
	// Extract the query string from the URL path
	var queryString = urlPath.split('?')[1];

	// Use the URLSearchParams API to parse the query string
	var params = new URLSearchParams(queryString);

	// Create an object to store the parsed parameters
	var parsedParams = {};

	// Iterate over each parameter and add it to the object
	for (var pair of params.entries()) {
		parsedParams[pair[0]] = pair[1];
	}

	// Return the object with the parsed parameters
	return parsedParams;
}


export const shareToEsheep= () => {
	app.graphToPrompt()
	.then(prompt => {
		const nodes = app.graph._nodes
		const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);
		const workflow = prompt['workflow']
		api.fetchApi(`/manager/set_esheep_workflow_and_images`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				workflow: workflow,
				images: potential_outputs
			})
		}).then(response => {
			var domain = window.location.hostname;
			var port = window.location.port;
			port = port || (window.location.protocol === 'http:' ? '80' : window.location.protocol === 'https:' ? '443' : '');
			var full_domin = domain + ':' + port
			window.open('https://www.esheep.com/app/workflow_upload?from_local='+ full_domin, '_blank');
		});
	})
}

export const showCopusShareDialog = () => {
  if (!CopusShareDialog.instance) {
    CopusShareDialog.instance = new CopusShareDialog();
  }

  return app.graphToPrompt()
    .then(prompt => {
      return app.graph._nodes;
    })
    .then(nodes => {
        const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);
        CopusShareDialog.instance.show({ potential_outputs, potential_output_nodes});
    })
}

export const showOpenArtShareDialog = () => {
  if (!OpenArtShareDialog.instance) {
    OpenArtShareDialog.instance = new OpenArtShareDialog();
  }

  return app.graphToPrompt()
    .then(prompt => {
      // console.log({ prompt })
      return app.graph._nodes;
    })
    .then(nodes => {
        const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);
        OpenArtShareDialog.instance.show({ potential_outputs, potential_output_nodes});
    })
}


export const showYouMLShareDialog = () => {
  if (!YouMLShareDialog.instance) {
    YouMLShareDialog.instance = new YouMLShareDialog();
  }

  return app.graphToPrompt()
    .then(prompt => {
      return app.graph._nodes;
    })
    .then(nodes => {
        const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);
        YouMLShareDialog.instance.show(potential_outputs, potential_output_nodes);
    })
}


export const showShareDialog = async (share_option) => {
  if (!ShareDialog.instance) {
    ShareDialog.instance = new ShareDialog(share_option);
  }
  return app.graphToPrompt()
    .then(prompt => {
      // console.log({ prompt })
      return app.graph._nodes;
    })
    .then(nodes => {
      // console.log({ nodes });
      const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);
      if (potential_outputs.length === 0) {
        if (potential_output_nodes.length === 0) {
          // todo: add support for other output node types (animatediff combine, etc.)
          const supported_nodes_string = SUPPORTED_OUTPUT_NODE_TYPES.join(", ");
          alert(`No supported output node found (${supported_nodes_string}). To share this workflow, please add an output node to your graph and re-run your prompt.`);
        } else {
          alert("To share this, first run a prompt. Once it's done, click 'Share'.\n\nNOTE: Images of the Share target can only be selected in the PreviewImage, SaveImage, and VHS_VideoCombine nodes. In the case of VHS_VideoCombine, only the image/gif and image/webp formats are supported.");
        }
        return false;
      }
      ShareDialog.instance.show({ potential_outputs, potential_output_nodes, share_option });
      return true;
    });
}

export class ShareDialogChooser extends ComfyDialog {
	static instance = null;
	constructor() {
		super();
		this.element = $el("div.comfy-modal", {
			parent: document.body, style: {
				'overflow-y': "auto",
			}
		},
			[$el("div.comfy-modal-content",
				{},
				[...this.createButtons()]),
			]);
		this.selectedNodeId = null;
	}
	createButtons() {
		const buttons = [
			{
				key: "openart",
				textContent: "OpenArt AI",
				website: "https://openart.ai/workflows/",
				description: "Share ComfyUI workflows and art on OpenArt.ai",
				onclick: () => {
				  showOpenArtShareDialog();
				  this.close();
				}
			},
			{
				key: "youml",
				textContent: "YouML",
				website: "https://youml.com",
				description: "Share your workflow or transform it into an interactive app on YouML.com",
				onclick: () => {
					showYouMLShareDialog();
				  this.close();
				}
			},
			{
				key: "matrix",
				textContent: "Matrix Server",
				website: "https://app.element.io/#/room/%23comfyui_space%3Amatrix.org",
				description: "Share your art on the official ComfyUI matrix server",
				onclick: async () => {
				  showShareDialog('matrix').then((suc) => {
				    suc && this.close();
				  })
				}
			},
			{
				key: "comfyworkflows",
				textContent: "ComfyWorkflows",
				website: "https://comfyworkflows.com",
				description: "Share & browse thousands of ComfyUI workflows and art 🎨<br/><br/><a style='color:white;' href='https://comfyworkflows.com' target='_blank'>ComfyWorkflows.com</a>",
				onclick: () => {
				  showShareDialog('comfyworkflows').then((suc) => {
				    suc && this.close();
				  })
				}
			},
			{
				key: "esheep",
				textContent: "eSheep",
				website: "https://www.esheep.com",
				description: "Share & download thousands of ComfyUI workflows on <a style='color:white;' href='https://www.esheep.com' target='_blank'>esheep.com</a>",
				onclick: () => {
					shareToEsheep();
				  	this.close();
				}
			},
			{
				key: "Copus",
				textContent: "Copus",
				website: "https://www.copus.io",
				description: "🔴 Permanently store and secure ownership of your workflow on the open-source platform: <a style='color:white;' href='https://copus.io' target='_blank'>Copus.io</a>",
				onclick: () => {
					showCopusShareDialog();
				  this.close();
				}
			},
		];

		function createShareButtonsWithDescriptions() {
			// Responsive container
			const container = $el("div", {
				style: {
					display: "flex",
					'flex-wrap': 'wrap',
					'justify-content': 'space-around',
					'padding': '10px',
				}
			});

			buttons.forEach(b => {
				const button = $el("button", {
					type: "button",
					textContent: b.textContent,
					onclick: b.onclick,
					style: {
						'width': '25%',
						'minWidth': '200px',
						'background-color': b.backgroundColor || '',
						'border-radius': '5px',
						'cursor': 'pointer',
						'padding': '5px 5px',
						'margin-bottom': '5px',
						'transition': 'background-color 0.3s',
					}
				});
				button.addEventListener('mouseover', () => {
					button.style.backgroundColor = '#007BFF'; // Change color on hover
				});
				button.addEventListener('mouseout', () => {
					button.style.backgroundColor = b.backgroundColor || '';
				});

				const description = $el("p", {
					innerHTML: b.description,
					style: {
						'text-align': 'left',
						color: 'white',
						'font-size': '14px',
						'margin-bottom': '0',
					},
				});

				const websiteLink = $el("a", {
					textContent: "🌐 Website",
					href: b.website,
					target: "_blank",
					style: {
						color: 'white',
						'margin-left': '10px',
						'font-size': '12px',
						'text-decoration': 'none',
						'align-self': 'center',
					},
				});

				// Add highlight to the website link
				websiteLink.addEventListener('mouseover', () => {
					websiteLink.style.opacity = '0.7';
				});

				websiteLink.addEventListener('mouseout', () => {
					websiteLink.style.opacity = '1';
				});

				const buttonLinkContainer = $el("div", {
					style: {
						display: 'flex',
						'align-items': 'center',
						'margin-bottom': '10px',
					}
				}, [button, websiteLink]);

				const column = $el("div", {
					style: {
						'flex-basis': '100%',
						'margin': '10px',
						'padding': '10px 20px',
						'border': '1px solid #ddd',
						'border-radius': '5px',
						'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
					}
				}, [buttonLinkContainer, description]);

				container.appendChild(column);
			});

			return container;
		}

		return [
		    $el("p", {
		      textContent: 'Choose a platform to share your workflow',
			    style: {
				  'text-align': 'center',
				  'color': 'white',
				  'font-size': '18px',
				  'margin-bottom': '10px',
				},
			  }
			),

			$el("div.cm-menu-container", {
				id: "comfyui-share-container"
			}, [
				$el("div.cm-menu-column", [
					createShareButtonsWithDescriptions(),
					$el("br", {}, []),
				]),
			]),
			$el("div.cm-menu-container", {
				id: "comfyui-share-container"
			}, [
				$el("button", {
					type: "button",
					style: {
						margin: "0 25px",
						width: "100%",
					},
					textContent: "Close",
					onclick: () => {
						this.close()
					}
				}),
				$el("br", {}, []),
			]),
		];
	}
	show() {
		this.element.style.display = "block";
		this.element.style.zIndex = 10001;
	}
}
export class ShareDialog extends ComfyDialog {
	static instance = null;
	static matrix_auth = { homeserver: "matrix.org", username: "", password: "" };
	static cw_sharekey = "";

	constructor(share_option) {
		super();
		this.share_option = share_option;
		this.element = $el("div.comfy-modal", {
			parent: document.body, style: {
				'overflow-y': "auto",
			}
		},
			[$el("div.comfy-modal-content",
				{},
				[...this.createButtons()]),
			]);
		this.selectedOutputIndex = 0;
	}

	createButtons() {
		this.radio_buttons = $el("div", {
			id: "selectOutputImages",
		}, []);

		this.is_nsfw_checkbox = $el("input", { type: 'checkbox', id: "is_nsfw" }, [])
		const is_nsfw_checkbox_text = $el("label", {
		}, [" Is this NSFW?"])
		this.is_nsfw_checkbox.style.color = "var(--fg-color)";
		this.is_nsfw_checkbox.checked = false;

		this.matrix_destination_checkbox = $el("input", { type: 'checkbox', id: "matrix_destination" }, [])
		const matrix_destination_checkbox_text = $el("label", {}, [" ComfyUI Matrix server"])
		this.matrix_destination_checkbox.style.color = "var(--fg-color)";
		this.matrix_destination_checkbox.checked = this.share_option === 'matrix'; //true;

		this.comfyworkflows_destination_checkbox = $el("input", { type: 'checkbox', id: "comfyworkflows_destination" }, [])
		const comfyworkflows_destination_checkbox_text = $el("label", {}, [" ComfyWorkflows.com"])
		this.comfyworkflows_destination_checkbox.style.color = "var(--fg-color)";
		this.comfyworkflows_destination_checkbox.checked = this.share_option !== 'matrix';

		this.matrix_homeserver_input = $el("input", { type: 'text', id: "matrix_homeserver", placeholder: "matrix.org", value: ShareDialog.matrix_auth.homeserver || 'matrix.org' }, []);
		this.matrix_username_input = $el("input", { type: 'text', placeholder: "Username", value: ShareDialog.matrix_auth.username || '' }, []);
		this.matrix_password_input = $el("input", { type: 'password', placeholder: "Password", value: ShareDialog.matrix_auth.password || '' }, []);

		this.cw_sharekey_input = $el("input", { type: 'text', placeholder: "Share key (found on your profile page)", value: ShareDialog.cw_sharekey || '' }, []);
		this.cw_sharekey_input.style.width = "100%";

		this.credits_input = $el("input", {
			type: "text",
			placeholder: "This will be used to give credits",
			required: false,
		}, []);

		this.title_input = $el("input", {
			type: "text",
			placeholder: "ex: My awesome art",
			required: false
		}, []);

		this.description_input = $el("textarea", {
			placeholder: "ex: Trying out a new workflow... ",
			required: false,
		}, []);

		this.share_button = $el("button", {
			type: "submit",
			textContent: "Share",
			style: {
				backgroundColor: "blue"
			}
		}, []);

		this.final_message = $el("div", {
			style: {
				color: "white",
				textAlign: "center",
				// marginTop: "10px",
				// backgroundColor: "black",
				padding: "10px",
			}
		}, []);

		this.share_finalmessage_container = $el("div.cm-menu-container", {
			id: "comfyui-share-finalmessage-container",
			style: {
				display: "none",
			}
		}, [
			$el("div.cm-menu-column", [
				this.final_message,
				$el("button", {
					type: "button",
					textContent: "Close",
					onclick: () => {
						// Reset state
						this.matrix_destination_checkbox.checked = this.share_option === 'matrix';
						this.comfyworkflows_destination_checkbox.checked = this.share_option !== 'matrix';
						this.share_button.textContent = "Share";
						this.share_button.style.display = "inline-block";
						this.final_message.innerHTML = "";
						this.final_message.style.color = "white";
						this.credits_input.value = "";
						this.title_input.value = "";
						this.description_input.value = "";
						this.is_nsfw_checkbox.checked = false;
						this.selectedOutputIndex = 0;

						// hide the final message
						this.share_finalmessage_container.style.display = "none";

						// show the share container
						this.share_container.style.display = "flex";

						this.close()
					}
				}),
			])
		]);
		this.share_container = $el("div.cm-menu-container", {
			id: "comfyui-share-container"
		}, [
			$el("div.cm-menu-column", [
				$el("details", {
					style: {
						border: "1px solid #999",
						padding: "5px",
						borderRadius: "5px",
						backgroundColor: "#222"
					}
				}, [
					$el("summary", {
						style: {
							color: "white",
							cursor: "pointer",
						}
					}, [`Matrix account`]),
					$el("div", {
						style: {
							display: "flex",
							flexDirection: "row",
						}
					}, [
						$el("div", {
							textContent: "Homeserver",
							style: {
								marginRight: "10px",
							}
						}, []),
						this.matrix_homeserver_input,
					]),

					$el("div", {
						style: {
							display: "flex",
							flexDirection: "row",
						}
					}, [
						$el("div", {
							textContent: "Username",
							style: {
								marginRight: "10px",
							}
						}, []),
						this.matrix_username_input,
					]),

					$el("div", {
						style: {
							display: "flex",
							flexDirection: "row",
						}
					}, [
						$el("div", {
							textContent: "Password",
							style: {
								marginRight: "10px",
							}
						}, []),
						this.matrix_password_input,
					]),

				]),
				$el("details", {
					style: {
						border: "1px solid #999",
						marginTop: "10px",
						padding: "5px",
						borderRadius: "5px",
						backgroundColor: "#222"
					},
				}, [
					$el("summary", {
						style: {
							color: "white",
							cursor: "pointer",
						}
					}, [`Comfyworkflows.com account`]),
					$el("h4", {
						textContent: "Share key (found on your profile page)",
					}, []),
					$el("p", { size: 3, color: "white" }, ["If provided, your art will be saved to your account. Otherwise, it will be shared anonymously."]),
					this.cw_sharekey_input,
				]),

				$el("div", {}, [
					$el("p", {
						size: 3, color: "white", style: {
							color: 'white'
						}
					}, [`Select where to share your art:`]),
					this.matrix_destination_checkbox,
					matrix_destination_checkbox_text,
					$el("br", {}, []),
					this.comfyworkflows_destination_checkbox,
					comfyworkflows_destination_checkbox_text,
				]),

				$el("h4", {
					textContent: "Credits (optional)",
					size: 3,
					color: "white",
					style: {
						color: 'white'
					}
				}, []),
				this.credits_input,
				// $el("br", {}, []),

				$el("h4", {
					textContent: "Title (optional)",
					size: 3,
					color: "white",
					style: {
						color: 'white'
					}
				}, []),
				this.title_input,
				// $el("br", {}, []),

				$el("h4", {
					textContent: "Description (optional)",
					size: 3,
					color: "white",
					style: {
						color: 'white'
					}
				}, []),
				this.description_input,
				$el("br", {}, []),

				$el("div", {}, [this.is_nsfw_checkbox, is_nsfw_checkbox_text]),
				// $el("br", {}, []),

				// this.final_message,
				// $el("br", {}, []),
			]),
			$el("div.cm-menu-column", [
				this.radio_buttons,
				$el("br", {}, []),

				this.share_button,

				$el("button", {
					type: "button",
					textContent: "Close",
					onclick: () => {
						// Reset state
						this.matrix_destination_checkbox.checked = this.share_option === 'matrix';
						this.comfyworkflows_destination_checkbox.checked = this.share_option !== 'matrix';
						this.share_button.textContent = "Share";
						this.share_button.style.display = "inline-block";
						this.final_message.innerHTML = "";
						this.final_message.style.color = "white";
						this.credits_input.value = "";
						this.title_input.value = "";
						this.description_input.value = "";
						this.is_nsfw_checkbox.checked = false;
						this.selectedOutputIndex = 0;

						// hide the final message
						this.share_finalmessage_container.style.display = "none";

						// show the share container
						this.share_container.style.display = "flex";

						this.close()
					}
				}),
				$el("br", {}, []),
			]),
		]);

		// get the user's existing matrix auth and share key
		ShareDialog.matrix_auth = { homeserver: "matrix.org", username: "", password: "" };
		try {
			api.fetchApi(`/manager/get_matrix_auth`)
				.then(response => response.json())
				.then(data => {
					ShareDialog.matrix_auth = data;
					this.matrix_homeserver_input.value = ShareDialog.matrix_auth.homeserver;
					this.matrix_username_input.value = ShareDialog.matrix_auth.username;
					this.matrix_password_input.value = ShareDialog.matrix_auth.password;
				})
				.catch(error => {
					// console.log(error);
				});
		} catch (error) {
			// console.log(error);
		}

		// get the user's existing comfyworkflows share key
		ShareDialog.cw_sharekey = "";
		try {
			// console.log("Fetching comfyworkflows share key")
			api.fetchApi(`/manager/get_comfyworkflows_auth`)
				.then(response => response.json())
				.then(data => {
					ShareDialog.cw_sharekey = data.comfyworkflows_sharekey;
					this.cw_sharekey_input.value = ShareDialog.cw_sharekey;
				})
				.catch(error => {
					// console.log(error);
				});
		} catch (error) {
			// console.log(error);
		}

		this.share_button.onclick = async () => {
			const prompt = await app.graphToPrompt();
			const nodes = app.graph._nodes;

			// console.log({ prompt, nodes });

			const destinations = [];
			if (this.matrix_destination_checkbox.checked) {
				destinations.push("matrix");
			}
			if (this.comfyworkflows_destination_checkbox.checked) {
				destinations.push("comfyworkflows");
			}

			// if destinations includes matrix, make an api call to /manager/check_matrix to ensure that the user has configured their matrix settings
			if (destinations.includes("matrix")) {
				let definedMatrixAuth = !!this.matrix_homeserver_input.value && !!this.matrix_username_input.value && !!this.matrix_password_input.value;
				if (!definedMatrixAuth) {
					alert("Please set your Matrix account details.");
					return;
				}
			}

			if (destinations.includes("comfyworkflows") && !this.cw_sharekey_input.value && false) { //!confirm("You have NOT set your ComfyWorkflows.com share key. Your art will NOT be connected to your account (it will be shared anonymously). Continue?")) {
				return;
			}

			const { potential_outputs, potential_output_nodes } = getPotentialOutputsAndOutputNodes(nodes);

			// console.log({ potential_outputs, potential_output_nodes })

			if (potential_outputs.length === 0) {
				if (potential_output_nodes.length === 0) {
					// todo: add support for other output node types (animatediff combine, etc.)
					const supported_nodes_string = SUPPORTED_OUTPUT_NODE_TYPES.join(", ");
					alert(`No supported output node found (${supported_nodes_string}). To share this workflow, please add an output node to your graph and re-run your prompt.`);
				} else {
					alert("To share this, first run a prompt. Once it's done, click 'Share'.\n\nNOTE: Images of the Share target can only be selected in the PreviewImage, SaveImage, and VHS_VideoCombine nodes. In the case of VHS_VideoCombine, only the image/gif and image/webp formats are supported.");
				}
				this.selectedOutputIndex = 0;
				this.close();
				return;
			}

			// Change the text of the share button to "Sharing..." to indicate that the share process has started
			this.share_button.textContent = "Sharing...";

			const response = await api.fetchApi(`/manager/share`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					matrix_auth: {
						homeserver: this.matrix_homeserver_input.value,
						username: this.matrix_username_input.value,
						password: this.matrix_password_input.value,
					},
					cw_auth: {
						cw_sharekey: this.cw_sharekey_input.value,
					},
					share_destinations: destinations,
					credits: this.credits_input.value,
					title: this.title_input.value,
					description: this.description_input.value,
					is_nsfw: this.is_nsfw_checkbox.checked,
					prompt,
					potential_outputs,
					selected_output_index: this.selectedOutputIndex,
					// potential_output_nodes
				})
			});

			if (response.status != 200) {
				try {
					const response_json = await response.json();
					if (response_json.error) {
						alert(response_json.error);
						this.close();
						return;
					} else {
						alert("Failed to share your art. Please try again.");
						this.close();
						return;
					}
				} catch (e) {
					alert("Failed to share your art. Please try again.");
					this.close();
					return;
				}
			}

			const response_json = await response.json();

			if (response_json.comfyworkflows.url) {
				this.final_message.innerHTML = "Your art has been shared: <a href='" + response_json.comfyworkflows.url + "' target='_blank'>" + response_json.comfyworkflows.url + "</a>";
				if (response_json.matrix.success) {
					this.final_message.innerHTML += "<br>Your art has been shared in the ComfyUI Matrix server's #share channel!";
				}
			} else {
				if (response_json.matrix.success) {
					this.final_message.innerHTML = "Your art has been shared in the ComfyUI Matrix server's #share channel!";
				}
			}

			this.final_message.style.color = "green";

			// hide #comfyui-share-container and show #comfyui-share-finalmessage-container
			this.share_container.style.display = "none";
			this.share_finalmessage_container.style.display = "block";

			// hide the share button
			this.share_button.textContent = "Shared!";
			this.share_button.style.display = "none";
			// this.close();
		}

		const res =
			[
				$el("tr.td", { width: "100%" }, [
					$el("font", { size: 6, color: "white" }, [`Share your art`]),
				]),
				$el("br", {}, []),

				this.share_finalmessage_container,
				this.share_container,
			];

		res[0].style.padding = "10px 10px 10px 10px";
		res[0].style.backgroundColor = "black"; //"linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%)";
		res[0].style.textAlign = "center";
		res[0].style.height = "45px";
		return res;
	}

	show({potential_outputs, potential_output_nodes, share_option}) {
		// Sort `potential_output_nodes` by node ID to make the order always
		// consistent, but we should also keep `potential_outputs` in the same
		// order as `potential_output_nodes`.
		const potential_output_to_order = {};
		potential_output_nodes.forEach((node, index) => {
      if (node.id in potential_output_to_order) {
        potential_output_to_order[node.id][1].push(potential_outputs[index]);
      } else {
        potential_output_to_order[node.id] = [node, [potential_outputs[index]]];
      }
    })
		// Sort the object `potential_output_to_order` by key (node ID)
		const sorted_potential_output_to_order = Object.fromEntries(
			Object.entries(potential_output_to_order).sort((a, b) => a[0].id - b[0].id)
		);
		const sorted_potential_outputs = []
		const sorted_potential_output_nodes = []
		for (const [key, value] of Object.entries(sorted_potential_output_to_order)) {
			sorted_potential_output_nodes.push(value[0]);
			sorted_potential_outputs.push(...value[1]);
		}
		potential_output_nodes = sorted_potential_output_nodes;
		potential_outputs = sorted_potential_outputs;

		// console.log({ potential_outputs, potential_output_nodes })
		this.radio_buttons.innerHTML = ""; // clear the radio buttons
		let is_radio_button_checked = false;  // only check the first radio button if multiple images from the same node
		const new_radio_buttons = $el("div", {
			id: "selectOutput-Options",
			style: {
				'overflow-y': 'scroll',
				'max-height': '400px',
			}
		}, potential_outputs.map((output, index) => {
			const {node_id} = output;
			const radio_button = $el("input", { type: 'radio', name: "selectOutputImages", value: index, required: index === 0 }, [])
			let radio_button_img;
			if (output.type === "image" || output.type === "temp") {
				radio_button_img = $el("img", { src: `/view?filename=${output.image.filename}&subfolder=${output.image.subfolder}&type=${output.image.type}`, style: { width: "auto", height: "100px" } }, []);
			} else if (output.type === "output") {
				radio_button_img = $el("img", { src: output.output.value, style: { width: "auto", height: "100px" } }, []);
			} else {
				// unsupported output type
				// this should never happen
				// TODO
				radio_button_img = $el("img", { src: "", style: { width: "auto", height: "100px" } }, []);
			}
			const radio_button_text = $el("label", {
				// style: {
				// 	color: 'white'
				// }
			}, [output.title])
			radio_button.style.color = "var(--fg-color)";

            // Make the radio button checked if it's the selected node,
			// otherwise make the first radio button checked.
			if (this.selectedNodeId) {
			    if (this.selectedNodeId === node_id && !is_radio_button_checked) {
			        radio_button.checked = true;
							is_radio_button_checked = true;
			    }
			} else {
			    radio_button.checked = index === 0;
			}

			if (radio_button.checked) {
				this.selectedOutputIndex = index;
			}

			radio_button.onchange = () => {
				this.selectedOutputIndex = parseInt(radio_button.value);
			};

			return $el("div", {
				style: {
					display: "flex",
					'align-items': 'center',
					'justify-content': 'space-between',
					'margin-bottom': '10px',
				}
			}, [radio_button, radio_button_text, radio_button_img]);
		}));
		const header = $el("h3", {
			textContent: "Select an image to share",
			size: 3,
			color: "white",
			style: {
				'text-align': 'center',
				color: 'white',
				backgroundColor: 'black',
				padding: '10px',
				'margin-top': '0px',
			}
		}, [
			$el("p", {
				textContent: "Scroll to see all outputs",
				size: 2,
				color: "white",
				style: {
					'text-align': 'center',
					color: 'white',
					'margin-bottom': '5px',
					'font-style': 'italic',
					'font-size': '12px',
				},
			}, [])
		]);
		this.radio_buttons.appendChild(header);
		// this.radio_buttons.appendChild(subheader);
		this.radio_buttons.appendChild(new_radio_buttons);
		this.element.style.display = "block";

		share_option = share_option || this.share_option;
		if (share_option === 'comfyworkflows') {
		    this.matrix_destination_checkbox.checked = false;
            this.comfyworkflows_destination_checkbox.checked = true;
        } else {
            this.matrix_destination_checkbox.checked = true;
            this.comfyworkflows_destination_checkbox.checked = false;
        }
	}
}
