import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";
import {ComfyDialog, $el} from "../../scripts/ui.js";

const BASE_URL = "https://youml.com";
//const BASE_URL = "http://localhost:3000";
const DEFAULT_HOMEPAGE_URL = `${BASE_URL}/?from=comfyui`;
const TOKEN_PAGE_URL = `${BASE_URL}/my-token`;
const API_ENDPOINT = `${BASE_URL}/api`;

const style = `
  .youml-share-dialog {
    overflow-y: auto;
  }
  .youml-share-dialog .dialog-header {
    text-align: center;
    color: white;
    margin: 0 0 10px 0;
  }  
  .youml-share-dialog .dialog-section {
    margin-bottom: 0;
    padding: 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: center;
  }  
  .youml-share-dialog input, .youml-share-dialog textarea {
    display: block;
    min-width: 500px;
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
    border: 1px solid #ddd;
    box-sizing: border-box;
   }
  .youml-share-dialog textarea {
    color: var(--input-text);
    background-color: var(--comfy-input-bg);
  } 
  .youml-share-dialog .workflow-description {
    min-height: 75px;
  } 
  .youml-share-dialog label {
    color: #f8f8f8;
    display: block;
    margin: 5px 0 0 0;
    font-weight: bold;
    text-decoration: none;
  }
  .youml-share-dialog .action-button {  
    padding: 10px 80px;
    margin: 10px 5px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
  }  
  .youml-share-dialog .share-button {
    color: #fff;
    background-color: #007bff;
  }  
  .youml-share-dialog .close-button {  
    background-color: none;
  }
  .youml-share-dialog .action-button-panel {  
    text-align: right;    
    display: flex;
    justify-content: space-between;
  }
  .youml-share-dialog .status-message {  
    color: #fd7909;
    text-align: center;
    padding: 5px;
    font-size: 18px;
  }
  .youml-share-dialog .status-message a {  
    color: white;
  }  
  .youml-share-dialog .output-panel {  
    overflow: auto;
    max-height: 180px;  
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    grid-template-rows: auto;
    grid-column-gap: 10px;
    grid-row-gap: 10px;
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    background-color: var(--bg-color);
  }    
  .youml-share-dialog .output-panel .output-image {  
    width: 100px;
    height: 100px;
    objectFit: cover;
    borderRadius: 5px;
  } 
  
  .youml-share-dialog .output-panel .radio-button {
    color:var(--fg-color);
  }  
  .youml-share-dialog .output-panel .radio-text {  
    color: gray;
    display: block;
    font-size: 12px;
    overflow-x: hidden;
    text-overflow: ellipsis;
    text-wrap: nowrap;
    max-width: 100px;
  }
  .youml-share-dialog .output-panel .node-id {  
    color: #FBFBFD;
    display: block;
    background-color: rgba(0, 0, 0, 0.5);
    font-size: 12px;
    overflow-x: hidden;
    padding: 2px 3px;
    text-overflow: ellipsis;
    text-wrap: nowrap;
    max-width: 100px;
    position: absolute;
    top: 3px;
    left: 3px;
    border-radius: 3px;
  }  
  .youml-share-dialog .output-panel .output-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    cursor: pointer;
    position: relative;
    border: 5px solid transparent;
  }
  .youml-share-dialog .output-panel .output-label:hover {
    border: 5px solid #007bff;
  }
  .youml-share-dialog .output-panel .output-label.checked {
    border: 5px solid #007bff;
  }  
  .youml-share-dialog .missing-output-message{
    color: #fd7909;
    font-size: 16px;  
    margin-bottom:10px  
  }
  .youml-share-dialog .select-output-message{
    color: white;
    margin-bottom:5px  
  }
`;

export class YouMLShareDialog extends ComfyDialog {
  static instance = null;

  constructor() {
    super();
    $el("style", {
      textContent: style,
      parent: document.head,
    });
    this.element = $el(
      "div.comfy-modal.youml-share-dialog",
      {
        parent: document.body,
      },
      [$el("div.comfy-modal-content", {}, [...this.createLayout()])]
    );
    this.selectedOutputIndex = 0;
    this.selectedNodeId = null;
    this.uploadedImages = [];
    this.selectedFile = null;
  }

  async loadToken() {
    let key = ""
    try {
      const response = await api.fetchApi(`/manager/youml/settings`)
      const settings = await response.json()
      return settings.token
    } catch (error) {
    }
    return key || "";
  }

  async saveToken(value) {
    await api.fetchApi(`/manager/youml/settings`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        token: value
      })
    });
  }

  createLayout() {
    // Header Section
    const headerSection = $el("h3.dialog-header", {
      textContent: "Share your workflow to YouML.com",
      size: 3,
    });

    // Workflow Info Section
    this.nameInput = $el("input", {
      type: "text",
      placeholder: "Name (required)",
    });
    this.descriptionInput = $el("textarea.workflow-description", {
      placeholder: "Description (optional, markdown supported)",
    });
    const workflowMetadata = $el("div.dialog-section", {}, [
      $el("label", {}, ["Workflow info"]),
      this.nameInput,
      this.descriptionInput,
    ]);

    // Outputs Section
    this.outputsSection = $el("div.dialog-section", {
      id: "selectOutputs",
    }, []);

    const outputUploadSection = $el("div.dialog-section", {}, [
      $el("label", {}, ["Thumbnail"]),
      this.outputsSection,
    ]);

    // API Token Section
    this.apiTokenInput = $el("input", {
      type: "password",
      placeholder: "Copy & paste your API token",
    });
    const getAPITokenButton = $el("button", {
      href: DEFAULT_HOMEPAGE_URL,
      target: "_blank",
      onclick: () => window.open(TOKEN_PAGE_URL, "_blank"),
    }, ["Get your API Token"])

    const apiTokenSection = $el("div.dialog-section", {}, [
      $el("label", {}, ["YouML API Token"]),
      this.apiTokenInput,
      getAPITokenButton,
    ]);

    // Message Section
    this.message = $el("div.status-message", {}, []);

    // Share and Close Buttons
    this.shareButton = $el("button.action-button.share-button", {
      type: "submit",
      textContent: "Share",
      onclick: () => {
        this.handleShareButtonClick();
      },
    });

    const buttonsSection = $el(
      "div.action-button-panel",
      {},
      [
        $el("button.action-button.close-button", {
          type: "button",
          textContent: "Close",
          onclick: () => {
            this.close();
          },
        }),
        this.shareButton,
      ]
    );

    // Composing the full layout
    const layout = [
      headerSection,
      workflowMetadata,
      outputUploadSection,
      apiTokenSection,
      this.message,
      buttonsSection,
    ];

    return layout;
  }

  async fetchYoumlApi(path, options, statusText) {
    if (statusText) {
      this.message.textContent = statusText;
    }

    const fullPath = new URL(API_ENDPOINT + path)

    const fetchOptions = Object.assign({}, options)

    fetchOptions.headers = {
      ...fetchOptions.headers,
      "Authorization": `Bearer ${this.apiTokenInput.value}`,
      "User-Agent": "ComfyUI-Manager-Youml/1.0.0",
    }

    const response = await fetch(fullPath, fetchOptions);

    if (!response.ok) {
      throw new Error(response.statusText + " " + (await response.text()));
    }

    if (statusText) {
      this.message.textContent = "";
    }
    const data = await response.json();
    return {
      ok: response.ok,
      statusText: response.statusText,
      status: response.status,
      data,
    };
  }

  async uploadThumbnail(uploadFile, recipeId) {
    const form = new FormData();
    form.append("file", uploadFile, uploadFile.name);
    try {
      const res = await this.fetchYoumlApi(
        `/v1/comfy/recipes/${recipeId}/thumbnail`,
        {
          method: "POST",
          body: form,
        },
        "Uploading thumbnail..."
      );

    } catch (e) {
      if (e?.response?.status === 413) {
        throw new Error("File size is too large (max 20MB)");
      } else {
        throw new Error("Error uploading thumbnail: " + e.message);
      }
    }
  }

  async handleShareButtonClick() {
    this.message.textContent = "";
    await this.saveToken(this.apiTokenInput.value);
    try {
      this.shareButton.disabled = true;
      this.shareButton.textContent = "Sharing...";
      await this.share();
    } catch (e) {
      alert(e.message);
    } finally {
      this.shareButton.disabled = false;
      this.shareButton.textContent = "Share";
    }
  }

  async share() {
    const prompt = await app.graphToPrompt();
    const workflowJSON = prompt["workflow"];
    const workflowAPIJSON = prompt["output"];
    const form_values = {
      name: this.nameInput.value,
      description: this.descriptionInput.value,
    };

    if (!this.apiTokenInput.value) {
      throw new Error("API token is required");
    }

    if (!this.selectedFile) {
      throw new Error("Thumbnail is required");
    }

    if (!form_values.name) {
      throw new Error("Title is required");
    }


    try {
      let snapshotData = null;
      try {
        const snapshot = await api.fetchApi(`/snapshot/get_current`)
        snapshotData = await snapshot.json()
      } catch (e) {
        console.error("Failed to get snapshot", e)
      }

      const request = {
        name: this.nameInput.value,
        description: this.descriptionInput.value,
        workflowUiJson: JSON.stringify(workflowJSON),
        workflowApiJson: JSON.stringify(workflowAPIJSON),
      }

      if (snapshotData) {
        request.snapshotJson = JSON.stringify(snapshotData)
      }

      const response = await this.fetchYoumlApi(
        "/v1/comfy/recipes",
        {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(request),
        },
        "Uploading workflow..."
      );

      if (response.ok) {
        const {id, recipePageUrl, editorPageUrl} = response.data;
        if (id) {
          let messagePrefix = "Workflow has been shared."
          if (this.selectedFile) {
            try {
              await this.uploadThumbnail(this.selectedFile, id);
            } catch (e) {
              console.error("Thumbnail upload failed: ", e);
              messagePrefix = "Workflow has been shared, but thumbnail upload failed. You can create a thumbnail on YouML later."
            }
          }
          this.message.innerHTML = `${messagePrefix} To turn your workflow into an interactive app, ` +
            `<a href="${recipePageUrl}" target="_blank">visit it on YouML</a>`;

          this.uploadedImages = [];
          this.nameInput.value = "";
          this.descriptionInput.value = "";
          this.radioButtons.forEach((ele) => {
            ele.checked = false;
            ele.parentElement.classList.remove("checked");
          });
          this.selectedOutputIndex = 0;
          this.selectedNodeId = null;
          this.selectedFile = null;
        }
      }
    } catch (e) {
      throw new Error("Error sharing workflow: " + e.message);
    }
  }

  async fetchImageBlob(url) {
    const response = await fetch(url);
    const blob = await response.blob();
    return blob;
  }

  async show(potentialOutputs, potentialOutputNodes) {
    const potentialOutputsToOrder = {};
    potentialOutputNodes.forEach((node, index) => {
      if (node.id in potentialOutputsToOrder) {
        potentialOutputsToOrder[node.id][1].push(potentialOutputs[index]);
      } else {
        potentialOutputsToOrder[node.id] = [node, [potentialOutputs[index]]];
      }
    })
    const sortedPotentialOutputsToOrder = Object.fromEntries(
      Object.entries(potentialOutputsToOrder).sort((a, b) => a[0].id - b[0].id)
    );
    const sortedPotentialOutputs = []
    const sortedPotentiaOutputNodes = []
    for (const [key, value] of Object.entries(sortedPotentialOutputsToOrder)) {
      sortedPotentiaOutputNodes.push(value[0]);
      sortedPotentialOutputs.push(...value[1]);
    }
    potentialOutputNodes = sortedPotentiaOutputNodes;
    potentialOutputs = sortedPotentialOutputs;


    // If `selectedNodeId` is provided, we will select the corresponding radio
    // button for the node. In addition, we move the selected radio button to
    // the top of the list.
    if (this.selectedNodeId) {
      const index = potentialOutputNodes.findIndex(node => node.id === this.selectedNodeId);
      if (index >= 0) {
        this.selectedOutputIndex = index;
      }
    }

    this.radioButtons = [];
    const newRadioButtons = $el("div.output-panel",
      {
        id: "selectOutput-Options",
      },
      potentialOutputs.map((output, index) => {
        const {node_id: nodeId} = output;
        const radioButton = $el("input.radio-button", {
          type: "radio",
          name: "selectOutputImages",
          value: index,
          required: index === 0
        }, [])
        let radioButtonImage;
        let filename;
        if (output.type === "image" || output.type === "temp") {
          radioButtonImage = $el("img.output-image", {
            src: `/view?filename=${output.image.filename}&subfolder=${output.image.subfolder}&type=${output.image.type}`,
          }, []);
          filename = output.image.filename
        } else if (output.type === "output") {
          radioButtonImage = $el("img.output-image", {
            src: output.output.value,
          }, []);
          filename = output.output.filename
        } else {
          radioButtonImage = $el("img.output-image", {
            src: "",
          }, []);
        }
        const radioButtonText = $el("span.radio-text", {}, [output.title])
        const nodeIdChip = $el("span.node-id", {}, [`Node: ${nodeId}`])
        radioButton.checked = this.selectedOutputIndex === index;

        radioButton.onchange = async () => {
          this.selectedOutputIndex = parseInt(radioButton.value);

          // Remove the "checked" class from all radio buttons
          this.radioButtons.forEach((ele) => {
            ele.parentElement.classList.remove("checked");
          });
          radioButton.parentElement.classList.add("checked");

          this.fetchImageBlob(radioButtonImage.src).then((blob) => {
            const file = new File([blob], filename, {
              type: blob.type,
            });
            this.selectedFile = file;
          })
        };

        if (radioButton.checked) {
          this.fetchImageBlob(radioButtonImage.src).then((blob) => {
            const file = new File([blob], filename, {
              type: blob.type,
            });
            this.selectedFile = file;
          })
        }

        this.radioButtons.push(radioButton);

        return $el(`label.output-label${radioButton.checked ? '.checked' : ''}`, {},
          [radioButtonImage, radioButtonText, radioButton, nodeIdChip]);
      })
    );

    let header;
    if (this.radioButtons.length === 0) {
      header = $el("div.missing-output-message", {textContent: "Queue Prompt to see the outputs and select a thumbnail"}, [])
    } else {
      header = $el("div.select-output-message", {textContent: "Choose one from the outputs (scroll to see all)"}, [])
    }

    this.outputsSection.innerHTML = "";
    this.outputsSection.appendChild(header);
    if (this.radioButtons.length > 0) {
      this.outputsSection.appendChild(newRadioButtons);
    }

    this.message.innerHTML = "";
    this.message.textContent = "";

    const token = await this.loadToken();
    this.apiTokenInput.value = token;
    this.uploadedImages = [];

    this.element.style.display = "block";
  }
}
