import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";
import {ComfyDialog, $el} from "../../scripts/ui.js";

const LOCAL_STORAGE_KEY = "openart_comfy_workflow_key";
const DEFAULT_HOMEPAGE_URL = "https://openart.ai/workflows/dev?developer=true";
//const DEFAULT_HOMEPAGE_URL = "http://localhost:8080/workflows/dev?developer=true";

const API_ENDPOINT = "https://openart.ai/api";
//const API_ENDPOINT = "http://localhost:8080/api";

const style = `
  .openart-share-dialog a {
    color: #f8f8f8;
  }
  .openart-share-dialog a:hover {
    color: #007bff;
  }
  .output_label {
    border: 5px solid transparent;
  }
  .output_label:hover {
    border: 5px solid #59E8C6;
  }
  .output_label.checked {
    border: 5px solid #59E8C6;
  }
`;

// Shared component styles
const sectionStyle = {
  marginBottom: 0,
  padding: 0,
  borderRadius: "8px",
  boxShadow: "0 2px 4px rgba(0, 0, 0, 0.05)",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
};

export class OpenArtShareDialog extends ComfyDialog {
  static instance = null;

  constructor() {
    super();
    $el("style", {
      textContent: style,
      parent: document.head,
    });
    this.element = $el(
      "div.comfy-modal.openart-share-dialog",
      {
        parent: document.body,
        style: {
          "overflow-y": "auto",
        },
      },
      [$el("div.comfy-modal-content", {}, [...this.createButtons()])]
    );
    this.selectedOutputIndex = 0;
    this.selectedNodeId = null;
    this.uploadedImages = [];
    this.selectedFile = null;
  }

  async readKey() {
    let key = ""
    try {
      key = await api.fetchApi(`/manager/get_openart_auth`)
        .then(response => response.json())
        .then(data => {
          return data.openart_key;
        })
        .catch(error => {
          // console.log(error);
        });
    } catch (error) {
      // console.log(error);
    }
    return key || "";
  }

  async saveKey(value) {
    await api.fetchApi(`/manager/set_openart_auth`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        openart_key: value
      })
    });
  }

  createButtons() {
    const inputStyle = {
      display: "block",
      minWidth: "500px",
      width: "100%",
      padding: "10px",
      margin: "10px 0",
      borderRadius: "4px",
      border: "1px solid #ddd",
      boxSizing: "border-box",
    };

    const hyperLinkStyle = {
      display: "block",
      marginBottom: "15px",
      fontWeight: "bold",
      fontSize: "14px",
    };

    const labelStyle = {
      color: "#f8f8f8",
      display: "block",
      margin: "10px 0 0 0",
      fontWeight: "bold",
      textDecoration: "none",
    };

    const buttonStyle = {
      padding: "10px 80px",
      margin: "10px 5px",
      borderRadius: "4px",
      border: "none",
      cursor: "pointer",
      color: "#fff",
      backgroundColor: "#007bff",
    };

    // upload images input
    this.uploadImagesInput = $el("input", {
      type: "file",
      multiple: false,
      style: inputStyle,
      accept: "image/*",
    });

    this.uploadImagesInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) {
        this.previewImage.src = "";
        this.previewImage.style.display = "none";
        return;
      }
      const reader = new FileReader();
      reader.onload = async (e) => {
        const imgData = e.target.result;
        this.previewImage.src = imgData;
        this.previewImage.style.display = "block";
        this.selectedFile = null
        // Once user uploads an image, we uncheck all radio buttons
        this.radioButtons.forEach((ele) => {
          ele.checked = false;
          ele.parentElement.classList.remove("checked");
        });

        // Add the opacity style toggle here to indicate that they only need
        // to upload one image or choose one from the outputs.
        this.outputsSection.style.opacity = 0.35;
        this.uploadImagesInput.style.opacity = 1;
      };
      reader.readAsDataURL(file);
    });

    // preview image
    this.previewImage = $el("img", {
      src: "",
      style: {
        width: "100%",
        maxHeight: "100px",
        objectFit: "contain",
        display: "none",
        marginTop: '10px',
      },
    });

    this.keyInput = $el("input", {
      type: "password",
      placeholder: "Copy & paste your API key",
      style: inputStyle,
    });
    this.NameInput = $el("input", {
      type: "text",
      placeholder: "Title (required)",
      style: inputStyle,
    });
    this.descriptionInput = $el("textarea", {
      placeholder: "Description (optional)",
      style: {
        ...inputStyle,
        minHeight: "100px",
      },
    });

    // Header Section
    const headerSection = $el("h3", {
      textContent: "Share your workflow to OpenArt",
      size: 3,
      color: "white",
      style: {
        'text-align': 'center',
        color: 'white',
        margin: '0 0 10px 0',
      }
    });

    // LinkSection
    this.communityLink = $el("a", {
      style: hyperLinkStyle,
      href: DEFAULT_HOMEPAGE_URL,
      target: "_blank"
    }, ["👉 Check out thousands of workflows shared from the community"])
    this.getAPIKeyLink = $el("a", {
      style: {
        ...hyperLinkStyle,
        color: "#59E8C6"
      },
      href: DEFAULT_HOMEPAGE_URL,
      target: "_blank"
    }, ["👉 Get your API key here"])
    const linkSection = $el(
      "div",
      {
        style: {
          marginTop: "10px",
          display: "flex",
          flexDirection: "column",
        },
      },
      [
        this.communityLink,
        this.getAPIKeyLink,
      ]
    );

    // Account Section
    const accountSection = $el("div", {style: sectionStyle}, [
      $el("label", {style: labelStyle}, ["1️⃣ OpenArt API Key"]),
      this.keyInput,
    ]);

    // Output Upload Section
    const outputUploadSection = $el("div", {style: sectionStyle}, [
      $el("label", {
        style: {
          ...labelStyle,
          margin: "10px 0 0 0"
        }
      }, ["2️⃣ Image/Thumbnail (Required)"]),
      this.previewImage,
      this.uploadImagesInput,
    ]);

    // Outputs Section
    this.outputsSection = $el("div", {
      id: "selectOutputs",
    }, []);

    // Additional Inputs Section
    const additionalInputsSection = $el("div", {style: sectionStyle}, [
      $el("label", {style: labelStyle}, ["3️⃣ Workflow Information"]),
      this.NameInput,
      this.descriptionInput,
    ]);

    // OpenArt Contest Section
    /*
    this.joinContestCheckbox = $el("input", {
      type: 'checkbox',
      id: "join_contest"s
    }, [])
    this.joinContestDescription = $el("a", {
      style: {
        ...hyperLinkStyle,
        display: 'inline-block',
        color: "#59E8C6",
        fontSize: '12px',
        marginLeft: '10px',
        marginBottom: 0,
      },
      href: "https://contest.openart.ai/",
      target: "_blank"
    }, ["🏆 I'm participating in the OpenArt workflow contest"])
    this.joinContestLabel = $el("label", {
      style: {
        display: 'flex',
        alignItems: 'center',
        cursor: 'pointer',
      }
    }, [this.joinContestCheckbox, this.joinContestDescription])
    const contestSection = $el("div", {style: sectionStyle}, [
      this.joinContestLabel,
    ]);
    */

    // Message Section
    this.message = $el(
      "div",
      {
        style: {
          color: "#ff3d00",
          textAlign: "center",
          padding: "10px",
          fontSize: "20px",
        },
      },
      []
    );

    this.shareButton = $el("button", {
      type: "submit",
      textContent: "Share",
      style: buttonStyle,
      onclick: () => {
        this.handleShareButtonClick();
      },
    });

    // Share and Close Buttons
    const buttonsSection = $el(
      "div",
      {
        style: {
          textAlign: "right",
          marginTop: "20px",
          display: "flex",
          justifyContent: "space-between",
        },
      },
      [
        $el("button", {
          type: "button",
          textContent: "Close",
          style: {
            ...buttonStyle,
            backgroundColor: undefined,
          },
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
      linkSection,
      accountSection,
      outputUploadSection,
      this.outputsSection,
      additionalInputsSection,
      // contestSection,
      this.message,
      buttonsSection,
    ];

    return layout;
  }

  async fetchApi(path, options, statusText) {
    if (statusText) {
      this.message.textContent = statusText;
    }
    const addSearchParams = (url, params = {}) =>
      new URL(
        `${url.origin}${url.pathname}?${new URLSearchParams([
          ...Array.from(url.searchParams.entries()),
          ...Object.entries(params),
        ])}`
      );

    const fullPath = addSearchParams(new URL(API_ENDPOINT + path), {
      workflow_api_key: this.keyInput.value,
    });

    const response = await fetch(fullPath, options);

    if (!response.ok) {
      throw new Error(response.statusText);
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

  async uploadThumbnail(uploadFile) {
    const form = new FormData();
    form.append("file", uploadFile);
    try {
      const res = await this.fetchApi(
        `/workflows/upload_thumbnail`,
        {
          method: "POST",
          body: form,
        },
        "Uploading thumbnail..."
      );

      if (res.ok && res.data) {
        const {image_url, width, height} = res.data;
        this.uploadedImages.push({
          url: image_url,
          width,
          height,
        });
      }
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
    await this.saveKey(this.keyInput.value);
    try {
      this.shareButton.disabled = true;
      this.shareButton.textContent = "Sharing...";
      await this.share();
    } catch (e) {
      alert(e.message);
    }
    this.shareButton.disabled = false;
    this.shareButton.textContent = "Share";
  }

  async share() {
    const prompt = await app.graphToPrompt();
    const workflowJSON = prompt["workflow"];
    const workflowAPIJSON = prompt["output"];
    const form_values = {
      name: this.NameInput.value,
      description: this.descriptionInput.value,
    };

    if (!this.keyInput.value) {
      throw new Error("API key is required");
    }

    if (!this.uploadImagesInput.files[0] && !this.selectedFile) {
      throw new Error("Thumbnail is required");
    }

    if (!form_values.name) {
      throw new Error("Title is required");
    }

    const current_snapshot = await api.fetchApi(`/snapshot/get_current`)
      .then(response => response.json())
      .catch(error => {
        // console.log(error);
      });


    if (!this.uploadedImages.length) {
      if (this.selectedFile) {
        await this.uploadThumbnail(this.selectedFile);
      } else {
        for (const file of this.uploadImagesInput.files) {
          try {
            await this.uploadThumbnail(file);
          } catch (e) {
            this.uploadedImages = [];
            throw new Error(e.message);
          }
        }

        if (this.uploadImagesInput.files.length === 0) {
          throw new Error("No thumbnail uploaded");
        }
      }
    }

    // const join_contest = this.joinContestCheckbox.checked;

    try {
      const response = await this.fetchApi(
        "/workflows/publish",
        {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            workflow_json: workflowJSON,
            upload_images: this.uploadedImages,
            form_values,
            advanced_config: {
              workflow_api_json: workflowAPIJSON,
              snapshot: current_snapshot,
            },
            // join_contest,
          }),
        },
        "Uploading workflow..."
      );

      if (response.ok) {
        const {workflow_id} = response.data;
        if (workflow_id) {
          const url = `https://openart.ai/workflows/-/-/${workflow_id}`;
          this.message.innerHTML = `Workflow has been shared successfully. <a href="${url}" target="_blank">Click here to view it.</a>`;
          this.previewImage.src = "";
          this.previewImage.style.display = "none";
          this.uploadedImages = [];
          this.NameInput.value = "";
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

  async show({potential_outputs, potential_output_nodes} = {}) {
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

    this.message.innerHTML = "";
    this.message.textContent = "";
    this.element.style.display = "block";
    this.previewImage.src = "";
    this.previewImage.style.display = "none";
    const key = await this.readKey();
    this.keyInput.value = key;
    this.uploadedImages = [];

    // If `selectedNodeId` is provided, we will select the corresponding radio
    // button for the node. In addition, we move the selected radio button to
    // the top of the list.
    if (this.selectedNodeId) {
      const index = potential_output_nodes.findIndex(node => node.id === this.selectedNodeId);
      if (index >= 0) {
        this.selectedOutputIndex = index;
      }
    }

    this.radioButtons = [];
    const new_radio_buttons = $el("div",
      {
        id: "selectOutput-Options",
        style: {
          'overflow-y': 'scroll',
          'max-height': '200px',

          'display': 'grid',
          'grid-template-columns': 'repeat(auto-fit, minmax(100px, 1fr))',
          'grid-template-rows': 'auto',
          'grid-column-gap': '10px',
          'grid-row-gap': '10px',
          'margin-bottom': '10px',
          'padding': '10px',
          'border-radius': '8px',
          'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
          'background-color': 'var(--bg-color)',
        }
      },
      potential_outputs.map((output, index) => {
        const {node_id} = output;
        const radio_button = $el("input", {
          type: 'radio',
          name: "selectOutputImages",
          value: index,
          required: index === 0
        }, [])
        let radio_button_img;
        let filename;
        if (output.type === "image" || output.type === "temp") {
          radio_button_img = $el("img", {
            src: `/view?filename=${output.image.filename}&subfolder=${output.image.subfolder}&type=${output.image.type}`,
            style: {
              width: "100px",
              height: "100px",
              objectFit: "cover",
              borderRadius: "5px"
            }
          }, []);
          filename = output.image.filename
        } else if (output.type === "output") {
          radio_button_img = $el("img", {
            src: output.output.value,
            style: {
              width: "auto",
              height: "100px",
              objectFit: "cover",
              borderRadius: "5px"
            }
          }, []);
          filename = output.filename
        } else {
          // unsupported output type
          // this should never happen
          // TODO
          radio_button_img = $el("img", {
            src: "",
            style: {width: "auto", height: "100px"}
          }, []);
        }
        const radio_button_text = $el("span", {
          style: {
            color: 'gray',
            display: 'block',
            fontSize: '12px',
            overflowX: 'hidden',
            textOverflow: 'ellipsis',
            textWrap: 'nowrap',
            maxWidth: '100px',
          }
        }, [output.title])
        const node_id_chip = $el("span", {
          style: {
            color: '#FBFBFD',
            display: 'block',
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            fontSize: '12px',
            overflowX: 'hidden',
            padding: '2px 3px',
            textOverflow: 'ellipsis',
            textWrap: 'nowrap',
            maxWidth: '100px',
            position: 'absolute',
            top: '3px',
            left: '3px',
            borderRadius: '3px',
          }
        }, [`Node: ${node_id}`])
        radio_button.style.color = "var(--fg-color)";
        radio_button.checked = this.selectedOutputIndex === index;

        radio_button.onchange = async () => {
          this.selectedOutputIndex = parseInt(radio_button.value);

          // Remove the "checked" class from all radio buttons
          this.radioButtons.forEach((ele) => {
            ele.parentElement.classList.remove("checked");
          });
          radio_button.parentElement.classList.add("checked");

          this.fetchImageBlob(radio_button_img.src).then((blob) => {
            const file = new File([blob], filename, {
              type: blob.type,
            });
            this.previewImage.src = radio_button_img.src;
            this.previewImage.style.display = "block";
            this.selectedFile = file;
          })

          // Add the opacity style toggle here to indicate that they only need
          // to upload one image or choose one from the outputs.
          this.outputsSection.style.opacity = 1;
          this.uploadImagesInput.style.opacity = 0.35;
        };

        if (radio_button.checked) {
          this.fetchImageBlob(radio_button_img.src).then((blob) => {
            const file = new File([blob], filename, {
              type: blob.type,
            });
            this.previewImage.src = radio_button_img.src;
            this.previewImage.style.display = "block";
            this.selectedFile = file;
          })
          // Add the opacity style toggle here to indicate that they only need
          // to upload one image or choose one from the outputs.
          this.outputsSection.style.opacity = 1;
          this.uploadImagesInput.style.opacity = 0.35;
        }

        this.radioButtons.push(radio_button);

        return $el(`label.output_label${radio_button.checked ? '.checked' : ''}`, {
          style: {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "10px",
            cursor: "pointer",
            position: 'relative',
          }
        }, [radio_button_img, radio_button_text, radio_button, node_id_chip]);
      })
    );

    const header =
      $el("p", {
        textContent: this.radioButtons.length === 0 ? "Queue Prompt to see the outputs" : "Or choose one from the outputs (scroll to see all)",
        size: 2,
        color: "white",
        style: {
          color: 'white',
          margin: '0 0 5px 0',
          fontSize: '12px',
        },
      }, [])
    this.outputsSection.innerHTML = "";
    this.outputsSection.appendChild(header);
    this.outputsSection.appendChild(new_radio_buttons);
  }
}
