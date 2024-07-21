import { app } from "../../scripts/app.js"
import { api } from "../../scripts/api.js"

function splitFilePath(path) {
  const folder_separator = path.lastIndexOf("/")
  if (folder_separator === -1) {
    return ["", path]
  }
  return [
    path.substring(0, folder_separator),
    path.substring(folder_separator + 1)
  ]
}

function getResourceURL(subfolder, filename, type = "input") {
  const params = [
    "filename=" + encodeURIComponent(filename),
    "type=" + type,
    "subfolder=" + subfolder,
    app.getRandParam().substring(1)
  ].join("&")

  return `/view?${params}`
}

async function uploadFile(
  audioWidget,
  audioUIWidget,
  file,
  updateNode,
  pasted = false
) {
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData()
    body.append("image", file)
    if (pasted) body.append("subfolder", "pasted")
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body
    })

    if (resp.status === 200) {
      const data = await resp.json()
      // Add the file to the dropdown list and update the widget value
      let path = data.name
      if (data.subfolder) path = data.subfolder + "/" + path

      if (!audioWidget.options.values.includes(path)) {
        audioWidget.options.values.push(path)
      }

      if (updateNode) {
        audioUIWidget.element.src = api.apiURL(
          getResourceURL(...splitFilePath(path))
        )
        audioWidget.value = path
      }
    } else {
      alert(resp.status + " - " + resp.statusText)
    }
  } catch (error) {
    alert(error)
  }
}

// AudioWidget MUST be registered first, as AUDIOUPLOAD depends on AUDIO_UI to be
// present.
app.registerExtension({
  name: "Comfy.AudioWidget",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (["LoadAudio", "SaveAudio", "PreviewAudio"].includes(nodeType.comfyClass)) {
      nodeData.input.required.audioUI = ["AUDIO_UI"]
    }
  },
  getCustomWidgets() {
    return {
      AUDIO_UI(node, inputName) {
        const audio = document.createElement("audio")
        audio.controls = true
        audio.classList.add("comfy-audio")
        audio.setAttribute("name", "media")

        const audioUIWidget = node.addDOMWidget(
          inputName,
          /* name=*/ "audioUI",
          audio
        )
        // @ts-ignore
        // TODO: Sort out the DOMWidget type.
        audioUIWidget.serialize = false

        const isOutputNode = node.constructor.nodeData.output_node
        if (isOutputNode) {
          // Hide the audio widget when there is no audio initially.
          audioUIWidget.element.classList.add("empty-audio-widget")
          // Populate the audio widget UI on node execution.
          const onExecuted = node.onExecuted
          node.onExecuted = function(message) {
            onExecuted?.apply(this, arguments)
            const audios = message.audio
            if (!audios) return
            const audio = audios[0]
            audioUIWidget.element.src = api.apiURL(
              getResourceURL(audio.subfolder, audio.filename, audio.type)
            )
            audioUIWidget.element.classList.remove("empty-audio-widget")
          }
        }
        return { widget: audioUIWidget }
      }
    }
  },
  onNodeOutputsUpdated(nodeOutputs) {
    for (const [nodeId, output] of Object.entries(nodeOutputs)) {
      const node = app.graph.getNodeById(Number.parseInt(nodeId));
      if ("audio" in output) {
        const audioUIWidget = node.widgets.find((w) => w.name === "audioUI");
        const audio = output.audio[0];
        audioUIWidget.element.src = api.apiURL(getResourceURL(audio.subfolder, audio.filename, audio.type));
        audioUIWidget.element.classList.remove("empty-audio-widget");
      }
    }
  },
})

app.registerExtension({
  name: "Comfy.UploadAudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.input?.required?.audio?.[1]?.audio_upload === true) {
      nodeData.input.required.upload = ["AUDIOUPLOAD"]
    }
  },
  getCustomWidgets() {
    return {
      AUDIOUPLOAD(node, inputName) {
        // The widget that allows user to select file.
        const audioWidget = node.widgets.find(w => w.name === "audio")
        const audioUIWidget = node.widgets.find(w => w.name === "audioUI")

        const onAudioWidgetUpdate = () => {
          audioUIWidget.element.src = api.apiURL(
            getResourceURL(...splitFilePath(audioWidget.value))
          )
        }
        // Initially load default audio file to audioUIWidget.
        if (audioWidget.value) {
          onAudioWidgetUpdate()
        }
        audioWidget.callback = onAudioWidgetUpdate

        // Load saved audio file widget values if restoring from workflow
        const onGraphConfigured = node.onGraphConfigured;
        node.onGraphConfigured = function() {
          onGraphConfigured?.apply(this, arguments)
          if (audioWidget.value) {
            onAudioWidgetUpdate()
          }
        }

        const fileInput = document.createElement("input")
        fileInput.type = "file"
        fileInput.accept = "audio/*"
        fileInput.style.display = "none"
        fileInput.onchange = () => {
          if (fileInput.files.length) {
            uploadFile(audioWidget, audioUIWidget, fileInput.files[0], true)
          }
        }
        // The widget to pop up the upload dialog.
        const uploadWidget = node.addWidget(
          "button",
          inputName,
          /* value=*/ "",
          () => {
            fileInput.click()
          }
        )
        uploadWidget.label = "choose file to upload"
        uploadWidget.serialize = false

        return { widget: uploadWidget }
      }
    }
  }
})
