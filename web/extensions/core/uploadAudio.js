import { app } from "../../scripts/app"
import { api } from "../../scripts/api"

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

function getResourceURL(filename, subfolder, type = "input") {
  const params = [
    "filename=" + encodeURIComponent(filename),
    "type=" + type,
    "subfolder=" + subfolder,
    app.getPreviewFormatParam().substring(1),
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
    if (["LoadAudio", "SaveAudio"].includes(nodeType.comfyClass)) {
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
        audioUIWidget.serialize = false

        const isOutputNode = node.constructor.nodeData.output_node
        if (isOutputNode) {
          const onExecuted = node.onExecuted
          node.onExecuted = function(message) {
            onExecuted?.apply(this, arguments)
            const audios = message.audio
            if (!audios) return
            const audio = audios[0]
            audioUIWidget.element.src = api.apiURL(
              getResourceURL(audio.filename, audio.subfolder, "output")
            )
          }
        }
        return { widget: audioUIWidget }
      }
    }
  }
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

        audioWidget.callback = function() {
          audioUIWidget.element.src = api.apiURL(
            getResourceURL(...splitFilePath(audioWidget.value))
          )
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
