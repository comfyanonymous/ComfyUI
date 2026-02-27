import { MediaRecorder as ExtendableMediaRecorder } from 'extendable-media-recorder'

import { useNodeDragAndDrop } from '@/composables/node/useNodeDragAndDrop'
import { useNodeFileInput } from '@/composables/node/useNodeFileInput'
import { useNodePaste } from '@/composables/node/useNodePaste'
import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { resolveNodeRootGraphId } from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  IStringWidget
} from '@/lib/litegraph/src/types/widgets'
import { useToastStore } from '@/platform/updates/common/toastStore'
import {
  getResourceURL,
  splitFilePath
} from '@/renderer/extensions/vueNodes/widgets/utils/audioUtils'
import type { NodeExecutionOutput } from '@/schemas/apiSchema'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import type { DOMWidget } from '@/scripts/domWidget'
import { useAudioService } from '@/services/audioService'
import { type NodeLocatorId } from '@/types'
import { getNodeByLocatorId } from '@/utils/graphTraversalUtil'

import { api } from '../../scripts/api'
import { app } from '../../scripts/app'
import { useWidgetValueStore } from '@/stores/widgetValueStore'

function updateUIWidget(
  audioUIWidget: DOMWidget<HTMLAudioElement, string>,
  url: string = ''
) {
  audioUIWidget.element.src = url
  audioUIWidget.value = url
  audioUIWidget.callback?.(url)
  if (url) audioUIWidget.element.classList.remove('empty-audio-widget')
  else audioUIWidget.element.classList.add('empty-audio-widget')
}

async function uploadFile(
  audioWidget: IStringWidget,
  audioUIWidget: DOMWidget<HTMLAudioElement, string>,
  file: File,
  updateNode: boolean,
  pasted: boolean = false
): Promise<boolean> {
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData()
    body.append('image', file)
    if (pasted) body.append('subfolder', 'pasted')
    const resp = await api.fetchApi('/upload/image', {
      method: 'POST',
      body
    })

    if (resp.status === 200) {
      const data = await resp.json()
      // Add the file to the dropdown list and update the widget value
      let path = data.name
      if (data.subfolder) path = data.subfolder + '/' + path

      // @ts-expect-error fixme ts strict error
      if (!audioWidget.options.values.includes(path)) {
        // @ts-expect-error fixme ts strict error
        audioWidget.options.values.push(path)
      }

      if (updateNode) {
        updateUIWidget(
          audioUIWidget,
          api.apiURL(getResourceURL(...splitFilePath(path)))
        )

        audioWidget.value = path
        // Manually trigger the callback to update VueNodes
        audioWidget.callback?.(path)
      }
      return true
    } else {
      useToastStore().addAlert(resp.status + ' - ' + resp.statusText)
      return false
    }
  } catch (error) {
    // @ts-expect-error fixme ts strict error
    useToastStore().addAlert(error)
    return false
  }
}

// AudioWidget MUST be registered first, as AUDIOUPLOAD depends on AUDIO_UI to be
// present.
app.registerExtension({
  name: 'Comfy.AudioWidget',
  async beforeRegisterNodeDef(
    nodeType: typeof LGraphNode,
    nodeData: ComfyNodeDef
  ) {
    if (
      [
        'LoadAudio',
        'SaveAudio',
        'PreviewAudio',
        'SaveAudioMP3',
        'SaveAudioOpus'
      ].includes(
        // @ts-expect-error fixme ts strict error
        nodeType.prototype.comfyClass
      )
    ) {
      // @ts-expect-error fixme ts strict error
      nodeData.input.required.audioUI = ['AUDIO_UI', {}]
    }
  },
  getCustomWidgets() {
    return {
      AUDIO_UI(node: LGraphNode, inputName: string) {
        const audio = document.createElement('audio')
        audio.controls = true
        audio.classList.add('comfy-audio')
        audio.setAttribute('name', 'media')

        const audioUIWidget: DOMWidget<HTMLAudioElement, string> =
          node.addDOMWidget(inputName, /* name=*/ 'audioUI', audio)
        audioUIWidget.serialize = false
        const { nodeData } = node.constructor
        if (nodeData == null) throw new TypeError('nodeData is null')

        const isOutputNode = nodeData.output_node
        if (isOutputNode) {
          // Hide the audio widget when there is no audio initially.
          audioUIWidget.element.classList.add('empty-audio-widget')
          // Populate the audio widget UI on node execution.
          const onExecuted = node.onExecuted
          node.onExecuted = function (output: NodeExecutionOutput) {
            onExecuted?.call(this, output)
            const audios = output.audio
            if (!audios?.length) return
            const audio = audios[0]
            const resourceUrl = getResourceURL(
              audio.subfolder ?? '',
              audio.filename ?? '',
              audio.type
            )
            updateUIWidget(audioUIWidget, api.apiURL(resourceUrl))
          }
        }

        audioUIWidget.options.getValue = () =>
          (useWidgetValueStore().getWidget(
            resolveNodeRootGraphId(node, app.rootGraph.id),
            node.id,
            inputName
          )?.value as string) ?? ''
        audioUIWidget.options.setValue = (v) => {
          const graphId = resolveNodeRootGraphId(node, app.rootGraph.id)
          const widgetState = useWidgetValueStore().getWidget(
            graphId,
            node.id,
            inputName
          )
          if (widgetState) widgetState.value = v
        }

        return { widget: audioUIWidget }
      }
    }
  },
  onNodeOutputsUpdated(
    nodeOutputs: Record<NodeLocatorId, NodeExecutionOutput>
  ) {
    for (const [nodeLocatorId, output] of Object.entries(nodeOutputs)) {
      if (!output.audio?.length) continue

      const node = getNodeByLocatorId(app.rootGraph, nodeLocatorId)
      if (!node) continue

      const audioUIWidget = node.widgets?.find(
        (w) => w.name === 'audioUI'
      ) as unknown as DOMWidget<HTMLAudioElement, string>
      const audio = output.audio[0]
      const resourceUrl = getResourceURL(
        audio.subfolder ?? '',
        audio.filename ?? '',
        audio.type
      )
      updateUIWidget(audioUIWidget, api.apiURL(resourceUrl))
    }
  }
})

app.registerExtension({
  name: 'Comfy.UploadAudio',
  async beforeRegisterNodeDef(
    _nodeType: typeof LGraphNode,
    nodeData: ComfyNodeDef
  ) {
    if (nodeData?.input?.required?.audio?.[1]?.audio_upload === true) {
      nodeData.input.required.upload = ['AUDIOUPLOAD', {}]
    }
  },
  getCustomWidgets() {
    return {
      AUDIOUPLOAD(node, inputName: string) {
        // The widget that allows user to select file.
        // @ts-expect-error fixme ts strict error
        const audioWidget = node.widgets.find(
          (w) => w.name === 'audio'
        ) as IStringWidget
        // @ts-expect-error fixme ts strict error
        const audioUIWidget = node.widgets.find(
          (w) => w.name === 'audioUI'
        ) as unknown as DOMWidget<HTMLAudioElement, string>

        const onAudioWidgetUpdate = () => {
          updateUIWidget(
            audioUIWidget,
            api.apiURL(
              getResourceURL(...splitFilePath(audioWidget.value ?? ''))
            )
          )
        }
        // Initially load default audio file to audioUIWidget.
        onAudioWidgetUpdate()

        audioWidget.callback = onAudioWidgetUpdate

        // Load saved audio file widget values if restoring from workflow
        const onGraphConfigured = node.onGraphConfigured
        node.onGraphConfigured = function () {
          // @ts-expect-error fixme ts strict error
          onGraphConfigured?.apply(this, arguments)
          onAudioWidgetUpdate()
        }

        const handleUpload = async (files: File[]) => {
          if (files?.length) {
            const previousValue = audioWidget.value
            audioWidget.value = files[0].name
            const success = await uploadFile(
              audioWidget,
              audioUIWidget,
              files[0],
              true
            )
            if (!success) {
              audioWidget.value = previousValue
            }
          }
          return files
        }

        const isAudioFile = (file: File) => file.type.startsWith('audio/')

        const { openFileSelection } = useNodeFileInput(node, {
          accept: 'audio/*',
          onSelect: handleUpload
        })

        // The widget to pop up the upload dialog.
        const uploadWidget = node.addWidget(
          'button',
          inputName,
          '',
          openFileSelection,
          { serialize: false, canvasOnly: true }
        )
        uploadWidget.label = t('g.choose_file_to_upload')

        useNodeDragAndDrop(node, {
          fileFilter: isAudioFile,
          onDrop: handleUpload
        })

        useNodePaste(node, {
          fileFilter: isAudioFile,
          onPaste: handleUpload
        })

        node.previewMediaType = 'audio'

        return { widget: uploadWidget }
      }
    }
  }
})

app.registerExtension({
  name: 'Comfy.RecordAudio',

  getCustomWidgets() {
    return {
      AUDIO_RECORD(node, inputName: string) {
        const audio = document.createElement('audio')
        audio.controls = true
        audio.classList.add('comfy-audio')
        audio.setAttribute('name', 'media')
        const audioUIWidget: DOMWidget<HTMLAudioElement, string> =
          node.addDOMWidget(inputName, /* name=*/ 'audioUI', audio)
        audioUIWidget.options.canvasOnly = false

        let mediaRecorder: MediaRecorder | null = null
        let isRecording = false
        let audioChunks: Blob[] = []
        let currentStream: MediaStream | null = null
        let recordWidget: IBaseWidget | null = null

        let stopPromise: Promise<void> | null = null
        let stopResolve: (() => void) | null = null

        audioUIWidget.serializeValue = async () => {
          if (isRecording && mediaRecorder) {
            stopPromise = new Promise((resolve) => {
              stopResolve = resolve
            })

            mediaRecorder.stop()

            await stopPromise
          }

          const audioSrc = audioUIWidget.element.src

          if (!audioSrc) {
            useToastStore().addAlert(t('g.noAudioRecorded'))
            return ''
          }

          const blob = await fetch(audioSrc).then((r) => r.blob())

          return await useAudioService().convertBlobToFileAndSubmit(blob)
        }

        recordWidget = node.addWidget(
          'button',
          inputName,
          '',
          async () => {
            if (!isRecording) {
              try {
                currentStream = await navigator.mediaDevices.getUserMedia({
                  audio: true
                })

                mediaRecorder = new ExtendableMediaRecorder(currentStream, {
                  mimeType: 'audio/wav'
                }) as unknown as MediaRecorder

                audioChunks = []

                mediaRecorder.ondataavailable = (event) => {
                  audioChunks.push(event.data)
                }

                mediaRecorder.onstop = async () => {
                  const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })

                  useAudioService().stopAllTracks(currentStream)

                  if (
                    audioUIWidget.element.src &&
                    audioUIWidget.element.src.startsWith('blob:')
                  ) {
                    URL.revokeObjectURL(audioUIWidget.element.src)
                  }

                  updateUIWidget(audioUIWidget, URL.createObjectURL(audioBlob))

                  isRecording = false

                  if (recordWidget) {
                    recordWidget.label = t('g.startRecording')
                  }

                  if (stopResolve) {
                    stopResolve()
                    stopResolve = null
                    stopPromise = null
                  }
                }

                mediaRecorder.onerror = (event) => {
                  console.error('MediaRecorder error:', event)
                  useAudioService().stopAllTracks(currentStream)
                  isRecording = false

                  if (recordWidget) {
                    recordWidget.label = t('g.startRecording')
                  }

                  if (stopResolve) {
                    stopResolve()
                    stopResolve = null
                    stopPromise = null
                  }
                }

                mediaRecorder.start()
                isRecording = true
                if (recordWidget) {
                  recordWidget.label = t('g.stopRecording')
                }
              } catch (err) {
                console.error('Error accessing microphone:', err)
                useToastStore().addAlert(t('g.micPermissionDenied'))

                if (mediaRecorder) {
                  try {
                    mediaRecorder.stop()
                  } catch {}
                }
                useAudioService().stopAllTracks(currentStream)
                currentStream = null
                isRecording = false
                if (recordWidget) {
                  recordWidget.label = t('g.startRecording')
                }
              }
            } else if (mediaRecorder && isRecording) {
              mediaRecorder.stop()
            }
          },
          { serialize: false, canvasOnly: false }
        )

        recordWidget.label = t('g.startRecording')
        // Override the type for Vue rendering while keeping 'button' for LiteGraph
        recordWidget.type = 'audiorecord'

        const originalOnRemoved = node.onRemoved
        node.onRemoved = function () {
          if (isRecording && mediaRecorder) {
            mediaRecorder.stop()
          }
          useAudioService().stopAllTracks(currentStream)
          if (audioUIWidget.element.src?.startsWith('blob:')) {
            URL.revokeObjectURL(audioUIWidget.element.src)
          }
          originalOnRemoved?.call(this)
        }

        return { widget: recordWidget }
      }
    }
  },

  async nodeCreated(node: LGraphNode) {
    if (node.constructor.comfyClass !== 'RecordAudio') return

    await useAudioService().registerWavEncoder()
  }
})
