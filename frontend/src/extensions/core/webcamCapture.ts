import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useToastStore } from '@/platform/updates/common/toastStore'

import { api } from '../../scripts/api'
import { app } from '../../scripts/app'

const WEBCAM_READY = Symbol()

app.registerExtension({
  name: 'Comfy.WebcamCapture',
  getCustomWidgets() {
    return {
      WEBCAM(node, inputName) {
        // @ts-expect-error fixme ts strict error
        let res
        // @ts-expect-error fixme ts strict error
        node[WEBCAM_READY] = new Promise((resolve) => (res = resolve))

        const container = document.createElement('div')
        container.style.background = 'rgba(0,0,0,0.25)'
        container.style.textAlign = 'center'

        const video = document.createElement('video')
        video.style.height = video.style.width = '100%'

        const loadVideo = async () => {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false
            })
            container.replaceChildren(video)

            // @ts-expect-error fixme ts strict error
            setTimeout(() => res(video), 500) // Fallback as loadedmetadata doesnt fire sometimes?
            // @ts-expect-error fixme ts strict error
            video.addEventListener('loadedmetadata', () => res(video), false)
            video.srcObject = stream
            video.play()
          } catch (error) {
            const label = document.createElement('div')
            label.style.color = 'red'
            label.style.overflow = 'auto'
            label.style.maxHeight = '100%'
            label.style.whiteSpace = 'pre-wrap'

            if (window.isSecureContext) {
              label.textContent =
                'Unable to load webcam, please ensure access is granted:\n' +
                // @ts-expect-error fixme ts strict error
                error.message
            } else {
              label.textContent =
                'Unable to load webcam. A secure context is required, if you are not accessing ComfyUI on localhost (127.0.0.1) you will have to enable TLS (https)\n\n' +
                // @ts-expect-error fixme ts strict error
                error.message
            }

            container.replaceChildren(label)
          }
        }

        loadVideo()

        return { widget: node.addDOMWidget(inputName, 'WEBCAM', container) }
      }
    }
  },
  nodeCreated(node: LGraphNode) {
    if ((node.type, node.constructor.comfyClass !== 'WebcamCapture')) return

    // @ts-expect-error fixme ts strict error
    let video
    // @ts-expect-error fixme ts strict error
    const camera = node.widgets.find((w) => w.name === 'image')
    // @ts-expect-error fixme ts strict error
    const w = node.widgets.find((w) => w.name === 'width')
    // @ts-expect-error fixme ts strict error
    const h = node.widgets.find((w) => w.name === 'height')
    // @ts-expect-error fixme ts strict error
    const captureOnQueue = node.widgets.find(
      (w) => w.name === 'capture_on_queue'
    )

    const canvas = document.createElement('canvas')

    const capture = () => {
      // @ts-expect-error widget value type narrow down
      canvas.width = w.value
      // @ts-expect-error widget value type narrow down
      canvas.height = h.value
      const ctx = canvas.getContext('2d')
      // @ts-expect-error widget value type narrow down
      ctx.drawImage(video, 0, 0, w.value, h.value)
      const data = canvas.toDataURL('image/png')

      const img = new Image()
      img.onload = () => {
        node.imgs = [img]
        app.canvas.setDirty(true)
      }
      img.src = data
    }

    const btn = node.addWidget(
      'button',
      'waiting for camera...',
      'capture',
      capture,
      { canvasOnly: true }
    )
    btn.disabled = true
    btn.serializeValue = () => undefined

    // @ts-expect-error fixme ts strict error
    camera.serializeValue = async () => {
      // @ts-expect-error fixme ts strict error
      if (captureOnQueue.value) {
        capture()
      } else if (!node.imgs?.length) {
        const err = `No webcam image captured`
        useToastStore().addAlert(err)
        throw new Error(err)
      }

      // Upload image to temp storage
      // @ts-expect-error fixme ts strict error
      const blob = await new Promise<Blob>((r) => canvas.toBlob(r))
      const name = `${+new Date()}.png`
      const file = new File([blob], name)
      const body = new FormData()
      body.append('image', file)
      body.append('subfolder', 'webcam')
      body.append('type', 'temp')
      const resp = await api.fetchApi('/upload/image', {
        method: 'POST',
        body
      })
      if (resp.status !== 200) {
        const err = `Error uploading camera image: ${resp.status} - ${resp.statusText}`
        useToastStore().addAlert(err)
        throw new Error(err)
      }
      return `webcam/${name} [temp]`
    }

    // @ts-expect-error fixme ts strict error
    node[WEBCAM_READY].then((v) => {
      video = v
      // If width isn't specified then use video output resolution
      // @ts-expect-error fixme ts strict error
      if (!w.value) {
        // @ts-expect-error fixme ts strict error
        w.value = video.videoWidth || 640
        // @ts-expect-error fixme ts strict error
        h.value = video.videoHeight || 480
      }
      btn.disabled = false
      btn.label = t('g.capture')
    })
  }
})
