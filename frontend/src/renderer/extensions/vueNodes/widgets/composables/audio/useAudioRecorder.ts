import { MediaRecorder as ExtendableMediaRecorder } from 'extendable-media-recorder'
import { onUnmounted, ref } from 'vue'

import { useAudioService } from '@/services/audioService'

interface AudioRecorderOptions {
  onRecordingComplete?: (audioBlob: Blob) => Promise<void>
  onError?: (error: Error) => void
}

export function useAudioRecorder(options: AudioRecorderOptions = {}) {
  const isRecording = ref(false)
  const mediaRecorder = ref<MediaRecorder | null>(null)
  const audioChunks = ref<Blob[]>([])
  const stream = ref<MediaStream | null>(null)
  const recordedURL = ref<string | null>(null)

  async function startRecording() {
    try {
      // Clean up previous recording
      if (recordedURL.value?.startsWith('blob:')) {
        URL.revokeObjectURL(recordedURL.value)
      }

      // Initialize
      audioChunks.value = []
      recordedURL.value = null

      // Register wav encoder and get media stream
      await useAudioService().registerWavEncoder()
      stream.value = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Create media recorder
      mediaRecorder.value = new ExtendableMediaRecorder(stream.value, {
        mimeType: 'audio/wav'
      }) as unknown as MediaRecorder

      mediaRecorder.value.ondataavailable = (e) => {
        audioChunks.value.push(e.data)
      }

      mediaRecorder.value.onstop = async () => {
        const blob = new Blob(audioChunks.value, { type: 'audio/wav' })

        // Create blob URL for preview
        if (recordedURL.value?.startsWith('blob:')) {
          URL.revokeObjectURL(recordedURL.value)
        }
        recordedURL.value = URL.createObjectURL(blob)

        // Notify completion
        if (options.onRecordingComplete) {
          await options.onRecordingComplete(blob)
        }

        cleanup()
      }

      // Start recording
      mediaRecorder.value.start(100)
      isRecording.value = true
    } catch (err) {
      if (options.onError) {
        options.onError(err as Error)
      }
      throw err
    }
  }

  function stopRecording() {
    if (mediaRecorder.value && mediaRecorder.value.state !== 'inactive') {
      mediaRecorder.value.stop()
    } else {
      cleanup()
    }
  }

  function cleanup() {
    isRecording.value = false

    if (stream.value) {
      stream.value.getTracks().forEach((track) => track.stop())
      stream.value = null
    }
  }

  function dispose() {
    stopRecording()
    if (recordedURL.value) {
      URL.revokeObjectURL(recordedURL.value)
      recordedURL.value = null
    }
  }

  onUnmounted(() => {
    dispose()
  })

  return {
    isRecording,
    recordedURL,
    mediaRecorder,
    startRecording,
    stopRecording,
    dispose
  }
}
