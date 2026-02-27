import { register } from 'extendable-media-recorder'
import { connect } from 'extendable-media-recorder-wav-encoder'

import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'

export interface AudioRecordingError {
  type: 'permission' | 'not_supported' | 'encoder' | 'recording' | 'unknown'
  message: string
  originalError?: unknown
}

let isEncoderRegistered: boolean = false

export const useAudioService = () => {
  const handleError = (
    type: AudioRecordingError['type'],
    message: string,
    originalError?: unknown
  ) => {
    console.error(`Audio Service Error (${type}):`, message, originalError)
  }

  const stopAllTracks = (currentStream: MediaStream | null) => {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => {
        track.stop()
      })
      currentStream = null
    }
  }

  const registerWavEncoder = async (): Promise<void> => {
    if (isEncoderRegistered) {
      return
    }

    try {
      await register(await connect())
      isEncoderRegistered = true
    } catch (err) {
      if (
        err instanceof Error &&
        err.message.includes('already an encoder stored')
      ) {
        isEncoderRegistered = true
      } else {
        handleError('encoder', 'Failed to register WAV encoder', err)
      }
    }
  }

  const convertBlobToFileAndSubmit = async (blob: Blob): Promise<string> => {
    const name = `recording-${Date.now()}.wav`
    const file = new File([blob], name, { type: blob.type || 'audio/wav' })

    const body = new FormData()
    body.append('image', file)
    body.append('subfolder', 'audio')
    body.append('type', 'temp')

    const resp = await api.fetchApi('/upload/image', {
      method: 'POST',
      body
    })

    if (resp.status !== 200) {
      const err = `Error uploading temp file: ${resp.status} - ${resp.statusText}`
      useToastStore().addAlert(err)
      throw new Error(err)
    }

    const tempAudio = await resp.json()

    return `audio/${tempAudio.name} [temp]`
  }

  return {
    // Methods
    convertBlobToFileAndSubmit,
    registerWavEncoder,
    stopAllTracks
  }
}
