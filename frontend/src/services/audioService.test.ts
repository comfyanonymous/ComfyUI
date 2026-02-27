import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAudioService } from '@/services/audioService'
import type { AudioRecordingError } from '@/services/audioService'

const mockRegister = vi.hoisted(() => vi.fn())
const mockConnect = vi.hoisted(() => vi.fn())

const mockApi = vi.hoisted(() => ({
  fetchApi: vi.fn()
}))

const mockToastStore = vi.hoisted(() => ({
  addAlert: vi.fn()
}))

vi.mock('extendable-media-recorder', () => ({
  register: mockRegister
}))

vi.mock('extendable-media-recorder-wav-encoder', () => ({
  connect: mockConnect
}))

vi.mock('@/scripts/api', () => ({
  api: mockApi
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: vi.fn(() => mockToastStore)
}))

describe('useAudioService', () => {
  let service: ReturnType<typeof useAudioService>

  const mockBlob = new Blob(['test audio data'], { type: 'audio/wav' })
  const mockUploadResponse = {
    name: 'test-audio-123.wav'
  }

  beforeEach(() => {
    vi.clearAllMocks()

    vi.spyOn(console, 'error').mockImplementation(() => {})

    mockConnect.mockResolvedValue('mock-encoder')
    mockRegister.mockResolvedValue(undefined)
    mockApi.fetchApi.mockResolvedValue({
      status: 200,
      json: () => Promise.resolve(mockUploadResponse)
    })

    service = useAudioService()
  })

  describe('initialization', () => {
    it('should initialize service with required methods', () => {
      expect(service).toHaveProperty('registerWavEncoder')
      expect(service).toHaveProperty('stopAllTracks')
      expect(service).toHaveProperty('convertBlobToFileAndSubmit')
      expect(typeof service.registerWavEncoder).toBe('function')
      expect(typeof service.stopAllTracks).toBe('function')
      expect(typeof service.convertBlobToFileAndSubmit).toBe('function')
    })
  })

  describe('registerWavEncoder', () => {
    it('should register WAV encoder successfully on first call', async () => {
      await service.registerWavEncoder()

      expect(mockConnect).toHaveBeenCalledTimes(1)
      expect(mockRegister).toHaveBeenCalledWith('mock-encoder')
    })

    it('should not register again if already registered', async () => {
      await service.registerWavEncoder()

      mockConnect.mockClear()
      mockRegister.mockClear()

      await service.registerWavEncoder()

      expect(mockConnect).not.toHaveBeenCalled()
      expect(mockRegister).not.toHaveBeenCalled()
    })

    it('should handle "already an encoder stored" error gracefully', async () => {
      const error = new Error(
        'There is already an encoder stored which handles exactly the same mime types.'
      )
      mockRegister.mockRejectedValueOnce(error)

      await service.registerWavEncoder()

      expect(mockConnect).toHaveBeenCalledTimes(0)
      expect(mockRegister).toHaveBeenCalledTimes(0)
      expect(console.error).not.toHaveBeenCalled()
    })
  })

  describe('stopAllTracks', () => {
    it('should stop all tracks in a stream', () => {
      const mockTrack1 = { stop: vi.fn() }
      const mockTrack2 = { stop: vi.fn() }
      const mockStream = {
        getTracks: vi.fn().mockReturnValue([mockTrack1, mockTrack2])
      } as Partial<MediaStream> as MediaStream

      service.stopAllTracks(mockStream)

      expect(mockStream.getTracks).toHaveBeenCalledTimes(1)
      expect(mockTrack1.stop).toHaveBeenCalledTimes(1)
      expect(mockTrack2.stop).toHaveBeenCalledTimes(1)
    })

    it('should handle null stream gracefully', () => {
      expect(() => service.stopAllTracks(null)).not.toThrow()
    })

    it('should handle stream with no tracks', () => {
      const mockStream = {
        getTracks: vi.fn().mockReturnValue([])
      } as Partial<MediaStream> as MediaStream

      expect(() => service.stopAllTracks(mockStream)).not.toThrow()
      expect(mockStream.getTracks).toHaveBeenCalledTimes(1)
    })

    it('should handle tracks that throw on stop', () => {
      const mockTrack1 = { stop: vi.fn() }
      const mockTrack2 = {
        stop: vi.fn().mockImplementation(() => {
          throw new Error('Stop failed')
        })
      }
      const mockStream = {
        getTracks: vi.fn().mockReturnValue([mockTrack1, mockTrack2])
      } as Partial<MediaStream> as MediaStream

      expect(() => service.stopAllTracks(mockStream)).toThrow()
      expect(mockTrack1.stop).toHaveBeenCalledTimes(1)
      expect(mockTrack2.stop).toHaveBeenCalledTimes(1)
    })
  })

  describe('convertBlobToFileAndSubmit', () => {
    it('should convert blob to file and upload successfully', async () => {
      const result = await service.convertBlobToFileAndSubmit(mockBlob)

      expect(mockApi.fetchApi).toHaveBeenCalledWith('/upload/image', {
        method: 'POST',
        body: expect.any(FormData)
      })

      expect(result).toBe('audio/test-audio-123.wav [temp]')
    })

    it('should create file with correct name and type', async () => {
      const mockTimestamp = 1640995200000
      vi.spyOn(Date, 'now').mockReturnValue(mockTimestamp)

      await service.convertBlobToFileAndSubmit(mockBlob)

      const formDataCall = mockApi.fetchApi.mock.calls[0][1].body as FormData
      const uploadedFile = formDataCall.get('image') as File

      expect(uploadedFile).toBeInstanceOf(File)
      expect(uploadedFile.name).toBe(`recording-${mockTimestamp}.wav`)
      expect(uploadedFile.type).toBe('audio/wav')
    })

    it('should set correct form data fields', async () => {
      await service.convertBlobToFileAndSubmit(mockBlob)

      const formDataCall = mockApi.fetchApi.mock.calls[0][1].body as FormData

      expect(formDataCall.get('subfolder')).toBe('audio')
      expect(formDataCall.get('type')).toBe('temp')
      expect(formDataCall.get('image')).toBeInstanceOf(File)
    })

    it('should handle blob with different type', async () => {
      const customBlob = new Blob(['test'], { type: 'audio/ogg' })

      await service.convertBlobToFileAndSubmit(customBlob)

      const formDataCall = mockApi.fetchApi.mock.calls[0][1].body as FormData
      const uploadedFile = formDataCall.get('image') as File

      expect(uploadedFile.type).toBe('audio/ogg')
    })

    it('should handle blob with no type', async () => {
      const customBlob = new Blob(['test'])

      await service.convertBlobToFileAndSubmit(customBlob)

      const formDataCall = mockApi.fetchApi.mock.calls[0][1].body as FormData
      const uploadedFile = formDataCall.get('image') as File

      expect(uploadedFile.type).toBe('audio/wav') // Should default to audio/wav
    })

    it('should handle upload failure with error status', async () => {
      mockApi.fetchApi.mockResolvedValueOnce({
        status: 500,
        statusText: 'Internal Server Error'
      })

      await expect(
        service.convertBlobToFileAndSubmit(mockBlob)
      ).rejects.toThrow(
        'Error uploading temp file: 500 - Internal Server Error'
      )

      expect(mockToastStore.addAlert).toHaveBeenCalledWith(
        'Error uploading temp file: 500 - Internal Server Error'
      )
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network Error')
      mockApi.fetchApi.mockRejectedValueOnce(networkError)

      await expect(
        service.convertBlobToFileAndSubmit(mockBlob)
      ).rejects.toThrow('Network Error')
    })

    it('should handle different status codes', async () => {
      const testCases = [
        { status: 400, statusText: 'Bad Request' },
        { status: 403, statusText: 'Forbidden' },
        { status: 404, statusText: 'Not Found' },
        { status: 413, statusText: 'Payload Too Large' }
      ]

      for (const testCase of testCases) {
        mockApi.fetchApi.mockResolvedValueOnce(testCase)

        await expect(
          service.convertBlobToFileAndSubmit(mockBlob)
        ).rejects.toThrow(
          `Error uploading temp file: ${testCase.status} - ${testCase.statusText}`
        )

        expect(mockToastStore.addAlert).toHaveBeenCalledWith(
          `Error uploading temp file: ${testCase.status} - ${testCase.statusText}`
        )

        mockToastStore.addAlert.mockClear()
      }
    })

    it('should handle malformed response JSON', async () => {
      mockApi.fetchApi.mockResolvedValueOnce({
        status: 200,
        json: () => Promise.reject(new Error('Invalid JSON'))
      })

      await expect(
        service.convertBlobToFileAndSubmit(mockBlob)
      ).rejects.toThrow('Invalid JSON')
    })

    it('should handle empty response', async () => {
      mockApi.fetchApi.mockResolvedValueOnce({
        status: 200,
        json: () => Promise.resolve({})
      })

      const result = await service.convertBlobToFileAndSubmit(mockBlob)

      expect(result).toBe('audio/undefined [temp]')
    })
  })

  describe('error handling', () => {
    it('should handle AudioRecordingError interface correctly', () => {
      const error: AudioRecordingError = {
        type: 'permission',
        message: 'Microphone access denied',
        originalError: new Error('Permission denied')
      }

      expect(error.type).toBe('permission')
      expect(error.message).toBe('Microphone access denied')
      expect(error.originalError).toBeInstanceOf(Error)
    })

    it('should support all error types', () => {
      const errorTypes = [
        'permission',
        'not_supported',
        'encoder',
        'recording',
        'unknown'
      ] as const

      errorTypes.forEach((type) => {
        const error: AudioRecordingError = {
          type,
          message: `Test error for ${type}`
        }

        expect(error.type).toBe(type)
      })
    })
  })

  describe('edge cases', () => {
    it('should handle very large blobs', async () => {
      const largeData = new Array(1000000).fill('a').join('')
      const largeBlob = new Blob([largeData], { type: 'audio/wav' })

      const result = await service.convertBlobToFileAndSubmit(largeBlob)

      expect(result).toBe('audio/test-audio-123.wav [temp]')
      expect(mockApi.fetchApi).toHaveBeenCalledTimes(1)
    })

    it('should handle empty blob', async () => {
      const emptyBlob = new Blob([], { type: 'audio/wav' })

      const result = await service.convertBlobToFileAndSubmit(emptyBlob)

      expect(result).toBe('audio/test-audio-123.wav [temp]')
    })
  })
})
