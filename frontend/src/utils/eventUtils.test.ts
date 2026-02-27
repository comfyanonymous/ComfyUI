import { describe, expect, it } from 'vitest'
import {
  hasAudioType,
  hasImageType,
  hasVideoType,
  isMediaFile
} from './eventUtils'

describe('hasImageType', () => {
  it('should return true for image types', () => {
    expect(hasImageType({ type: 'image/png' } as File)).toBe(true)
    expect(hasImageType({ type: 'image/jpeg' } as File)).toBe(true)
  })

  it('should return false for non-image types', () => {
    expect(hasImageType({ type: 'audio/mpeg' } as File)).toBe(false)
    expect(hasImageType({ type: 'video/mp4' } as File)).toBe(false)
  })
})

describe('hasAudioType', () => {
  it('should return true for audio types', () => {
    expect(hasAudioType({ type: 'audio/mpeg' } as File)).toBe(true)
    expect(hasAudioType({ type: 'audio/wav' } as File)).toBe(true)
  })

  it('should return false for non-audio types', () => {
    expect(hasAudioType({ type: 'image/png' } as File)).toBe(false)
    expect(hasAudioType({ type: 'video/mp4' } as File)).toBe(false)
  })
})

describe('hasVideoType', () => {
  it('should return true for video types', () => {
    expect(hasVideoType({ type: 'video/mp4' } as File)).toBe(true)
    expect(hasVideoType({ type: 'video/webm' } as File)).toBe(true)
  })

  it('should return false for non-video types', () => {
    expect(hasVideoType({ type: 'audio/mpeg' } as File)).toBe(false)
    expect(hasVideoType({ type: 'image/png' } as File)).toBe(false)
  })
})

describe('isMediaFile', () => {
  it('should return true for image, audio, and video types', () => {
    expect(isMediaFile({ type: 'image/png' } as File)).toBe(true)
    expect(isMediaFile({ type: 'audio/mpeg' } as File)).toBe(true)
    expect(isMediaFile({ type: 'video/mp4' } as File)).toBe(true)
  })

  it('should return false for non-media types', () => {
    expect(isMediaFile({ type: 'text/plain' } as File)).toBe(false)
    expect(isMediaFile({ type: 'application/json' } as File)).toBe(false)
  })
})
