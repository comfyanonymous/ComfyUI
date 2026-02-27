import { describe, expect, it } from 'vitest'

import { iconForMediaType } from './mediaIconUtil'

describe('iconForMediaType', () => {
  it('maps text and misc fallbacks correctly', () => {
    expect(iconForMediaType('text')).toBe('icon-[lucide--text]')
    expect(iconForMediaType('other')).toBe('icon-[lucide--check-check]')
  })

  it('preserves existing mappings for core media types', () => {
    expect(iconForMediaType('image')).toBe('icon-[lucide--image]')
    expect(iconForMediaType('video')).toBe('icon-[lucide--video]')
    expect(iconForMediaType('audio')).toBe('icon-[lucide--music]')
    expect(iconForMediaType('3D')).toBe('icon-[lucide--box]')
  })
})
