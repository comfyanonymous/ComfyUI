import { describe, expect, it } from 'vitest'

import {
  generateCategoryId,
  getCategoryIcon,
  getProviderBorderStyle,
  getProviderIcon
} from './categoryUtil'

describe('getCategoryIcon', () => {
  it('returns mapped icon for known category', () => {
    expect(getCategoryIcon('all')).toBe('icon-[lucide--list]')
    expect(getCategoryIcon('image')).toBe('icon-[lucide--image]')
    expect(getCategoryIcon('video')).toBe('icon-[lucide--film]')
  })

  it('returns folder icon for unknown category', () => {
    expect(getCategoryIcon('unknown-category')).toBe('icon-[lucide--folder]')
  })

  it('is case insensitive', () => {
    expect(getCategoryIcon('ALL')).toBe('icon-[lucide--list]')
    expect(getCategoryIcon('Image')).toBe('icon-[lucide--image]')
  })
})

describe('getProviderIcon', () => {
  it('returns icon class for simple provider name', () => {
    expect(getProviderIcon('BFL')).toBe('icon-[comfy--bfl]')
    expect(getProviderIcon('OpenAI')).toBe('icon-[comfy--openai]')
  })

  it('converts spaces to hyphens', () => {
    expect(getProviderIcon('Stability AI')).toBe('icon-[comfy--stability-ai]')
    expect(getProviderIcon('Moonvalley Marey')).toBe(
      'icon-[comfy--moonvalley-marey]'
    )
  })

  it('converts to lowercase', () => {
    expect(getProviderIcon('GEMINI')).toBe('icon-[comfy--gemini]')
  })
})

describe('getProviderBorderStyle', () => {
  it('returns solid color for single-color providers', () => {
    expect(getProviderBorderStyle('BFL')).toBe('#ffffff')
    expect(getProviderBorderStyle('OpenAI')).toBe('#B6B6B6')
    expect(getProviderBorderStyle('Bria')).toBe('#B6B6B6')
  })

  it('returns gradient for dual-color providers', () => {
    expect(getProviderBorderStyle('Gemini')).toBe(
      'linear-gradient(90deg, #3186FF, #FABC12)'
    )
    expect(getProviderBorderStyle('Stability AI')).toBe(
      'linear-gradient(90deg, #9D39FF, #E80000)'
    )
  })

  it('returns fallback color for unknown providers', () => {
    expect(getProviderBorderStyle('Unknown Provider')).toBe('#525252')
  })

  it('handles provider names with spaces', () => {
    expect(getProviderBorderStyle('Stability AI')).toBe(
      'linear-gradient(90deg, #9D39FF, #E80000)'
    )
    expect(getProviderBorderStyle('Moonvalley Marey')).toBe('#DAD9C5')
  })
})

describe('generateCategoryId', () => {
  it('generates category ID from group and title', () => {
    expect(generateCategoryId('Generation', 'Image')).toBe('generation-image')
  })

  it('converts spaces to hyphens', () => {
    expect(generateCategoryId('API Nodes', 'Open Source')).toBe(
      'api-nodes-open-source'
    )
  })

  it('converts to lowercase', () => {
    expect(generateCategoryId('GENERATION', 'VIDEO')).toBe('generation-video')
  })
})
