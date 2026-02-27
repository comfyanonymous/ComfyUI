import { afterEach, describe, expect, it, vi } from 'vitest'

import { getDevOverride } from '@/utils/devFeatureFlagOverride'

describe('getDevOverride', () => {
  afterEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('returns undefined when no override is set', () => {
    expect(getDevOverride('some_flag')).toBeUndefined()
  })

  it('returns parsed boolean true', () => {
    localStorage.setItem('ff:some_flag', 'true')
    expect(getDevOverride<boolean>('some_flag')).toBe(true)
  })

  it('returns parsed boolean false', () => {
    localStorage.setItem('ff:some_flag', 'false')
    expect(getDevOverride<boolean>('some_flag')).toBe(false)
  })

  it('returns parsed number', () => {
    localStorage.setItem('ff:max_upload_size', '209715200')
    expect(getDevOverride<number>('max_upload_size')).toBe(209715200)
  })

  it('returns parsed string', () => {
    localStorage.setItem('ff:some_flag', '"hello"')
    expect(getDevOverride<string>('some_flag')).toBe('hello')
  })

  it('returns parsed object', () => {
    localStorage.setItem('ff:complex', '{"nested": true}')
    expect(getDevOverride<Record<string, boolean>>('complex')).toEqual({
      nested: true
    })
  })

  it('uses ff: prefix for localStorage keys', () => {
    localStorage.setItem('some_flag', 'true')
    expect(getDevOverride('some_flag')).toBeUndefined()

    localStorage.setItem('ff:some_flag', 'true')
    expect(getDevOverride('some_flag')).toBe(true)
  })

  it('returns undefined and warns on invalid JSON', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    localStorage.setItem('ff:bad', 'True')

    expect(getDevOverride('bad')).toBeUndefined()
    expect(warnSpy).toHaveBeenCalledWith(
      '[ff] Invalid JSON for override "bad":',
      'True'
    )
  })
})
