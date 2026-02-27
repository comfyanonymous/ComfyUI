import { describe, expect, it } from 'vitest'

import { normalizePackId, normalizePackKeys } from '@/utils/packUtils'

describe('packUtils', () => {
  describe('normalizePackId', () => {
    it('should return pack ID unchanged when no version suffix exists', () => {
      expect(normalizePackId('ComfyUI-GGUF')).toBe('ComfyUI-GGUF')
      expect(normalizePackId('ComfyUI-Manager')).toBe('ComfyUI-Manager')
      expect(normalizePackId('simple-pack')).toBe('simple-pack')
    })

    it('should remove version suffix with underscores', () => {
      expect(normalizePackId('ComfyUI-GGUF@1_1_4')).toBe('ComfyUI-GGUF')
      expect(normalizePackId('ComfyUI-Manager@2_0_0')).toBe('ComfyUI-Manager')
      expect(normalizePackId('pack@1_0_0_beta')).toBe('pack')
    })

    it('should remove version suffix with dots', () => {
      expect(normalizePackId('ComfyUI-GGUF@1.1.4')).toBe('ComfyUI-GGUF')
      expect(normalizePackId('pack@2.0.0')).toBe('pack')
    })

    it('should handle multiple @ symbols by only removing after first @', () => {
      expect(normalizePackId('pack@1_0_0@extra')).toBe('pack')
      expect(normalizePackId('my@pack@1_0_0')).toBe('my')
    })

    it('should handle empty string', () => {
      expect(normalizePackId('')).toBe('')
    })

    it('should handle pack ID with @ but no version', () => {
      expect(normalizePackId('pack@')).toBe('pack')
    })

    it('should handle special characters in pack name', () => {
      expect(normalizePackId('my-pack_v2@1_0_0')).toBe('my-pack_v2')
      expect(normalizePackId('pack.with.dots@2_0_0')).toBe('pack.with.dots')
      expect(normalizePackId('UPPERCASE-Pack@1_0_0')).toBe('UPPERCASE-Pack')
    })

    it('should handle edge cases', () => {
      // Only @ symbol
      expect(normalizePackId('@')).toBe('')
      expect(normalizePackId('@1_0_0')).toBe('')

      // Whitespace
      expect(normalizePackId(' pack @1_0_0')).toBe(' pack ')
      expect(normalizePackId('pack @1_0_0')).toBe('pack ')
    })
  })

  describe('normalizePackKeys', () => {
    it('should normalize all keys with version suffixes', () => {
      const input = {
        'ComfyUI-GGUF': { ver: '1.1.4', enabled: true },
        'ComfyUI-Manager@2_0_0': { ver: '2.0.0', enabled: false },
        'another-pack@1_0_0': { ver: '1.0.0', enabled: true }
      }

      const expected = {
        'ComfyUI-GGUF': { ver: '1.1.4', enabled: true },
        'ComfyUI-Manager': { ver: '2.0.0', enabled: false },
        'another-pack': { ver: '1.0.0', enabled: true }
      }

      expect(normalizePackKeys(input)).toEqual(expected)
    })

    it('should handle empty object', () => {
      expect(normalizePackKeys({})).toEqual({})
    })

    it('should handle keys without version suffixes', () => {
      const input = {
        pack1: { data: 'value1' },
        pack2: { data: 'value2' }
      }

      expect(normalizePackKeys(input)).toEqual(input)
    })

    it('should handle mixed keys (with and without versions)', () => {
      const input = {
        'normal-pack': { ver: '1.0.0' },
        'versioned-pack@2_0_0': { ver: '2.0.0' },
        'another-normal': { ver: '3.0.0' },
        'another-versioned@4_0_0': { ver: '4.0.0' }
      }

      const expected = {
        'normal-pack': { ver: '1.0.0' },
        'versioned-pack': { ver: '2.0.0' },
        'another-normal': { ver: '3.0.0' },
        'another-versioned': { ver: '4.0.0' }
      }

      expect(normalizePackKeys(input)).toEqual(expected)
    })

    it('should handle duplicate keys after normalization (last one wins)', () => {
      const input = {
        'pack@1_0_0': { ver: '1.0.0', data: 'first' },
        'pack@2_0_0': { ver: '2.0.0', data: 'second' },
        pack: { ver: '3.0.0', data: 'third' }
      }

      const result = normalizePackKeys(input)

      // The exact behavior depends on object iteration order,
      // but there should only be one 'pack' key in the result
      expect(Object.keys(result)).toEqual(['pack'])
      expect(result.pack).toBeDefined()
      expect(result.pack.ver).toBeDefined()
    })

    it('should preserve value references', () => {
      const value1 = { ver: '1.0.0', complex: { nested: 'data' } }
      const value2 = { ver: '2.0.0', complex: { nested: 'data2' } }

      const input = {
        'pack1@1_0_0': value1,
        'pack2@2_0_0': value2
      }

      const result = normalizePackKeys(input)

      // Values should be the same references, not cloned
      expect(result.pack1).toBe(value1)
      expect(result.pack2).toBe(value2)
    })

    it('should handle special characters in keys', () => {
      const input = {
        '@1_0_0': { ver: '1.0.0' },
        'my-pack.v2@2_0_0': { ver: '2.0.0' },
        'UPPERCASE@3_0_0': { ver: '3.0.0' }
      }

      const expected = {
        '': { ver: '1.0.0' },
        'my-pack.v2': { ver: '2.0.0' },
        UPPERCASE: { ver: '3.0.0' }
      }

      expect(normalizePackKeys(input)).toEqual(expected)
    })

    it('should work with different value types', () => {
      const input = {
        'pack1@1_0_0': 'string value',
        'pack2@2_0_0': 123,
        'pack3@3_0_0': null,
        'pack4@4_0_0': undefined,
        'pack5@5_0_0': true,
        pack6: []
      }

      const expected = {
        pack1: 'string value',
        pack2: 123,
        pack3: null,
        pack4: undefined,
        pack5: true,
        pack6: []
      }

      expect(normalizePackKeys(input)).toEqual(expected)
    })
  })

  describe('Integration scenarios from JSDoc examples', () => {
    it('should handle the examples from normalizePackId JSDoc', () => {
      expect(normalizePackId('ComfyUI-GGUF')).toBe('ComfyUI-GGUF')
      expect(normalizePackId('ComfyUI-GGUF@1_1_4')).toBe('ComfyUI-GGUF')
    })

    it('should handle the examples from normalizePackKeys JSDoc', () => {
      const input = {
        'ComfyUI-GGUF': { ver: '1.1.4', enabled: true },
        'ComfyUI-Manager@2_0_0': { ver: '2.0.0', enabled: false }
      }

      const expected = {
        'ComfyUI-GGUF': { ver: '1.1.4', enabled: true },
        'ComfyUI-Manager': { ver: '2.0.0', enabled: false }
      }

      expect(normalizePackKeys(input)).toEqual(expected)
    })
  })

  describe('Real-world scenarios', () => {
    it('should handle typical ComfyUI-Manager response with mixed enabled/disabled packs', () => {
      // Simulating actual server response pattern
      const serverResponse = {
        // Enabled packs come without version suffix
        'ComfyUI-Essential': { ver: '1.2.3', enabled: true, aux_id: undefined },
        'ComfyUI-Impact': { ver: '2.0.0', enabled: true, aux_id: undefined },
        // Disabled packs come with version suffix
        'ComfyUI-GGUF@1_1_4': {
          ver: '1.1.4',
          enabled: false,
          aux_id: undefined
        },
        'ComfyUI-Manager@2_5_0': {
          ver: '2.5.0',
          enabled: false,
          aux_id: undefined
        }
      }

      const normalized = normalizePackKeys(serverResponse)

      // All keys should be normalized (no version suffixes)
      expect(Object.keys(normalized)).toEqual([
        'ComfyUI-Essential',
        'ComfyUI-Impact',
        'ComfyUI-GGUF',
        'ComfyUI-Manager'
      ])

      // Values should be preserved
      expect(normalized['ComfyUI-GGUF']).toEqual({
        ver: '1.1.4',
        enabled: false,
        aux_id: undefined
      })
    })

    it('should allow consistent access by pack ID regardless of enabled state', () => {
      const packsBeforeToggle = {
        'my-pack': { ver: '1.0.0', enabled: true }
      }

      const packsAfterToggle = {
        'my-pack@1_0_0': { ver: '1.0.0', enabled: false }
      }

      const normalizedBefore = normalizePackKeys(packsBeforeToggle)
      const normalizedAfter = normalizePackKeys(packsAfterToggle)

      // Both should have the same key after normalization
      expect(normalizedBefore['my-pack']).toBeDefined()
      expect(normalizedAfter['my-pack']).toBeDefined()

      // Can access by the same key regardless of the original format
      expect(Object.keys(normalizedBefore)).toEqual(
        Object.keys(normalizedAfter)
      )
    })
  })
})
