import { describe, expect, it } from 'vitest'

import type {
  ComboInputSpec,
  ComboInputSpecV2,
  FloatInputSpec,
  InputSpec,
  IntInputSpec
} from '@/schemas/nodeDefSchema'
import { mergeInputSpec } from '@/utils/nodeDefUtil'

describe('nodeDefUtil', () => {
  describe('mergeInputSpec', () => {
    // Test numeric input specs (INT and FLOAT)
    describe('numeric input specs', () => {
      it('should merge INT specs with overlapping ranges', () => {
        const spec1: IntInputSpec = ['INT', { min: 0, max: 10 }]
        const spec2: IntInputSpec = ['INT', { min: 5, max: 15 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('INT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].min).toBe(5)
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].max).toBe(10)
      })

      it('should return null for INT specs with non-overlapping ranges', () => {
        const spec1: IntInputSpec = ['INT', { min: 0, max: 5 }]
        const spec2: IntInputSpec = ['INT', { min: 10, max: 15 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).toBeNull()
      })

      it('should merge FLOAT specs with overlapping ranges', () => {
        const spec1: FloatInputSpec = ['FLOAT', { min: 0.5, max: 10.5 }]
        const spec2: FloatInputSpec = ['FLOAT', { min: 5.5, max: 15.5 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('FLOAT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].min).toBe(5.5)
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].max).toBe(10.5)
      })

      it('should handle specs with undefined min/max values', () => {
        const spec1: FloatInputSpec = ['FLOAT', { min: 0.5 }]
        const spec2: FloatInputSpec = ['FLOAT', { max: 15.5 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('FLOAT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].min).toBe(0.5)
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].max).toBe(15.5)
      })

      it('should merge step values using least common multiple', () => {
        const spec1: IntInputSpec = ['INT', { min: 0, max: 10, step: 2 }]
        const spec2: IntInputSpec = ['INT', { min: 0, max: 10, step: 3 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('INT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].step).toBe(6) // LCM of 2 and 3 is 6
      })

      it('should use default step of 1 when step is not specified', () => {
        const spec1: IntInputSpec = ['INT', { min: 0, max: 10 }]
        const spec2: IntInputSpec = ['INT', { min: 0, max: 10, step: 4 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('INT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].step).toBe(4) // LCM of 1 and 4 is 4
      })

      it('should handle step values for FLOAT specs', () => {
        const spec1: FloatInputSpec = ['FLOAT', { min: 0, max: 10, step: 0.5 }]
        const spec2: FloatInputSpec = ['FLOAT', { min: 0, max: 10, step: 0.25 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('FLOAT')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].step).toBe(0.5)
      })
    })

    // Test combo input specs
    describe('combo input specs', () => {
      it('should merge COMBO specs with overlapping options', () => {
        const spec1: ComboInputSpecV2 = ['COMBO', { options: ['A', 'B', 'C'] }]
        const spec2: ComboInputSpecV2 = ['COMBO', { options: ['B', 'C', 'D'] }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('COMBO')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].options).toEqual(['B', 'C'])
      })

      it('should return null for COMBO specs with no overlapping options', () => {
        const spec1: ComboInputSpecV2 = ['COMBO', { options: ['A', 'B'] }]
        const spec2: ComboInputSpecV2 = ['COMBO', { options: ['C', 'D'] }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).toBeNull()
      })

      it('should handle COMBO specs with additional properties', () => {
        const spec1: ComboInputSpecV2 = [
          'COMBO',
          {
            options: ['A', 'B', 'C'],
            default: 'A',
            tooltip: 'Select an option'
          }
        ]
        const spec2: ComboInputSpecV2 = [
          'COMBO',
          {
            options: ['B', 'C', 'D'],
            default: 'B',
            multiline: true
          }
        ]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('COMBO')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].options).toEqual(['B', 'C'])
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].default).toBe('B')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].tooltip).toBe('Select an option')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].multiline).toBe(true)
      })

      it('should handle v1 and v2 combo specs', () => {
        const spec1: ComboInputSpec = [['A', 'B', 'C', 'D'], {}]
        const spec2: ComboInputSpecV2 = ['COMBO', { options: ['C', 'D'] }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('COMBO')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].options).toEqual(['C', 'D'])
      })
    })

    // Test common input spec behavior
    describe('common input spec behavior', () => {
      it('should return null for specs with different types', () => {
        const spec1: IntInputSpec = ['INT', { min: 0, max: 10 }]
        const spec2: ComboInputSpecV2 = ['COMBO', { options: ['A', 'B'] }]

        const result = mergeInputSpec(spec1, spec2 as unknown as IntInputSpec)

        expect(result).toBeNull()
      })

      it('should ignore specified keys when comparing specs', () => {
        const spec1: InputSpec = [
          'STRING',
          {
            default: 'value1',
            tooltip: 'Tooltip 1',
            step: 1
          }
        ]
        const spec2: InputSpec = [
          'STRING',
          {
            default: 'value2',
            tooltip: 'Tooltip 2',
            step: 1
          }
        ]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).not.toBeNull()
        expect(result?.[0]).toBe('STRING')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].default).toBe('value2')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].tooltip).toBe('Tooltip 2')
        // @ts-expect-error fixme ts strict error
        expect(result?.[1].step).toBe(1)
      })

      it('should return null if non-ignored properties differ', () => {
        const spec1: InputSpec = ['STRING', { step: 1 }]
        const spec2: InputSpec = ['STRING', { step: 2 }]

        const result = mergeInputSpec(spec1, spec2)

        expect(result).toBeNull()
      })
    })
  })
})
