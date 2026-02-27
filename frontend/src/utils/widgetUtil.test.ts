import { describe, expect, it } from 'vitest'

import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'

import { getWidgetDefaultValue } from '@/utils/widgetUtil'

describe('getWidgetDefaultValue', () => {
  it('returns undefined for undefined spec', () => {
    expect(getWidgetDefaultValue(undefined)).toBeUndefined()
  })

  it('returns explicit default when provided', () => {
    const spec = { type: 'INT', default: 42 } as InputSpec
    expect(getWidgetDefaultValue(spec)).toBe(42)
  })

  it.for([
    { type: 'INT', expected: 0 },
    { type: 'FLOAT', expected: 0 },
    { type: 'BOOLEAN', expected: false },
    { type: 'STRING', expected: '' }
  ])(
    'returns $expected for $type type without default',
    ({ type, expected }) => {
      const spec = { type } as InputSpec
      expect(getWidgetDefaultValue(spec)).toBe(expected)
    }
  )

  it('returns first option for array options without default', () => {
    const spec = { type: 'COMBO', options: ['a', 'b', 'c'] } as InputSpec
    expect(getWidgetDefaultValue(spec)).toBe('a')
  })

  it('returns undefined for unknown type without options', () => {
    const spec = { type: 'CUSTOM' } as InputSpec
    expect(getWidgetDefaultValue(spec)).toBeUndefined()
  })
})
