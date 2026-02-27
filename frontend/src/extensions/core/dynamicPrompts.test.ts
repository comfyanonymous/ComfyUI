import { afterEach, describe, expect, it, test, vi } from 'vitest'

import { processDynamicPrompt } from '@/utils/formatUtil'

describe('dynamic prompts', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('handles single and multiline comments', () => {
    const input =
      '/*\nStart\n*/Hello /* this is a comment */ world!\n// it\nEnd'
    expect(processDynamicPrompt(input)).toBe('Hello  world!\n\nEnd')
  })

  it('handles simple option groups', () => {
    const input = '{option1|option2}'

    vi.spyOn(Math, 'random').mockReturnValue(0)
    expect(processDynamicPrompt(input)).toBe('option1')

    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('option2')
  })

  test('handles trailing empty options', () => {
    const input = '{a|}'
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles leading empty options', () => {
    const input = '{|a}'
    vi.spyOn(Math, 'random').mockReturnValue(0)
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles multiple empty alternatives', () => {
    const input = '{||}'
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles multiple nested empty alternatives', () => {
    const input = '{a|{b||c}|}'
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles unescaped special characters gracefully', () => {
    const input = '{a|\\}'
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('}')
  })

  it('handles nested option groups', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0) // pick the first option at each level
    const input = '{a|{b|{c|d}}}'
    expect(processDynamicPrompt(input)).toBe('a')

    vi.spyOn(Math, 'random').mockReturnValue(0.99) // pick the last option at each level
    expect(processDynamicPrompt(input)).toBe('d')
  })

  test('handles escaped braces', () => {
    const input = '\\{a|b\\}'
    expect(processDynamicPrompt(input)).toBe('{a|b}')
  })

  test('handles escaped pipe', () => {
    const input = 'a\\|b'
    expect(processDynamicPrompt(input)).toBe('a|b')
  })

  it('handles escaped characters', () => {
    const input = '{\\{escaped\\}\\|escaped pipe}'
    expect(processDynamicPrompt(input)).toBe('{escaped}|escaped pipe')
  })

  test('handles deeply nested escaped characters', () => {
    const input = '{a|{b|\\{c\\}}}'
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('{c}')
  })

  it('handles mixed input', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    const input =
      '<{option1|option2}>/*comment*/ ({something|else}:2) \\{escaped\\}!'
    expect(processDynamicPrompt(input)).toBe('<option2> (else:2) {escaped}!')
  })

  it('handles non-paired braces gracefully', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const input = '{option1|option2|{nested1|nested2'
    expect(processDynamicPrompt(input)).toBe('option1')

    vi.spyOn(Math, 'random').mockReturnValue(0.4)
    expect(processDynamicPrompt(input)).toBe('option2')

    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('nested2')
  })

  it('handles deep nesting', () => {
    const input = '{a|{b|{c|{d|{e|{f|{g}}1}2}3}4}5}'
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    expect(processDynamicPrompt(input)).toBe('g12345')
  })

  test('handles empty alternative inside braces', () => {
    const input = '{|a||b|}'
    vi.spyOn(Math, 'random').mockReturnValue(0)
    expect(processDynamicPrompt(input)).toBe('')
    vi.spyOn(Math, 'random').mockReturnValue(0.3)
    expect(processDynamicPrompt(input)).toBe('a')
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    expect(processDynamicPrompt(input)).toBe('')
    vi.spyOn(Math, 'random').mockReturnValue(0.75)
    expect(processDynamicPrompt(input)).toBe('b')
    vi.spyOn(Math, 'random').mockReturnValue(0.999)
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles no braces', () => {
    const input = 'abcdef'
    expect(processDynamicPrompt(input)).toBe('abcdef')
  })

  test('handles empty input', () => {
    const input = ''
    expect(processDynamicPrompt(input)).toBe('')
  })

  test('handles complex mixed cases', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5) //pick the second option from each group
    const input = '1{a|b|{c|d}}2{e|f}3'
    expect(processDynamicPrompt(input)).toBe('1b2f3')
  })
})
