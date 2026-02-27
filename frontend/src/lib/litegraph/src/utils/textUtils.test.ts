import type { Mock } from 'vitest'
import { describe, expect, it, vi } from 'vitest'

import { truncateText } from '@/lib/litegraph/src/litegraph'

describe('truncateText', () => {
  const createMockContext = (charWidth: number = 10) => {
    return {
      measureText: vi.fn(
        (text: string) =>
          ({
            width: text.length * charWidth
          }) as TextMetrics
      )
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D
  }

  it('should return original text if it fits within maxWidth', () => {
    const ctx = createMockContext()
    const result = truncateText(ctx, 'Short', 100)
    expect(result).toBe('Short')
  })

  it('should return original text if maxWidth is 0 or negative', () => {
    const ctx = createMockContext()
    expect(truncateText(ctx, 'Text', 0)).toBe('Text')
    expect(truncateText(ctx, 'Text', -10)).toBe('Text')
  })

  it('should truncate text and add ellipsis when text is too long', () => {
    const ctx = createMockContext(10) // 10 pixels per character
    const result = truncateText(ctx, 'This is a very long text', 100)
    // 100px total, "..." takes 30px, leaving 70px for text (7 chars)
    expect(result).toBe('This is...')
  })

  it('should use custom ellipsis when provided', () => {
    const ctx = createMockContext(10)
    const result = truncateText(ctx, 'This is a very long text', 100, '…')
    // 100px total, "…" takes 10px, leaving 90px for text (9 chars)
    expect(result).toBe('This is a…')
  })

  it('should return only ellipsis if available width is too small', () => {
    const ctx = createMockContext(10)
    const result = truncateText(ctx, 'Text', 20) // Only room for 2 chars, but "..." needs 3
    expect(result).toBe('...')
  })

  it('should handle empty text', () => {
    const ctx = createMockContext()
    const result = truncateText(ctx, '', 100)
    expect(result).toBe('')
  })

  it('should use binary search efficiently', () => {
    const ctx = createMockContext(10)
    const longText = 'A'.repeat(100) // 100 characters

    const result = truncateText(ctx, longText, 200) // Room for 20 chars total
    expect(result).toBe(`${'A'.repeat(17)}...`) // 17 chars + "..." = 20 chars = 200px

    // Verify binary search efficiency - should not measure every possible substring
    // Binary search for 100 chars should take around log2(100) ≈ 7 iterations
    // Plus a few extra calls for measuring the full text and ellipsis
    const callCount = (ctx.measureText as Mock).mock.calls.length
    expect(callCount).toBeLessThan(20)
    expect(callCount).toBeGreaterThan(5)
  })

  it('should handle unicode characters correctly', () => {
    const ctx = createMockContext(10)
    const result = truncateText(ctx, 'Hello 👋 World', 80)
    // Assuming each char (including emoji) is 10px, total is 130px
    // 80px total, "..." takes 30px, leaving 50px for text (5 chars)
    expect(result).toBe('Hello...')
  })

  it('should handle exact boundary cases', () => {
    const ctx = createMockContext(10)

    // Text exactly fits
    expect(truncateText(ctx, 'Exact', 50)).toBe('Exact') // 5 chars = 50px

    // Text is exactly 1 pixel too long
    expect(truncateText(ctx, 'Exact!', 50)).toBe('Ex...') // 6 chars = 60px, truncated
  })
})
