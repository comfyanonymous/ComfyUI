import { describe, expect, it } from 'vitest'

import { fnv1a, hashPath } from './hashUtil'

describe('fnv1a', () => {
  it('returns consistent hash for same input', () => {
    const hash1 = fnv1a('workflows/test.json')
    const hash2 = fnv1a('workflows/test.json')
    expect(hash1).toBe(hash2)
  })

  it('returns different hashes for different inputs', () => {
    const hash1 = fnv1a('workflows/a.json')
    const hash2 = fnv1a('workflows/b.json')
    expect(hash1).not.toBe(hash2)
  })

  it('returns unsigned 32-bit integer', () => {
    const hash = fnv1a('test')
    expect(hash).toBeGreaterThanOrEqual(0)
    expect(hash).toBeLessThanOrEqual(0xffffffff)
  })

  it('handles empty string', () => {
    const hash = fnv1a('')
    expect(hash).toBe(2166136261)
  })

  it('handles unicode characters', () => {
    const hash = fnv1a('workflows/工作流程.json')
    expect(hash).toBeGreaterThanOrEqual(0)
    expect(hash).toBeLessThanOrEqual(0xffffffff)
  })

  it('handles special characters', () => {
    const hash = fnv1a('workflows/My Workflow (Copy 2).json')
    expect(hash).toBeGreaterThanOrEqual(0)
  })
})

describe('hashPath', () => {
  it('returns 8-character hex string', () => {
    const result = hashPath('workflows/test.json')
    expect(result).toMatch(/^[0-9a-f]{8}$/)
  })

  it('pads short hashes with leading zeros', () => {
    const result = hashPath('')
    expect(result).toHaveLength(8)
    expect(result).toBe('811c9dc5')
  })

  it('returns consistent results', () => {
    const path = 'workflows/My Complex Workflow Name.json'
    const hash1 = hashPath(path)
    const hash2 = hashPath(path)
    expect(hash1).toBe(hash2)
  })

  it('produces different hashes for similar paths', () => {
    const hash1 = hashPath('workflows/Untitled.json')
    const hash2 = hashPath('workflows/Untitled (2).json')
    expect(hash1).not.toBe(hash2)
  })
})
