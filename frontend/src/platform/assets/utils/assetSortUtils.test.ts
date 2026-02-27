import { describe, expect, it } from 'vitest'

import type { SortableItem } from './assetSortUtils'
import { sortAssets } from './assetSortUtils'

function createItem(
  name: string,
  options: { label?: string; created_at?: string } = {}
): SortableItem {
  return { name, ...options }
}

describe('sortAssets', () => {
  describe('default sort', () => {
    it('preserves original order', () => {
      const items = [createItem('z'), createItem('a'), createItem('m')]
      const result = sortAssets(items, 'default')
      expect(result.map((i) => i.name)).toEqual(['z', 'a', 'm'])
    })

    it('returns a new array (does not mutate)', () => {
      const items = [createItem('z'), createItem('a')]
      const result = sortAssets(items, 'default')
      expect(result).not.toBe(items)
    })
  })

  describe('name-asc sort', () => {
    it('sorts alphabetically A-Z by name', () => {
      const items = [
        createItem('cherry'),
        createItem('apple'),
        createItem('banana')
      ]
      const result = sortAssets(items, 'name-asc')
      expect(result.map((i) => i.name)).toEqual(['apple', 'banana', 'cherry'])
    })

    it('prefers label over name when label exists', () => {
      const items = [
        createItem('file_c.png', { label: 'Cherry' }),
        createItem('file_a.png', { label: 'Apple' }),
        createItem('file_b.png', { label: 'Banana' })
      ]
      const result = sortAssets(items, 'name-asc')
      expect(result.map((i) => i.name)).toEqual([
        'file_a.png',
        'file_b.png',
        'file_c.png'
      ])
    })

    it('uses natural sort for numeric values', () => {
      const items = [
        createItem('img_10.png'),
        createItem('img_2.png'),
        createItem('img_1.png'),
        createItem('img_20.png')
      ]
      const result = sortAssets(items, 'name-asc')
      expect(result.map((i) => i.name)).toEqual([
        'img_1.png',
        'img_2.png',
        'img_10.png',
        'img_20.png'
      ])
    })

    it('is case-insensitive', () => {
      const items = [
        createItem('Banana'),
        createItem('apple'),
        createItem('CHERRY')
      ]
      const result = sortAssets(items, 'name-asc')
      expect(result.map((i) => i.name)).toEqual(['apple', 'Banana', 'CHERRY'])
    })
  })

  describe('name-desc sort', () => {
    it('sorts alphabetically Z-A by name', () => {
      const items = [
        createItem('apple'),
        createItem('cherry'),
        createItem('banana')
      ]
      const result = sortAssets(items, 'name-desc')
      expect(result.map((i) => i.name)).toEqual(['cherry', 'banana', 'apple'])
    })
  })

  describe('recent sort', () => {
    it('sorts by created_at descending (newest first)', () => {
      const items = [
        createItem('old', { created_at: '2024-01-01T00:00:00Z' }),
        createItem('newest', { created_at: '2024-03-01T00:00:00Z' }),
        createItem('middle', { created_at: '2024-02-01T00:00:00Z' })
      ]
      const result = sortAssets(items, 'recent')
      expect(result.map((i) => i.name)).toEqual(['newest', 'middle', 'old'])
    })

    it('handles null/undefined created_at (sorts to end)', () => {
      const items = [
        createItem('no-date'),
        createItem('has-date', { created_at: '2024-01-01T00:00:00Z' }),
        createItem('undefined-date', { created_at: undefined })
      ]
      const result = sortAssets(items, 'recent')
      expect(result[0].name).toBe('has-date')
    })
  })

  describe('immutability', () => {
    it('does not mutate the original array', () => {
      const items = [createItem('z'), createItem('a')]
      const original = [...items]
      sortAssets(items, 'name-asc')
      expect(items).toEqual(original)
    })
  })
})
