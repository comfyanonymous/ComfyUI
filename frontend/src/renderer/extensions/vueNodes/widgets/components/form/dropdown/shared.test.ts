import { describe, expect, it } from 'vitest'

import { defaultSearcher, getDefaultSortOptions } from './shared'
import type { FormDropdownItem } from './types'

function createItem(name: string, label?: string): FormDropdownItem {
  return {
    id: name,
    preview_url: '',
    name,
    label
  }
}

describe('defaultSearcher', () => {
  const items: FormDropdownItem[] = [
    createItem('apple.png'),
    createItem('banana.jpg'),
    createItem('cherry.gif')
  ]

  it('returns all items when query is empty', async () => {
    const result = await defaultSearcher('', items)
    expect(result).toEqual(items)
  })

  it('returns all items when query is whitespace', async () => {
    const result = await defaultSearcher('   ', items)
    expect(result).toEqual(items)
  })

  it('filters items by single word', async () => {
    const result = await defaultSearcher('apple', items)
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('apple.png')
  })

  it('filters items case-insensitively', async () => {
    const result = await defaultSearcher('APPLE', items)
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('apple.png')
  })

  it('filters items by multiple words (AND logic)', async () => {
    const result = await defaultSearcher('a png', items)
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('apple.png')
  })

  it('returns empty array when no matches', async () => {
    const result = await defaultSearcher('xyz', items)
    expect(result).toHaveLength(0)
  })
})

describe('getDefaultSortOptions', () => {
  const sortOptions = getDefaultSortOptions()

  describe('Default sorter', () => {
    const defaultSorter = sortOptions.find((o) => o.id === 'default')!.sorter

    it('returns items in original order', () => {
      const items = [createItem('z'), createItem('a'), createItem('m')]
      const result = defaultSorter({ items })
      expect(result.map((i) => i.name)).toEqual(['z', 'a', 'm'])
    })

    it('does not mutate original array', () => {
      const items = [createItem('z'), createItem('a')]
      const result = defaultSorter({ items })
      expect(result).not.toBe(items)
    })
  })

  describe('A-Z sorter', () => {
    const azSorter = sortOptions.find((o) => o.id === 'name-asc')!.sorter

    it('sorts items alphabetically by name', () => {
      const items = [
        createItem('cherry'),
        createItem('apple'),
        createItem('banana')
      ]
      const result = azSorter({ items })
      expect(result.map((i) => i.name)).toEqual(['apple', 'banana', 'cherry'])
    })

    it('sorts items alphabetically by label when available', () => {
      const items = [
        createItem('file_c.png', 'Cherry'),
        createItem('file_a.png', 'Apple'),
        createItem('file_b.png', 'Banana')
      ]
      const result = azSorter({ items })
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
      const result = azSorter({ items })
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
      const result = azSorter({ items })
      expect(result.map((i) => i.name)).toEqual(['apple', 'Banana', 'CHERRY'])
    })

    it('falls back to name when label is undefined', () => {
      const items = [
        createItem('z_file.png', 'Alpha'),
        createItem('a_file.png'),
        createItem('m_file.png', 'Beta')
      ]
      const result = azSorter({ items })
      // 'a_file.png' (no label, uses name), 'Alpha', 'Beta'
      expect(result.map((i) => i.name)).toEqual([
        'a_file.png',
        'z_file.png',
        'm_file.png'
      ])
    })

    it('does not mutate original array', () => {
      const items = [createItem('z'), createItem('a')]
      const result = azSorter({ items })
      expect(result).not.toBe(items)
      expect(items[0].name).toBe('z')
    })
  })
})
