import { describe, expect, it } from 'vitest'

import {
  createEmptyIndex,
  getEntryByPath,
  getMostRecentKey,
  moveEntry,
  removeEntry,
  removeOrphanedEntries,
  touchOrder,
  upsertEntry
} from './draftCacheV2'
import { hashPath } from './hashUtil'

describe('draftCacheV2', () => {
  describe('createEmptyIndex', () => {
    it('creates index with version 2', () => {
      const index = createEmptyIndex()
      expect(index.v).toBe(2)
      expect(index.order).toEqual([])
      expect(index.entries).toEqual({})
    })
  })

  describe('touchOrder', () => {
    it('moves existing key to end', () => {
      expect(touchOrder(['a', 'b', 'c'], 'a')).toEqual(['b', 'c', 'a'])
    })

    it('adds new key to end', () => {
      expect(touchOrder(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
    })

    it('handles empty order', () => {
      expect(touchOrder([], 'a')).toEqual(['a'])
    })
  })

  describe('upsertEntry', () => {
    it('adds new entry to empty index', () => {
      const index = createEmptyIndex()
      const { index: updated, evicted } = upsertEntry(
        index,
        'workflows/a.json',
        {
          name: 'a',
          isTemporary: true,
          updatedAt: Date.now()
        }
      )

      expect(updated.order).toHaveLength(1)
      expect(Object.keys(updated.entries)).toHaveLength(1)
      expect(evicted).toEqual([])
    })

    it('updates existing entry and moves to end of order', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index

      const keyA = hashPath('workflows/a.json')
      const keyB = hashPath('workflows/b.json')
      expect(index.order).toEqual([keyA, keyB])

      const { index: updated } = upsertEntry(index, 'workflows/a.json', {
        name: 'a-updated',
        isTemporary: false,
        updatedAt: 3000
      })

      expect(updated.order).toEqual([keyB, keyA])
      expect(updated.entries[keyA].name).toBe('a-updated')
      expect(updated.entries[keyA].isTemporary).toBe(false)
    })

    it('evicts oldest entries when over limit', () => {
      let index = createEmptyIndex()
      const limit = 3

      for (let i = 0; i < limit; i++) {
        index = upsertEntry(
          index,
          `workflows/draft${i}.json`,
          { name: `draft${i}`, isTemporary: true, updatedAt: i },
          limit
        ).index
      }

      expect(index.order).toHaveLength(3)

      const { index: updated, evicted } = upsertEntry(
        index,
        'workflows/new.json',
        { name: 'new', isTemporary: true, updatedAt: 100 },
        limit
      )

      expect(updated.order).toHaveLength(3)
      expect(evicted).toHaveLength(1)
      expect(evicted[0]).toBe(hashPath('workflows/draft0.json'))
    })

    it('does not evict the entry being upserted', () => {
      let index = createEmptyIndex()

      index = upsertEntry(
        index,
        'workflows/only.json',
        { name: 'only', isTemporary: true, updatedAt: 1000 },
        1
      ).index

      const { index: updated, evicted } = upsertEntry(
        index,
        'workflows/only.json',
        { name: 'only-updated', isTemporary: true, updatedAt: 2000 },
        1
      )

      expect(updated.order).toHaveLength(1)
      expect(evicted).toHaveLength(0)
    })
  })

  describe('removeEntry', () => {
    it('removes existing entry', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/test.json', {
        name: 'test',
        isTemporary: true,
        updatedAt: Date.now()
      }).index

      const { index: updated, removedKey } = removeEntry(
        index,
        'workflows/test.json'
      )

      expect(updated.order).toHaveLength(0)
      expect(Object.keys(updated.entries)).toHaveLength(0)
      expect(removedKey).toBe(hashPath('workflows/test.json'))
    })

    it('returns null for non-existent path', () => {
      const index = createEmptyIndex()
      const { index: updated, removedKey } = removeEntry(
        index,
        'workflows/missing.json'
      )

      expect(updated).toBe(index)
      expect(removedKey).toBeNull()
    })
  })

  describe('moveEntry', () => {
    it('moves entry to new path with new name', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/old.json', {
        name: 'old',
        isTemporary: true,
        updatedAt: 1000
      }).index

      const result = moveEntry(
        index,
        'workflows/old.json',
        'workflows/new.json',
        'new'
      )

      expect(result).not.toBeNull()
      expect(result!.index.entries[result!.newKey].name).toBe('new')
      expect(result!.index.entries[result!.newKey].path).toBe(
        'workflows/new.json'
      )
      expect(result!.index.entries[result!.oldKey]).toBeUndefined()
    })

    it('returns null for non-existent source', () => {
      const index = createEmptyIndex()
      const result = moveEntry(
        index,
        'workflows/missing.json',
        'workflows/new.json',
        'new'
      )
      expect(result).toBeNull()
    })

    it('returns null when destination path already exists', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index

      const result = moveEntry(
        index,
        'workflows/a.json',
        'workflows/b.json',
        'b-renamed'
      )
      expect(result).toBeNull()
    })

    it('moves entry to end of order', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index

      const result = moveEntry(
        index,
        'workflows/a.json',
        'workflows/c.json',
        'c'
      )

      const keyB = hashPath('workflows/b.json')
      const keyC = hashPath('workflows/c.json')
      expect(result!.index.order).toEqual([keyB, keyC])
    })
  })

  describe('getMostRecentKey', () => {
    it('returns last key in order', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index

      expect(getMostRecentKey(index)).toBe(hashPath('workflows/b.json'))
    })

    it('returns null for empty index', () => {
      expect(getMostRecentKey(createEmptyIndex())).toBeNull()
    })
  })

  describe('getEntryByPath', () => {
    it('returns entry for existing path', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/test.json', {
        name: 'test',
        isTemporary: true,
        updatedAt: 1000
      }).index

      const entry = getEntryByPath(index, 'workflows/test.json')
      expect(entry).not.toBeNull()
      expect(entry!.name).toBe('test')
    })

    it('returns null for missing path', () => {
      const index = createEmptyIndex()
      expect(getEntryByPath(index, 'workflows/missing.json')).toBeNull()
    })
  })

  describe('removeOrphanedEntries', () => {
    it('removes entries without payloads', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index

      const existingKeys = new Set([hashPath('workflows/a.json')])
      const cleaned = removeOrphanedEntries(index, existingKeys)

      expect(cleaned.order).toHaveLength(1)
      expect(Object.keys(cleaned.entries)).toHaveLength(1)
      expect(cleaned.entries[hashPath('workflows/a.json')]).toBeDefined()
    })

    it('preserves order of remaining entries', () => {
      let index = createEmptyIndex()
      index = upsertEntry(index, 'workflows/a.json', {
        name: 'a',
        isTemporary: true,
        updatedAt: 1000
      }).index
      index = upsertEntry(index, 'workflows/b.json', {
        name: 'b',
        isTemporary: true,
        updatedAt: 2000
      }).index
      index = upsertEntry(index, 'workflows/c.json', {
        name: 'c',
        isTemporary: true,
        updatedAt: 3000
      }).index

      const keyA = hashPath('workflows/a.json')
      const keyC = hashPath('workflows/c.json')
      const existingKeys = new Set([keyA, keyC])
      const cleaned = removeOrphanedEntries(index, existingKeys)

      expect(cleaned.order).toEqual([keyA, keyC])
    })
  })
})
