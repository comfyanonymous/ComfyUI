import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { hashPath } from '../base/hashUtil'
import {
  cleanupV1Data,
  getMigrationStatus,
  isV2MigrationComplete,
  migrateV1toV2
} from './migrateV1toV2'

describe('migrateV1toV2', () => {
  const workspaceId = 'test-workspace'

  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
    sessionStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
    sessionStorage.clear()
  })

  function setV1Data(
    drafts: Record<
      string,
      { data: string; updatedAt: number; name: string; isTemporary: boolean }
    >,
    order: string[]
  ) {
    localStorage.setItem(
      `Comfy.Workflow.Drafts:${workspaceId}`,
      JSON.stringify(drafts)
    )
    localStorage.setItem(
      `Comfy.Workflow.DraftOrder:${workspaceId}`,
      JSON.stringify(order)
    )
  }

  describe('isV2MigrationComplete', () => {
    it('returns false when no V2 index exists', () => {
      expect(isV2MigrationComplete(workspaceId)).toBe(false)
    })

    it('returns true when V2 index exists', () => {
      localStorage.setItem(
        `Comfy.Workflow.DraftIndex.v2:${workspaceId}`,
        JSON.stringify({ v: 2, order: [], entries: {}, updatedAt: Date.now() })
      )
      expect(isV2MigrationComplete(workspaceId)).toBe(true)
    })
  })

  describe('migrateV1toV2', () => {
    it('returns -1 if V2 already exists', () => {
      localStorage.setItem(
        `Comfy.Workflow.DraftIndex.v2:${workspaceId}`,
        JSON.stringify({ v: 2, order: [], entries: {}, updatedAt: Date.now() })
      )

      expect(migrateV1toV2(workspaceId)).toBe(-1)
    })

    it('creates empty V2 index if no V1 data', () => {
      expect(migrateV1toV2(workspaceId)).toBe(0)

      const indexJson = localStorage.getItem(
        `Comfy.Workflow.DraftIndex.v2:${workspaceId}`
      )
      expect(indexJson).not.toBeNull()

      const index = JSON.parse(indexJson!)
      expect(index.v).toBe(2)
      expect(index.order).toEqual([])
    })

    it('migrates V1 drafts to V2 format', () => {
      const v1Drafts = {
        'workflows/a.json': {
          data: '{"nodes":[1]}',
          updatedAt: 1000,
          name: 'a',
          isTemporary: true
        },
        'workflows/b.json': {
          data: '{"nodes":[2]}',
          updatedAt: 2000,
          name: 'b',
          isTemporary: false
        }
      }
      setV1Data(v1Drafts, ['workflows/a.json', 'workflows/b.json'])

      const migrated = migrateV1toV2(workspaceId)
      expect(migrated).toBe(2)

      // Check V2 index
      const indexJson = localStorage.getItem(
        `Comfy.Workflow.DraftIndex.v2:${workspaceId}`
      )
      const index = JSON.parse(indexJson!)
      expect(index.order).toHaveLength(2)

      // Check payloads
      const keyA = hashPath('workflows/a.json')
      const keyB = hashPath('workflows/b.json')

      const payloadA = localStorage.getItem(
        `Comfy.Workflow.Draft.v2:${workspaceId}:${keyA}`
      )
      const payloadB = localStorage.getItem(
        `Comfy.Workflow.Draft.v2:${workspaceId}:${keyB}`
      )

      expect(payloadA).not.toBeNull()
      expect(payloadB).not.toBeNull()

      expect(JSON.parse(payloadA!).data).toBe('{"nodes":[1]}')
      expect(JSON.parse(payloadB!).data).toBe('{"nodes":[2]}')
    })

    it('preserves LRU order during migration', () => {
      const v1Drafts = {
        'workflows/first.json': {
          data: '{}',
          updatedAt: 1000,
          name: 'first',
          isTemporary: true
        },
        'workflows/second.json': {
          data: '{}',
          updatedAt: 2000,
          name: 'second',
          isTemporary: true
        },
        'workflows/third.json': {
          data: '{}',
          updatedAt: 3000,
          name: 'third',
          isTemporary: true
        }
      }
      setV1Data(v1Drafts, [
        'workflows/first.json',
        'workflows/second.json',
        'workflows/third.json'
      ])

      migrateV1toV2(workspaceId)

      const indexJson = localStorage.getItem(
        `Comfy.Workflow.DraftIndex.v2:${workspaceId}`
      )
      const index = JSON.parse(indexJson!)

      // Order should be preserved (oldest to newest)
      const expectedOrder = [
        hashPath('workflows/first.json'),
        hashPath('workflows/second.json'),
        hashPath('workflows/third.json')
      ]
      expect(index.order).toEqual(expectedOrder)
    })

    it('keeps V1 data intact after migration', () => {
      const v1Drafts = {
        'workflows/test.json': {
          data: '{}',
          updatedAt: 1000,
          name: 'test',
          isTemporary: true
        }
      }
      setV1Data(v1Drafts, ['workflows/test.json'])

      migrateV1toV2(workspaceId)

      // V1 data should still exist
      expect(
        localStorage.getItem(`Comfy.Workflow.Drafts:${workspaceId}`)
      ).not.toBeNull()
      expect(
        localStorage.getItem(`Comfy.Workflow.DraftOrder:${workspaceId}`)
      ).not.toBeNull()
    })
  })

  describe('cleanupV1Data', () => {
    it('removes V1 keys', () => {
      setV1Data(
        {
          'workflows/test.json': {
            data: '{}',
            updatedAt: 1,
            name: 'test',
            isTemporary: true
          }
        },
        ['workflows/test.json']
      )

      cleanupV1Data(workspaceId)

      expect(
        localStorage.getItem(`Comfy.Workflow.Drafts:${workspaceId}`)
      ).toBeNull()
      expect(
        localStorage.getItem(`Comfy.Workflow.DraftOrder:${workspaceId}`)
      ).toBeNull()
    })
  })

  describe('getMigrationStatus', () => {
    it('reports correct status', () => {
      setV1Data(
        {
          'workflows/a.json': {
            data: '{}',
            updatedAt: 1,
            name: 'a',
            isTemporary: true
          },
          'workflows/b.json': {
            data: '{}',
            updatedAt: 2,
            name: 'b',
            isTemporary: true
          }
        },
        ['workflows/a.json', 'workflows/b.json']
      )

      const status = getMigrationStatus(workspaceId)
      expect(status.v1Exists).toBe(true)
      expect(status.v2Exists).toBe(false)
      expect(status.v1DraftCount).toBe(2)
      expect(status.v2DraftCount).toBe(0)
    })
  })
})
