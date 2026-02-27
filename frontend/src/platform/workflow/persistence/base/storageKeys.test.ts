import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('storageKeys', () => {
  beforeEach(() => {
    vi.resetModules()
    sessionStorage.clear()
  })

  describe('getWorkspaceId', () => {
    it('returns personal when no workspace is set', async () => {
      const { getWorkspaceId } = await import('./storageKeys')
      expect(getWorkspaceId()).toBe('personal')
    })

    it('returns personal for personal workspace type', async () => {
      sessionStorage.setItem(
        'Comfy.Workspace.Current',
        JSON.stringify({ type: 'personal', id: null })
      )
      const { getWorkspaceId } = await import('./storageKeys')
      expect(getWorkspaceId()).toBe('personal')
    })

    it('returns workspace ID for team workspace', async () => {
      sessionStorage.setItem(
        'Comfy.Workspace.Current',
        JSON.stringify({ type: 'team', id: 'ws-abc-123' })
      )
      const { getWorkspaceId } = await import('./storageKeys')
      expect(getWorkspaceId()).toBe('ws-abc-123')
    })

    it('returns personal when JSON parsing fails', async () => {
      sessionStorage.setItem('Comfy.Workspace.Current', 'invalid-json')
      const { getWorkspaceId } = await import('./storageKeys')
      expect(getWorkspaceId()).toBe('personal')
    })

    it('returns personal when workspace has no id', async () => {
      sessionStorage.setItem(
        'Comfy.Workspace.Current',
        JSON.stringify({ type: 'team', id: '' })
      )
      const { getWorkspaceId } = await import('./storageKeys')
      expect(getWorkspaceId()).toBe('personal')
    })

    it('reads fresh value on each call (not cached)', async () => {
      const { getWorkspaceId } = await import('./storageKeys')

      // Initially no workspace set
      expect(getWorkspaceId()).toBe('personal')

      // Set workspace after import
      sessionStorage.setItem(
        'Comfy.Workspace.Current',
        JSON.stringify({ type: 'team', id: 'ws-new' })
      )

      // Should read the new value (not cached 'personal')
      expect(getWorkspaceId()).toBe('ws-new')

      // Change workspace again
      sessionStorage.setItem(
        'Comfy.Workspace.Current',
        JSON.stringify({ type: 'team', id: 'ws-another' })
      )

      expect(getWorkspaceId()).toBe('ws-another')
    })
  })

  describe('StorageKeys', () => {
    it('generates draftIndex key with workspace scope', async () => {
      const { StorageKeys } = await import('./storageKeys')

      expect(StorageKeys.draftIndex('ws-123')).toBe(
        'Comfy.Workflow.DraftIndex.v2:ws-123'
      )
    })

    it('generates draftPayload key with hash', async () => {
      const { StorageKeys } = await import('./storageKeys')
      const key = StorageKeys.draftPayload('workflows/test.json', 'ws-1')

      expect(key).toMatch(/^Comfy\.Workflow\.Draft\.v2:ws-1:[0-9a-f]{8}$/)
    })

    it('generates consistent draftKey from path', async () => {
      const { StorageKeys } = await import('./storageKeys')
      const key1 = StorageKeys.draftKey('workflows/test.json')
      const key2 = StorageKeys.draftKey('workflows/test.json')

      expect(key1).toBe(key2)
      expect(key1).toMatch(/^[0-9a-f]{8}$/)
    })

    it('generates activePath key with clientId', async () => {
      const { StorageKeys } = await import('./storageKeys')
      expect(StorageKeys.activePath('client-abc')).toBe(
        'Comfy.Workflow.ActivePath:client-abc'
      )
    })

    it('generates openPaths key with clientId', async () => {
      const { StorageKeys } = await import('./storageKeys')
      expect(StorageKeys.openPaths('client-abc')).toBe(
        'Comfy.Workflow.OpenPaths:client-abc'
      )
    })

    it('exposes prefix patterns for cleanup', async () => {
      const { StorageKeys } = await import('./storageKeys')

      expect(StorageKeys.prefixes.draftIndex).toBe(
        'Comfy.Workflow.DraftIndex.v2:'
      )
      expect(StorageKeys.prefixes.draftPayload).toBe('Comfy.Workflow.Draft.v2:')
      expect(StorageKeys.prefixes.activePath).toBe('Comfy.Workflow.ActivePath:')
      expect(StorageKeys.prefixes.openPaths).toBe('Comfy.Workflow.OpenPaths:')
    })
  })
})
