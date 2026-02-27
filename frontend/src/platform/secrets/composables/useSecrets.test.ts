import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SecretMetadata } from '../types'
import { useSecrets } from './useSecrets'

const mockAdd = vi.fn()

vi.mock('vue-i18n', () => ({
  useI18n: () => ({ t: (key: string) => key })
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: () => ({ add: mockAdd })
}))

const mockListSecrets = vi.fn()
const mockDeleteSecret = vi.fn()

vi.mock('../api/secretsApi', () => ({
  listSecrets: () => mockListSecrets(),
  deleteSecret: (id: string) => mockDeleteSecret(id),
  SecretsApiError: class SecretsApiError extends Error {
    constructor(
      message: string,
      public readonly status?: number,
      public readonly code?: string
    ) {
      super(message)
      this.name = 'SecretsApiError'
    }
  }
}))

function createMockSecret(
  overrides: Partial<SecretMetadata> = {}
): SecretMetadata {
  return {
    id: 'secret-1',
    name: 'Test Secret',
    provider: 'huggingface',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    ...overrides
  }
}

describe('useSecrets', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('fetchSecrets', () => {
    it('fetches and populates secrets list', async () => {
      const mockSecrets = [
        createMockSecret({ id: '1', name: 'Secret 1' }),
        createMockSecret({ id: '2', name: 'Secret 2' })
      ]
      mockListSecrets.mockResolvedValue(mockSecrets)

      const { secrets, loading, fetchSecrets } = useSecrets()

      expect(loading.value).toBe(false)
      const fetchPromise = fetchSecrets()
      expect(loading.value).toBe(true)

      await fetchPromise

      expect(loading.value).toBe(false)
      expect(secrets.value).toEqual(mockSecrets)
    })

    it('shows error toast on API failure', async () => {
      const { SecretsApiError } = await import('../api/secretsApi')
      mockListSecrets.mockRejectedValue(
        new SecretsApiError('Network error', 500)
      )

      const { secrets, fetchSecrets } = useSecrets()

      await fetchSecrets()

      expect(secrets.value).toEqual([])
      expect(mockAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'g.error',
        detail: 'Network error',
        life: 5000
      })
    })
  })

  describe('deleteSecret', () => {
    it('deletes secret and removes from list', async () => {
      const secretToDelete = createMockSecret({ id: '1' })
      const remainingSecret = createMockSecret({ id: '2' })
      mockListSecrets.mockResolvedValue([secretToDelete, remainingSecret])
      mockDeleteSecret.mockResolvedValue(undefined)

      const { secrets, operatingSecretId, fetchSecrets, deleteSecret } =
        useSecrets()

      await fetchSecrets()
      expect(secrets.value).toHaveLength(2)

      const deletePromise = deleteSecret(secretToDelete)
      expect(operatingSecretId.value).toBe('1')

      await deletePromise

      expect(operatingSecretId.value).toBe(null)
      expect(secrets.value).toHaveLength(1)
      expect(secrets.value[0].id).toBe('2')
      expect(mockDeleteSecret).toHaveBeenCalledWith('1')
    })

    it('shows error toast on delete failure', async () => {
      const { SecretsApiError } = await import('../api/secretsApi')
      const secret = createMockSecret()
      mockListSecrets.mockResolvedValue([secret])
      mockDeleteSecret.mockRejectedValue(
        new SecretsApiError('Delete failed', 500)
      )

      const { secrets, fetchSecrets, deleteSecret } = useSecrets()

      await fetchSecrets()
      await deleteSecret(secret)

      expect(secrets.value).toHaveLength(1)
      expect(mockAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'g.error',
        detail: 'Delete failed',
        life: 5000
      })
    })
  })

  describe('existingProviders', () => {
    it('returns list of providers from secrets', async () => {
      mockListSecrets.mockResolvedValue([
        createMockSecret({ id: '1', provider: 'huggingface' }),
        createMockSecret({ id: '2', provider: 'civitai' }),
        createMockSecret({ id: '3', provider: undefined })
      ])

      const { existingProviders, fetchSecrets } = useSecrets()

      await fetchSecrets()

      expect(existingProviders.value).toEqual(['huggingface', 'civitai'])
    })
  })
})
