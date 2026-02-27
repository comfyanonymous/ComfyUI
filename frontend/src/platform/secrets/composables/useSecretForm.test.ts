import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SecretMetadata, SecretProvider } from '../types'
import { useSecretForm } from './useSecretForm'

vi.mock('vue-i18n', () => ({
  useI18n: () => ({ t: (key: string) => key })
}))

const mockCreate = vi.fn()
const mockUpdate = vi.fn()

vi.mock('../api/secretsApi', () => ({
  createSecret: (payload: unknown) => mockCreate(payload),
  updateSecret: (id: string, payload: unknown) => mockUpdate(id, payload),
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

describe('useSecretForm', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('validation via handleSubmit', () => {
    it('requires name in create mode', async () => {
      const visible = ref(true)
      const { form, errors, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = ''
      form.provider = 'huggingface'
      form.secretValue = 'secret123'

      await handleSubmit()

      expect(mockCreate).not.toHaveBeenCalled()
      expect(errors.name).toBe('secrets.errors.nameRequired')
    })

    it('requires name not to exceed 255 characters', async () => {
      const visible = ref(true)
      const { form, errors, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'a'.repeat(256)
      form.provider = 'huggingface'
      form.secretValue = 'secret123'

      await handleSubmit()

      expect(mockCreate).not.toHaveBeenCalled()
      expect(errors.name).toBe('secrets.errors.nameTooLong')
    })

    it('requires provider in create mode', async () => {
      const visible = ref(true)
      const { form, errors, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'My Secret'
      form.provider = null
      form.secretValue = 'secret123'

      await handleSubmit()

      expect(mockCreate).not.toHaveBeenCalled()
      expect(errors.provider).toBe('secrets.errors.providerRequired')
    })

    it('requires secret value in create mode', async () => {
      const visible = ref(true)
      const { form, errors, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'My Secret'
      form.provider = 'huggingface'
      form.secretValue = ''

      await handleSubmit()

      expect(mockCreate).not.toHaveBeenCalled()
      expect(errors.secretValue).toBe('secrets.errors.secretValueRequired')
    })

    it('allows empty secret value in edit mode', async () => {
      const visible = ref(false)
      const secret = createMockSecret()
      mockUpdate.mockResolvedValue({})
      const { form, handleSubmit } = useSecretForm({
        mode: 'edit',
        secret: () => secret,
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      visible.value = true
      await Promise.resolve()

      form.name = 'Updated Name'
      form.secretValue = ''

      await handleSubmit()

      expect(mockUpdate).toHaveBeenCalled()
    })

    it('requires provider in edit mode', async () => {
      const visible = ref(true)
      const secret = createMockSecret({ provider: undefined })
      const { form, errors, handleSubmit } = useSecretForm({
        mode: 'edit',
        secret: () => secret,
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'Updated Name'
      form.provider = null

      await handleSubmit()

      expect(mockUpdate).not.toHaveBeenCalled()
      expect(errors.provider).toBe('secrets.errors.providerRequired')
    })
  })

  describe('providerOptions', () => {
    it('marks existing providers as disabled', () => {
      const visible = ref(true)
      const { providerOptions } = useSecretForm({
        mode: 'create',
        existingProviders: () => ['huggingface'],
        visible,
        onSaved: vi.fn()
      })

      const huggingface = providerOptions.value.find(
        (o) => o.value === 'huggingface'
      )
      const civitai = providerOptions.value.find((o) => o.value === 'civitai')

      expect(huggingface?.disabled).toBe(true)
      expect(civitai?.disabled).toBe(false)
    })

    it('updates disabled state when existingProviders changes', () => {
      const visible = ref(true)
      const existingProviders = ref<SecretProvider[]>(['huggingface'])
      const { providerOptions } = useSecretForm({
        mode: 'create',
        existingProviders: () => existingProviders.value,
        visible,
        onSaved: vi.fn()
      })

      expect(
        providerOptions.value.find((o) => o.value === 'huggingface')?.disabled
      ).toBe(true)

      existingProviders.value = []

      expect(
        providerOptions.value.find((o) => o.value === 'huggingface')?.disabled
      ).toBe(false)
    })
  })

  describe('handleSubmit', () => {
    it('calls create API with correct payload', async () => {
      const visible = ref(true)
      const onSaved = vi.fn()
      mockCreate.mockResolvedValue({})

      const { form, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved
      })

      form.name = 'My Secret'
      form.provider = 'civitai'
      form.secretValue = 'secret123'

      await handleSubmit()

      expect(mockCreate).toHaveBeenCalledWith({
        name: 'My Secret',
        secret_value: 'secret123',
        provider: 'civitai'
      })
      expect(onSaved).toHaveBeenCalled()
      expect(visible.value).toBe(false)
    })

    it('calls update API with correct payload', async () => {
      const visible = ref(false)
      const onSaved = vi.fn()
      const secret = createMockSecret({ id: 'secret-123' })
      mockUpdate.mockResolvedValue({})

      const { form, handleSubmit } = useSecretForm({
        mode: 'edit',
        secret: () => secret,
        existingProviders: () => [],
        visible,
        onSaved
      })

      visible.value = true
      await Promise.resolve()

      form.name = 'Updated Name'
      form.secretValue = 'newvalue'

      await handleSubmit()

      expect(mockUpdate).toHaveBeenCalledWith('secret-123', {
        name: 'Updated Name',
        secret_value: 'newvalue'
      })
      expect(onSaved).toHaveBeenCalled()
    })

    it('omits secret_value in update if empty', async () => {
      const visible = ref(false)
      const secret = createMockSecret({ id: 'secret-123' })
      mockUpdate.mockResolvedValue({})

      const { form, handleSubmit } = useSecretForm({
        mode: 'edit',
        secret: () => secret,
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      visible.value = true
      await Promise.resolve()

      form.name = 'Updated Name'
      form.secretValue = ''

      await handleSubmit()

      expect(mockUpdate).toHaveBeenCalledWith('secret-123', {
        name: 'Updated Name'
      })
    })

    it('sets apiError on API failure', async () => {
      const { SecretsApiError } = await import('../api/secretsApi')
      const visible = ref(true)
      mockCreate.mockRejectedValue(
        new SecretsApiError('Duplicate', 400, 'DUPLICATE_NAME')
      )

      const { form, apiError, handleSubmit } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'My Secret'
      form.provider = 'huggingface'
      form.secretValue = 'secret123'

      await handleSubmit()

      expect(apiError.value).toBe('secrets.errors.duplicateName')
      expect(visible.value).toBe(true)
    })
  })

  describe('form reset on visible', () => {
    it('resets to empty state in create mode when visible becomes true', async () => {
      const visible = ref(false)
      const { form, errors } = useSecretForm({
        mode: 'create',
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'Some Name'
      form.secretValue = 'some value'
      errors.name = 'some error'

      visible.value = true
      await Promise.resolve()

      expect(form.name).toBe('')
      expect(form.secretValue).toBe('')
      expect(form.provider).toBeNull()
      expect(errors.name).toBe('')
    })

    it('resets to secret values in edit mode when visible becomes true', async () => {
      const visible = ref(false)
      const secret = createMockSecret({
        name: 'Original Name',
        provider: 'civitai'
      })
      const { form } = useSecretForm({
        mode: 'edit',
        secret: () => secret,
        existingProviders: () => [],
        visible,
        onSaved: vi.fn()
      })

      form.name = 'Changed Name'

      visible.value = true
      await Promise.resolve()

      expect(form.name).toBe('Original Name')
      expect(form.provider).toBe('civitai')
      expect(form.secretValue).toBe('')
    })
  })
})
