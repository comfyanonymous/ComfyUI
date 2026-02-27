import { whenever } from '@vueuse/core'
import type { MaybeRefOrGetter } from 'vue'
import { computed, reactive, ref, toValue } from 'vue'
import { useI18n } from 'vue-i18n'

import { createSecret, SecretsApiError, updateSecret } from '../api/secretsApi'
import { SECRET_PROVIDERS } from '../providers'
import type { SecretErrorCode, SecretMetadata, SecretProvider } from '../types'

interface SecretFormState {
  name: string
  secretValue: string
  provider: SecretProvider | null
}

interface SecretFormErrors {
  name: string
  secretValue: string
  provider: string
}

interface ProviderOption {
  label: string
  value: SecretProvider
  disabled: boolean
}

interface UseSecretFormOptions {
  mode: 'create' | 'edit'
  secret?: MaybeRefOrGetter<SecretMetadata | undefined>
  existingProviders: MaybeRefOrGetter<SecretProvider[]>
  visible: { value: boolean }
  onSaved: () => void
}

export function useSecretForm(options: UseSecretFormOptions) {
  const { t } = useI18n()
  const {
    mode,
    secret: secretRef,
    existingProviders,
    visible,
    onSaved
  } = options

  const loading = ref(false)
  const apiErrorCode = ref<SecretErrorCode | null>(null)
  const apiErrorMessage = ref<string | null>(null)

  const form = reactive<SecretFormState>({
    name: '',
    secretValue: '',
    provider: null
  })

  const errors = reactive<SecretFormErrors>({
    name: '',
    secretValue: '',
    provider: ''
  })

  const providerOptions = computed<ProviderOption[]>(() =>
    SECRET_PROVIDERS.map((p) => ({
      label: p.label,
      value: p.value,
      disabled:
        mode === 'edit' ? false : toValue(existingProviders).includes(p.value)
    }))
  )

  const apiError = computed(() => {
    if (!apiErrorCode.value && !apiErrorMessage.value) return null
    switch (apiErrorCode.value) {
      case 'DUPLICATE_NAME':
        return t('secrets.errors.duplicateName')
      case 'DUPLICATE_PROVIDER':
        return t('secrets.errors.duplicateProvider')
      default:
        return apiErrorMessage.value
    }
  })

  function resetForm() {
    const secret = toValue(secretRef)
    if (mode === 'edit' && secret) {
      form.name = secret.name
      form.provider = secret.provider ?? null
      form.secretValue = ''
    } else {
      form.name = ''
      form.secretValue = ''
      form.provider = null
    }
    errors.name = ''
    errors.secretValue = ''
    errors.provider = ''
    apiErrorCode.value = null
    apiErrorMessage.value = null
  }

  whenever(() => visible.value, resetForm)

  function validate(): boolean {
    errors.name = ''
    errors.secretValue = ''
    errors.provider = ''

    if (!form.name.trim()) {
      errors.name = t('secrets.errors.nameRequired')
      return false
    }
    if (form.name.length > 255) {
      errors.name = t('secrets.errors.nameTooLong')
      return false
    }
    if (!form.provider) {
      errors.provider = t('secrets.errors.providerRequired')
      return false
    }
    if (mode === 'create' && !form.secretValue) {
      errors.secretValue = t('secrets.errors.secretValueRequired')
      return false
    }
    return true
  }

  async function handleSubmit() {
    if (!validate()) return

    loading.value = true
    apiErrorCode.value = null
    apiErrorMessage.value = null

    try {
      const secret = toValue(secretRef)
      if (mode === 'create') {
        await createSecret({
          name: form.name.trim(),
          secret_value: form.secretValue,
          provider: form.provider!
        })
      } else if (secret) {
        const updatePayload: { name: string; secret_value?: string } = {
          name: form.name.trim()
        }
        if (form.secretValue) {
          updatePayload.secret_value = form.secretValue
        }
        await updateSecret(secret.id, updatePayload)
      }
      onSaved()
      visible.value = false
    } catch (err) {
      if (err instanceof SecretsApiError) {
        apiErrorCode.value = err.code ?? null
        apiErrorMessage.value = err.message
      }
    } finally {
      loading.value = false
    }
  }

  return {
    form,
    errors,
    loading,
    apiError,
    providerOptions,
    handleSubmit
  }
}
