import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useToastStore } from '@/platform/updates/common/toastStore'

import {
  deleteSecret as deleteSecretApi,
  listSecrets,
  SecretsApiError
} from '../api/secretsApi'
import type { SecretMetadata, SecretProvider } from '../types'

export function useSecrets() {
  const { t } = useI18n()
  const toastStore = useToastStore()

  const loading = ref(false)
  const secrets = ref<SecretMetadata[]>([])
  const operatingSecretId = ref<string | null>(null)

  const existingProviders = computed<SecretProvider[]>(() =>
    secrets.value
      .map((s) => s.provider)
      .filter((p): p is SecretProvider => p !== undefined)
  )

  async function fetchSecrets() {
    loading.value = true
    try {
      secrets.value = await listSecrets()
    } catch (err) {
      if (err instanceof SecretsApiError) {
        toastStore.add({
          severity: 'error',
          summary: t('g.error'),
          detail: err.message,
          life: 5000
        })
      } else {
        console.error('Unexpected error fetching secrets:', err)
        toastStore.add({
          severity: 'error',
          summary: t('g.error'),
          detail: t('g.unknownError'),
          life: 5000
        })
      }
    } finally {
      loading.value = false
    }
  }

  async function deleteSecret(secret: SecretMetadata) {
    operatingSecretId.value = secret.id
    try {
      await deleteSecretApi(secret.id)
      secrets.value = secrets.value.filter((s) => s.id !== secret.id)
    } catch (err) {
      if (err instanceof SecretsApiError) {
        toastStore.add({
          severity: 'error',
          summary: t('g.error'),
          detail: err.message,
          life: 5000
        })
      } else {
        console.error('Unexpected error deleting secret:', err)
        toastStore.add({
          severity: 'error',
          summary: t('g.error'),
          detail: t('g.unknownError'),
          life: 5000
        })
      }
    } finally {
      operatingSecretId.value = null
    }
  }

  return {
    loading,
    secrets,
    operatingSecretId,
    existingProviders,
    fetchSecrets,
    deleteSecret
  }
}
