import { useClipboard } from '@vueuse/core'
import { useToast } from 'primevue/usetoast'

import { t } from '@/i18n'

export function useCopyToClipboard() {
  const { copy, copied } = useClipboard({ legacy: true })
  const toast = useToast()

  async function copyToClipboard(text: string) {
    try {
      await copy(text)
      if (copied.value) {
        toast.add({
          severity: 'success',
          summary: t('g.success'),
          detail: t('clipboard.successMessage'),
          life: 3000
        })
      } else {
        toast.add({
          severity: 'error',
          summary: t('g.error'),
          detail: t('clipboard.errorMessage')
        })
      }
    } catch {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('clipboard.errorMessage')
      })
    }
  }

  return {
    copyToClipboard
  }
}
