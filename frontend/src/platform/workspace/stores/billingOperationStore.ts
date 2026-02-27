import type { ToastMessageOptions } from 'primevue/toast'
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useBillingContext } from '@/composables/billing/useBillingContext'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { workspaceApi } from '@/platform/workspace/api/workspaceApi'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogStore } from '@/stores/dialogStore'

const INITIAL_INTERVAL_MS = 1000
const MAX_INTERVAL_MS = 8000
const BACKOFF_MULTIPLIER = 1.5
const TIMEOUT_MS = 120_000 // 2 minutes

type OperationType = 'subscription' | 'topup'
type OperationStatus = 'pending' | 'succeeded' | 'failed' | 'timeout'

interface BillingOperation {
  opId: string
  type: OperationType
  status: OperationStatus
  errorMessage: string | null
  startedAt: number
}

export const useBillingOperationStore = defineStore('billingOperation', () => {
  const operations = ref<Map<string, BillingOperation>>(new Map())
  const timeouts = new Map<string, ReturnType<typeof setTimeout>>()
  const intervals = new Map<string, number>()
  const receivedToasts = new Map<string, ToastMessageOptions>()

  const hasPendingOperations = computed(() =>
    [...operations.value.values()].some((op) => op.status === 'pending')
  )

  const isSettingUp = computed(() =>
    [...operations.value.values()].some(
      (op) => op.status === 'pending' && op.type === 'subscription'
    )
  )

  const isAddingCredits = computed(() =>
    [...operations.value.values()].some(
      (op) => op.status === 'pending' && op.type === 'topup'
    )
  )

  function getOperation(opId: string) {
    return operations.value.get(opId)
  }

  function startOperation(opId: string, type: OperationType) {
    if (operations.value.has(opId)) return

    const operation: BillingOperation = {
      opId,
      type,
      status: 'pending',
      errorMessage: null,
      startedAt: Date.now()
    }

    operations.value = new Map(operations.value).set(opId, operation)
    intervals.set(opId, INITIAL_INTERVAL_MS)

    // Show immediate feedback toast (persists until operation completes)
    const messageKey =
      type === 'subscription'
        ? 'billingOperation.subscriptionProcessing'
        : 'billingOperation.topupProcessing'

    const toastMessage: ToastMessageOptions = {
      severity: 'info',
      summary: t(messageKey),
      group: 'billing-operation'
    }
    receivedToasts.set(opId, toastMessage)
    useToastStore().add(toastMessage)

    void poll(opId)
  }

  async function poll(opId: string) {
    const operation = operations.value.get(opId)
    if (!operation || operation.status !== 'pending') return

    if (Date.now() - operation.startedAt > TIMEOUT_MS) {
      handleTimeout(opId)
      return
    }

    try {
      const response = await workspaceApi.getBillingOpStatus(opId)

      if (response.status === 'succeeded') {
        await handleSuccess(opId)
        return
      }

      if (response.status === 'failed') {
        handleFailure(opId, response.error_message ?? null)
        return
      }

      scheduleNextPoll(opId)
    } catch {
      if (Date.now() - operation.startedAt > TIMEOUT_MS) {
        handleTimeout(opId)
        return
      }
      scheduleNextPoll(opId)
    }
  }

  function scheduleNextPoll(opId: string) {
    const currentInterval = intervals.get(opId) ?? INITIAL_INTERVAL_MS
    const nextInterval = Math.min(
      currentInterval * BACKOFF_MULTIPLIER,
      MAX_INTERVAL_MS
    )
    intervals.set(opId, nextInterval)

    const timeoutId = setTimeout(() => void poll(opId), nextInterval)
    timeouts.set(opId, timeoutId)
  }

  async function handleSuccess(opId: string) {
    const operation = operations.value.get(opId)
    if (!operation) return

    updateOperationStatus(opId, 'succeeded', null)
    cleanup(opId)

    const billingContext = useBillingContext()
    await Promise.all([
      billingContext.fetchStatus(),
      billingContext.fetchBalance()
    ])

    // Close any open billing dialogs and show settings
    const dialogStore = useDialogStore()
    dialogStore.closeDialog({ key: 'subscription-required' })
    dialogStore.closeDialog({ key: 'top-up-credits' })
    useSettingsDialog().show('workspace')

    const toastStore = useToastStore()
    const messageKey =
      operation.type === 'subscription'
        ? 'billingOperation.subscriptionSuccess'
        : 'billingOperation.topupSuccess'

    toastStore.add({
      severity: 'success',
      summary: t(messageKey),
      life: 5000
    })
  }

  function handleFailure(opId: string, errorMessage: string | null) {
    const operation = operations.value.get(opId)
    if (!operation) return

    const defaultMessage =
      operation.type === 'subscription'
        ? t('billingOperation.subscriptionFailed')
        : t('billingOperation.topupFailed')

    updateOperationStatus(opId, 'failed', errorMessage ?? defaultMessage)
    cleanup(opId)

    useToastStore().add({
      severity: 'error',
      summary: defaultMessage,
      detail: errorMessage ?? undefined,
      life: 5000
    })
  }

  function handleTimeout(opId: string) {
    const operation = operations.value.get(opId)
    if (!operation) return

    const message =
      operation.type === 'subscription'
        ? t('billingOperation.subscriptionTimeout')
        : t('billingOperation.topupTimeout')

    updateOperationStatus(opId, 'timeout', message)
    cleanup(opId)

    useToastStore().add({
      severity: 'error',
      summary: message,
      life: 5000
    })
  }

  function updateOperationStatus(
    opId: string,
    status: OperationStatus,
    errorMessage: string | null
  ) {
    const operation = operations.value.get(opId)
    if (!operation) return

    const updated = { ...operation, status, errorMessage }
    operations.value = new Map(operations.value).set(opId, updated)
  }

  function cleanup(opId: string) {
    const timeoutId = timeouts.get(opId)
    if (timeoutId) {
      clearTimeout(timeoutId)
      timeouts.delete(opId)
    }
    intervals.delete(opId)

    // Remove the "received" toast
    const receivedToast = receivedToasts.get(opId)
    if (receivedToast) {
      useToastStore().remove(receivedToast)
      receivedToasts.delete(opId)
    }
  }

  function clearOperation(opId: string) {
    cleanup(opId)
    const newMap = new Map(operations.value)
    newMap.delete(opId)
    operations.value = newMap
  }

  return {
    operations,
    hasPendingOperations,
    isSettingUp,
    isAddingCredits,
    getOperation,
    startOperation,
    clearOperation
  }
})
