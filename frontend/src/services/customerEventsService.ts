import type { AxiosError, AxiosResponse } from 'axios'
import axios from 'axios'
import { ref, watch } from 'vue'

import { getComfyApiBaseUrl } from '@/config/comfyApi'
import { d } from '@/i18n'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import type { components, operations } from '@/types/comfyRegistryTypes'
import { isAbortError } from '@/utils/typeGuardUtil'

export enum EventType {
  CREDIT_ADDED = 'credit_added',
  ACCOUNT_CREATED = 'account_created',
  API_USAGE_STARTED = 'api_usage_started',
  API_USAGE_COMPLETED = 'api_usage_completed'
}

type CustomerEventsResponse =
  operations['GetCustomerEvents']['responses']['200']['content']['application/json']

type CustomerEventsResponseQuery =
  operations['GetCustomerEvents']['parameters']['query']

export type AuditLog = components['schemas']['AuditLog']

const customerApiClient = axios.create({
  baseURL: getComfyApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json'
  }
})

export const useCustomerEventsService = () => {
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  watch(
    () => getComfyApiBaseUrl(),
    (url) => {
      customerApiClient.defaults.baseURL = url
    }
  )

  const handleRequestError = (
    err: unknown,
    context: string,
    routeSpecificErrors?: Record<number, string>
  ) => {
    // Don't treat cancellation as an error
    if (isAbortError(err)) return

    let message: string
    if (!axios.isAxiosError(err)) {
      message = `${context} failed: ${err instanceof Error ? err.message : String(err)}`
    } else {
      const axiosError = err as AxiosError<{ message: string }>
      const status = axiosError.response?.status
      if (status && routeSpecificErrors?.[status]) {
        message = routeSpecificErrors[status]
      } else {
        message =
          axiosError.response?.data?.message ??
          `${context} failed with status ${status}`
      }
    }

    error.value = message
  }

  const executeRequest = async <T>(
    requestCall: () => Promise<AxiosResponse<T>>,
    options: {
      errorContext: string
      routeSpecificErrors?: Record<number, string>
    }
  ): Promise<T | null> => {
    const { errorContext, routeSpecificErrors } = options

    isLoading.value = true
    error.value = null

    try {
      const response = await requestCall()
      return response.data
    } catch (err) {
      handleRequestError(err, errorContext, routeSpecificErrors)
      return null
    } finally {
      isLoading.value = false
    }
  }

  function formatEventType(eventType: string) {
    switch (eventType) {
      case 'credit_added':
        return 'Credits Added'
      case 'account_created':
        return 'Account Created'
      case 'api_usage_completed':
        return 'API Usage'
      default:
        return eventType
    }
  }

  function formatDate(dateString: string): string {
    const date = new Date(dateString)

    return d(date, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  function formatJsonKey(key: string) {
    return key
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  function formatJsonValue(value: unknown) {
    if (typeof value === 'number') {
      // Format numbers with commas and decimals if needed
      return value.toLocaleString()
    }
    if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}/)) {
      // Format dates nicely
      return new Date(value).toLocaleString()
    }
    return value
  }

  function getEventSeverity(eventType: string) {
    switch (eventType) {
      case 'credit_added':
        return 'success'
      case 'account_created':
        return 'info'
      case 'api_usage_completed':
        return 'warning'
      default:
        return 'info'
    }
  }

  function hasAdditionalInfo(event: AuditLog) {
    const { amount, api_name, model, ...otherParams } = event.params || {}
    return Object.keys(otherParams).length > 0
  }

  function getTooltipContent(event: AuditLog) {
    const { ...params } = event.params || {}

    return Object.entries(params)
      .map(([key, value]) => {
        const formattedKey = formatJsonKey(key)
        const formattedValue = formatJsonValue(value)
        return `<strong>${formattedKey}:</strong> ${formattedValue}`
      })
      .join('<br>')
  }

  function formatAmount(amountMicros?: number) {
    if (!amountMicros) return '0.00'
    return (amountMicros / 100).toFixed(2)
  }

  async function getMyEvents({
    page = 1,
    limit = 10
  }: CustomerEventsResponseQuery = {}): Promise<CustomerEventsResponse | null> {
    const errorContext = 'Fetching customer events'
    const routeSpecificErrors = {
      400: 'Invalid input, object invalid',
      404: 'Not found'
    }

    // Get auth headers
    const authHeaders = await useFirebaseAuthStore().getAuthHeader()
    if (!authHeaders) {
      error.value = 'Authentication header is missing'
      return null
    }

    const result = await executeRequest<CustomerEventsResponse>(
      () =>
        customerApiClient.get('/customers/events', {
          params: { page, limit },
          headers: authHeaders
        }),
      { errorContext, routeSpecificErrors }
    )

    return result
  }

  return {
    // State
    isLoading,
    error,

    // Methods
    getMyEvents,
    formatEventType,
    getEventSeverity,
    formatAmount,
    hasAdditionalInfo,
    formatDate,
    formatJsonKey,
    formatJsonValue,
    getTooltipContent
  }
}
