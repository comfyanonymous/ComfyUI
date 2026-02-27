import type { AxiosError, AxiosResponse } from 'axios'
import axios from 'axios'
import { ref, watch } from 'vue'

import { getComfyApiBaseUrl } from '@/config/comfyApi'
import type { components, operations } from '@/types/comfyRegistryTypes'
import { isAbortError } from '@/utils/typeGuardUtil'

// Use generated types from OpenAPI spec
export type ReleaseNote = components['schemas']['ReleaseNote']
type GetReleasesParams = operations['getReleaseNotes']['parameters']['query']

// Use generated error response type
type ErrorResponse = components['schemas']['ErrorResponse']

const releaseApiClient = axios.create({
  baseURL: getComfyApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json'
  }
})

// Release service for fetching release notes
export const useReleaseService = () => {
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  watch(
    () => getComfyApiBaseUrl(),
    (url) => {
      releaseApiClient.defaults.baseURL = url
    }
  )

  // No transformation needed - API response matches the generated type

  // Handle API errors with context
  const handleApiError = (
    err: unknown,
    context: string,
    routeSpecificErrors?: Record<number, string>
  ): string => {
    if (!axios.isAxiosError(err))
      return err instanceof Error
        ? `${context}: ${err.message}`
        : `${context}: Unknown error occurred`

    const axiosError = err as AxiosError<ErrorResponse>

    if (axiosError.response) {
      const { status, data } = axiosError.response

      if (routeSpecificErrors && routeSpecificErrors[status])
        return routeSpecificErrors[status]

      switch (status) {
        case 400:
          return `Bad request: ${data?.message || 'Invalid input'}`
        case 401:
          return 'Unauthorized: Authentication required'
        case 403:
          return `Forbidden: ${data?.message || 'Access denied'}`
        case 404:
          return `Not found: ${data?.message || 'Resource not found'}`
        case 500:
          return `Server error: ${data?.message || 'Internal server error'}`
        default:
          return `${context}: ${data?.message || axiosError.message}`
      }
    }

    return `${context}: ${axiosError.message}`
  }

  // Execute API request with error handling
  const executeApiRequest = async <T>(
    apiCall: () => Promise<AxiosResponse<T>>,
    errorContext: string,
    routeSpecificErrors?: Record<number, string>
  ): Promise<T | null> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiCall()
      return response.data
    } catch (err) {
      // Don't treat cancellations as errors
      if (isAbortError(err)) return null

      error.value = handleApiError(err, errorContext, routeSpecificErrors)
      return null
    } finally {
      isLoading.value = false
    }
  }

  // Fetch release notes from API
  const getReleases = async (
    params: GetReleasesParams,
    signal?: AbortSignal
  ): Promise<ReleaseNote[] | null> => {
    const endpoint = '/releases'
    const errorContext = 'Failed to get releases'
    const routeSpecificErrors = {
      400: 'Invalid project or version parameter'
    }

    const apiResponse = await executeApiRequest(
      () =>
        releaseApiClient.get<ReleaseNote[]>(endpoint, {
          params,
          signal
        }),
      errorContext,
      routeSpecificErrors
    )

    return apiResponse
  }

  return {
    isLoading,
    error,
    getReleases
  }
}
