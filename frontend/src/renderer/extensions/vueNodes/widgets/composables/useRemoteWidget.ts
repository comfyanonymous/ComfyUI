import axios from 'axios'

import { useChainCallback } from '@/composables/functional/useChainCallback'
import type { IWidget, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { isCloud } from '@/platform/distribution/types'
import type { RemoteWidgetConfig } from '@/schemas/nodeDefSchema'
import { api } from '@/scripts/api'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

const MAX_RETRIES = 5
const TIMEOUT = 4096

interface CacheEntry<T> {
  data: T
  timestamp?: number
  error?: Error | null
  fetchPromise?: Promise<T>
  controller?: AbortController
  lastErrorTime?: number
  retryCount?: number
  failed?: boolean
}

async function getAuthHeaders() {
  if (isCloud) {
    const authStore = useFirebaseAuthStore()
    const authHeader = await authStore.getAuthHeader()
    return {
      ...(authHeader && { headers: authHeader })
    }
  }
  return {}
}

const dataCache = new Map<string, CacheEntry<unknown>>()

const createCacheKey = (config: RemoteWidgetConfig): string => {
  const { route, query_params = {}, refresh = 0 } = config

  const paramsKey = Object.entries(query_params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join('&')

  return [route, `r=${refresh}`, paramsKey].join(';')
}

const getBackoff = (retryCount: number) =>
  Math.min(1000 * Math.pow(2, retryCount), 512)

const isInitialized = (entry: CacheEntry<unknown> | undefined) =>
  entry?.data !== undefined &&
  entry?.timestamp !== undefined &&
  entry.timestamp > 0

const isStale = (entry: CacheEntry<unknown> | undefined, ttl: number) =>
  entry?.timestamp && Date.now() - entry.timestamp >= ttl

const isFetching = (entry: CacheEntry<unknown> | undefined) =>
  entry?.fetchPromise !== undefined

const isFailed = (entry: CacheEntry<unknown> | undefined) =>
  entry?.failed === true

const isBackingOff = (entry: CacheEntry<unknown> | undefined) =>
  entry?.error &&
  entry?.lastErrorTime &&
  Date.now() - entry.lastErrorTime < getBackoff(entry.retryCount || 0)

const fetchData = async (
  config: RemoteWidgetConfig,
  controller: AbortController
) => {
  const { route, response_key, query_params, timeout = TIMEOUT } = config

  const authHeaders = await getAuthHeaders()

  const res = await axios.get(route, {
    params: query_params,
    signal: controller.signal,
    timeout,
    ...authHeaders
  })

  return response_key ? res.data[response_key] : res.data
}

export function useRemoteWidget<
  T extends string | number | boolean | object
>(options: {
  remoteConfig: RemoteWidgetConfig
  defaultValue: T
  node: LGraphNode
  widget: IWidget
}) {
  const { remoteConfig, defaultValue, node, widget } = options
  const { refresh = 0, max_retries = MAX_RETRIES } = remoteConfig
  const isPermanent = refresh <= 0
  const cacheKey = createCacheKey(remoteConfig)
  let isLoaded = false
  let refreshQueued = false

  const setSuccess = (entry: CacheEntry<T>, data: T) => {
    entry.retryCount = 0
    entry.lastErrorTime = 0
    entry.error = null
    entry.timestamp = Date.now()
    entry.data = data ?? defaultValue
  }

  const setError = (entry: CacheEntry<T>, error: Error | unknown) => {
    entry.retryCount = (entry.retryCount || 0) + 1
    entry.lastErrorTime = Date.now()
    entry.error = error instanceof Error ? error : new Error(String(error))
    entry.data ??= defaultValue
    entry.fetchPromise = undefined
    if (entry.retryCount >= max_retries) {
      setFailed(entry)
    }
  }

  const setFailed = (entry: CacheEntry<T>) => {
    dataCache.set(cacheKey, {
      data: entry.data ?? defaultValue,
      failed: true
    })
  }

  const isFirstLoad = () => {
    return !isLoaded && isInitialized(dataCache.get(cacheKey))
  }

  const onFirstLoad = (data: T | T[]) => {
    isLoaded = true
    const nextValue =
      Array.isArray(data) && data.length > 0 ? data[0] : undefined
    widget.value = nextValue ?? (Array.isArray(data) ? defaultValue : data)
    widget.callback?.(widget.value)
    node.graph?.setDirtyCanvas(true)
  }

  const fetchValue = async () => {
    const entry = dataCache.get(cacheKey)

    if (isFailed(entry)) return entry!.data as T

    const isValid =
      isInitialized(entry) && (isPermanent || !isStale(entry, refresh))
    if (isValid || isBackingOff(entry) || isFetching(entry))
      return entry!.data as T

    const currentEntry: CacheEntry<T> = (entry as
      | CacheEntry<T>
      | undefined) || { data: defaultValue }
    dataCache.set(cacheKey, currentEntry)

    try {
      currentEntry.controller = new AbortController()
      currentEntry.fetchPromise = fetchData(
        remoteConfig,
        currentEntry.controller
      )
      const data = await currentEntry.fetchPromise

      setSuccess(currentEntry, data)
      return currentEntry.data
    } catch (err) {
      setError(currentEntry, err)
      return currentEntry.data
    } finally {
      currentEntry.fetchPromise = undefined
      currentEntry.controller = undefined
    }
  }

  const onRefresh = () => {
    if (remoteConfig.control_after_refresh) {
      const data = getCachedValue()
      if (!Array.isArray(data)) return // control_after_refresh is only supported for array values

      switch (remoteConfig.control_after_refresh) {
        case 'first':
          widget.value = data[0] ?? defaultValue
          break
        case 'last':
          widget.value = data.at(-1) ?? defaultValue
          break
      }
      widget.callback?.(widget.value)
      node.graph?.setDirtyCanvas(true)
    }
  }

  /**
   * Clear the widget's cached value, forcing a refresh on next access (e.g., a new render)
   */
  const clearCachedValue = () => {
    const entry = dataCache.get(cacheKey)
    if (!entry) return
    if (entry.fetchPromise) entry.controller?.abort() // Abort in-flight request
    dataCache.delete(cacheKey)
  }

  /**
   * Get the cached value of the widget without starting a new fetch.
   * @returns the most recently computed value of the widget.
   */
  function getCachedValue() {
    return dataCache.get(cacheKey)?.data as T
  }

  /**
   * Getter of the remote property of the widget (e.g., options.values, value, etc.).
   * Starts the fetch process then returns the cached value immediately.
   * @returns the most recent value of the widget.
   */
  function getValue(onFulfilled?: () => void) {
    void fetchValue()
      .then((data) => {
        if (isFirstLoad()) onFirstLoad(data)
        if (refreshQueued && data !== defaultValue) {
          onRefresh()
          refreshQueued = false
        }
        onFulfilled?.()
      })
      .catch((err) => {
        console.error(err)
      })
    return getCachedValue() ?? defaultValue
  }

  /**
   * Force the widget to refresh its value
   */
  widget.refresh = function () {
    refreshQueued = true
    clearCachedValue()
    getValue()
  }

  /**
   * Add a refresh button to the node that, when clicked, will force the widget to refresh
   */
  function addRefreshButton() {
    node.addWidget('button', 'refresh', 'refresh', widget.refresh)
  }

  /**
   * Add auto-refresh toggle widget and execution success listener
   */
  function addAutoRefreshToggle() {
    let autoRefreshEnabled = false

    // Handler for execution success
    const handleExecutionSuccess = () => {
      if (autoRefreshEnabled && widget.refresh) {
        widget.refresh()
      }
    }

    // Add toggle widget
    const autoRefreshWidget = node.addWidget(
      'toggle',
      'Auto-refresh after generation',
      false,
      (value: boolean) => {
        autoRefreshEnabled = value
      },
      {
        serialize: false
      }
    )

    // Register event listener
    api.addEventListener('execution_success', handleExecutionSuccess)

    // Cleanup on node removal
    node.onRemoved = useChainCallback(node.onRemoved, function () {
      api.removeEventListener('execution_success', handleExecutionSuccess)
    })

    return autoRefreshWidget
  }

  // Always add auto-refresh toggle for remote widgets
  addAutoRefreshToggle()

  return {
    getCachedValue,
    getValue,
    refreshValue: widget.refresh,
    addRefreshButton,
    getCacheEntry: () => dataCache.get(cacheKey),

    cacheKey
  }
}
