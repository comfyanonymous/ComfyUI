import { promiseTimeout, until } from '@vueuse/core'
import axios from 'axios'
import { get } from 'es-toolkit/compat'
import { trimEnd } from 'es-toolkit'
import { ref } from 'vue'

import defaultClientFeatureFlags from '@/config/clientFeatureFlags.json' with { type: 'json' }
import { getDevOverride } from '@/utils/devFeatureFlagOverride'
import type {
  ModelFile,
  ModelFolderInfo
} from '@/platform/assets/schemas/assetSchema'
import { isCloud } from '@/platform/distribution/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import type { IFuseOptions } from 'fuse.js'
import type {
  TemplateIncludeOnDistributionEnum,
  TemplateInfo,
  WorkflowTemplates
} from '@/platform/workflow/templates/types/template'
import type {
  ComfyApiWorkflow,
  ComfyWorkflowJSON,
  NodeId
} from '@/platform/workflow/validation/schemas/workflowSchema'
import type {
  AssetDownloadWsMessage,
  AssetExportWsMessage,
  CustomNodesI18n,
  EmbeddingsResponse,
  ExecutedWsMessage,
  ExecutingWsMessage,
  ExecutionCachedWsMessage,
  ExecutionErrorWsMessage,
  ExecutionInterruptedWsMessage,
  ExecutionStartWsMessage,
  ExecutionSuccessWsMessage,
  ExtensionsResponse,
  FeatureFlagsWsMessage,
  LogsRawResponse,
  LogsWsMessage,
  NotificationWsMessage,
  PreviewMethod,
  ProgressStateWsMessage,
  ProgressTextWsMessage,
  ProgressWsMessage,
  PromptResponse,
  Settings,
  StatusWsMessage,
  StatusWsMessageStatus,
  SystemStats,
  User,
  UserDataFullInfo
} from '@/schemas/apiSchema'
import type {
  JobDetail,
  JobListItem
} from '@/platform/remote/comfyui/jobs/jobTypes'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import type { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import type { AuthHeader } from '@/types/authTypes'
import type { NodeExecutionId } from '@/types/nodeIdentification'
import {
  fetchHistory,
  fetchJobDetail,
  fetchQueue
} from '@/platform/remote/comfyui/jobs/fetchJobs'

interface QueuePromptRequestBody {
  client_id: string
  prompt: ComfyApiWorkflow
  partial_execution_targets?: NodeExecutionId[]
  extra_data: {
    extra_pnginfo: {
      workflow: ComfyWorkflowJSON
    }
    /**
     * The auth token for the comfy org account if the user is logged in.
     *
     * Backend node can access this token by specifying following input:
     * ```python
      @classmethod
      def INPUT_TYPES(s):
        return {
          "hidden": { "auth_token": "AUTH_TOKEN_COMFY_ORG"}
        }

      def execute(self, auth_token: str):
        print(f"Auth token: {auth_token}")
     * ```
     */
    auth_token_comfy_org?: string
    /**
     * The auth token for the comfy org account if the user is logged in.
     *
     * Backend node can access this token by specifying following input:
     * ```python
     * def INPUT_TYPES(s):
     *   return {
     *     "hidden": { "api_key": "API_KEY_COMFY_ORG" }
     *   }
     *
     * def execute(self, api_key: str):
     *   print(f"API Key: {api_key}")
     * ```
     */
    api_key_comfy_org?: string
    /**
     * Override the preview method for this prompt execution.
     * 'default' uses the server's CLI setting.
     */
    preview_method?: PreviewMethod
  }
  front?: boolean
  number?: number
}

/**
 * Options for queuePrompt method
 */
interface QueuePromptOptions {
  /**
   * Optional list of node execution IDs to execute (partial execution).
   * Each ID represents a node's position in nested subgraphs.
   * Format: Colon-separated path of node IDs (e.g., "123:456:789")
   */
  partialExecutionTargets?: NodeExecutionId[]
  /**
   * Override the preview method for this prompt execution.
   * 'default' uses the server's CLI setting and is not sent to backend.
   */
  previewMethod?: PreviewMethod
}

/** Dictionary of Frontend-generated API calls */
interface FrontendApiCalls {
  graphChanged: ComfyWorkflowJSON
  promptQueueing: { requestId: number; batchCount: number; number?: number }
  promptQueued: { number: number; batchCount: number; requestId?: number }
  graphCleared: never
  reconnecting: never
  reconnected: never
}

export type PromptQueueingEventPayload = FrontendApiCalls['promptQueueing']
export type PromptQueuedEventPayload = FrontendApiCalls['promptQueued']

/** Dictionary of calls originating from ComfyUI core */
interface BackendApiCalls {
  progress: ProgressWsMessage
  executing: ExecutingWsMessage
  executed: ExecutedWsMessage
  status: StatusWsMessage
  notification: NotificationWsMessage
  execution_start: ExecutionStartWsMessage
  execution_success: ExecutionSuccessWsMessage
  execution_error: ExecutionErrorWsMessage
  execution_interrupted: ExecutionInterruptedWsMessage
  execution_cached: ExecutionCachedWsMessage
  logs: LogsWsMessage
  /** Binary preview/progress data */
  b_preview: Blob
  /** Binary preview with metadata (node_id, job_id) */
  b_preview_with_metadata: {
    blob: Blob
    nodeId: string
    parentNodeId: string
    displayNodeId: string
    realNodeId: string
    jobId: string
  }
  progress_text: ProgressTextWsMessage
  progress_state: ProgressStateWsMessage
  feature_flags: FeatureFlagsWsMessage
  asset_download: AssetDownloadWsMessage
  asset_export: AssetExportWsMessage
}

/** Dictionary of all api calls */
interface ApiCalls extends BackendApiCalls, FrontendApiCalls {}

/** Used to create a discriminating union on type value. */
interface ApiMessage<T extends keyof ApiCalls> {
  type: T
  data: ApiCalls[T]
}

export class UnauthorizedError extends Error {}

/** Ensures workers get a fair shake. */
type Unionize<T> = T[keyof T]

/**
 *  Discriminated union of generic, i.e.:
 * ```ts
 * // Convert
 * type ApiMessageUnion = ApiMessage<'status' | 'executing' | ...>
 * // To
 * type ApiMessageUnion = ApiMessage<'status'> | ApiMessage<'executing'> | ...
 * ```
 */
type ApiMessageUnion = Unionize<{
  [Key in keyof ApiCalls]: ApiMessage<Key>
}>

/** Wraps all properties in {@link CustomEvent}. */
type AsCustomEvents<T> = {
  readonly [K in keyof T]: CustomEvent<T[K]>
}

/** Handles differing event and API signatures. */
type ApiToEventType<T = ApiCalls> = {
  [K in keyof T]: K extends 'status'
    ? StatusWsMessageStatus
    : K extends 'executing'
      ? NodeId
      : T[K]
}

/** Dictionary of types used in the detail for a custom event */
type ApiEventTypes = ApiToEventType<ApiCalls>

/** Dictionary of API events: `[name]: CustomEvent<Type>` */
type ApiEvents = AsCustomEvents<ApiEventTypes>

/** {@link Omit} all properties that evaluate to `never`. */
type NeverNever<T> = {
  [K in keyof T as T[K] extends never ? never : K]: T[K]
}

/** {@link Pick} only properties that evaluate to `never`. */
type PickNevers<T> = {
  [K in keyof T as T[K] extends never ? K : never]: T[K]
}

/** Keys (names) of API events that _do not_ pass a {@link CustomEvent} `detail` object. */
type SimpleApiEvents = keyof PickNevers<ApiEventTypes>
/** Keys (names) of API events that pass a {@link CustomEvent} `detail` object. */
type ComplexApiEvents = keyof NeverNever<ApiEventTypes>

export type GlobalSubgraphData = {
  name: string
  info: {
    node_pack: string
    category?: string
    search_aliases?: string[]
    requiresCustomNodes?: string[]
    includeOnDistributions?: TemplateIncludeOnDistributionEnum[]
  }
  data: string | Promise<string>
  essentials_category?: string
}

function addHeaderEntry(headers: HeadersInit, key: string, value: string) {
  if (Array.isArray(headers)) {
    headers.push([key, value])
  } else if (headers instanceof Headers) {
    headers.set(key, value)
  } else {
    headers[key] = value
  }
}

/** EventTarget typing has no generic capability. */
export interface ComfyApi extends EventTarget {
  addEventListener<TEvent extends keyof ApiEvents>(
    type: TEvent,
    callback: ((event: ApiEvents[TEvent]) => void) | null,
    options?: AddEventListenerOptions | boolean
  ): void

  removeEventListener<TEvent extends keyof ApiEvents>(
    type: TEvent,
    callback: ((event: ApiEvents[TEvent]) => void) | null,
    options?: EventListenerOptions | boolean
  ): void
}

export class PromptExecutionError extends Error {
  response: PromptResponse

  constructor(response: PromptResponse) {
    super('Prompt execution failed')
    this.response = response
  }

  override toString() {
    let message = ''
    if (typeof this.response.error === 'string') {
      message += this.response.error
    } else if (this.response.error) {
      message +=
        this.response.error.message + ': ' + this.response.error.details
    }

    for (const [_, nodeError] of Object.entries(
      this.response.node_errors ?? []
    )) {
      message += '\n' + nodeError.class_type + ':'
      for (const errorReason of nodeError.errors) {
        message += '\n    - ' + errorReason.message + ': ' + errorReason.details
      }
    }

    return message
  }
}

export class ComfyApi extends EventTarget {
  private _registered = new Set()
  api_host: string
  api_base: string
  /**
   * The client id from the initial session storage.
   */
  initialClientId: string | null
  /**
   * The current client id from websocket status updates.
   */
  clientId?: string
  /**
   * The current user id.
   */
  user: string
  socket: WebSocket | null = null

  /**
   * Cache Firebase auth store composable function.
   */
  private authStoreComposable?: typeof useFirebaseAuthStore

  reportedUnknownMessageTypes = new Set<string>()

  /**
   * Get feature flags supported by this frontend client.
   * Returns a copy to prevent external modification.
   */
  getClientFeatureFlags(): Record<string, unknown> {
    return { ...defaultClientFeatureFlags }
  }

  /**
   * Feature flags received from the backend server.
   */
  serverFeatureFlags = ref<Record<string, unknown>>({})

  /**
   * The auth token for the comfy org account if the user is logged in.
   * This is only used for {@link queuePrompt} now. It is not directly
   * passed as parameter to the function because some custom nodes are hijacking
   * {@link queuePrompt} improperly, which causes extra parameters to be lost
   * in the function call chain.
   *
   * Ref: https://cs.comfy.org/search?q=context:global+%22api.queuePrompt+%3D%22&patternType=keyword&sm=0
   *
   * TODO: Move this field to parameter of {@link queuePrompt} once all
   * custom nodes are patched.
   */
  authToken?: string
  /**
   * The API key for the comfy org account if the user logged in via API key.
   */
  apiKey?: string

  constructor() {
    super()
    this.user = ''
    this.api_host = location.host
    this.api_base = isCloud
      ? ''
      : location.pathname.split('/').slice(0, -1).join('/')
    this.initialClientId = sessionStorage.getItem('clientId')
  }

  internalURL(route: string): string {
    return this.api_base + '/internal' + route
  }

  apiURL(route: string): string {
    return this.api_base + '/api' + route
  }

  fileURL(route: string): string {
    return this.api_base + route
  }

  /**
   * Gets the Firebase auth store instance using cached composable function.
   * Caches the composable function on first call, then reuses it.
   * Returns null for non-cloud distributions.
   * @returns The Firebase auth store instance, or null if not in cloud
   */
  private async getAuthStore() {
    if (isCloud) {
      if (!this.authStoreComposable) {
        const module = await import('@/stores/firebaseAuthStore')
        this.authStoreComposable = module.useFirebaseAuthStore
      }

      return this.authStoreComposable()
    }
  }

  /**
   * Waits for Firebase auth to be initialized before proceeding.
   * Includes 10-second timeout to prevent infinite hanging.
   */
  private async waitForAuthInitialization(): Promise<void> {
    if (isCloud) {
      const authStore = await this.getAuthStore()
      if (!authStore) return

      if (authStore.isInitialized) return

      try {
        await Promise.race([
          until(authStore.isInitialized),
          promiseTimeout(10000)
        ])
      } catch {
        console.warn('Firebase auth initialization timeout after 10 seconds')
      }
    }
  }

  async fetchApi(route: string, options?: RequestInit) {
    const headers: HeadersInit = options?.headers ?? {}

    if (isCloud) {
      await this.waitForAuthInitialization()

      // Get Firebase JWT token if user is logged in
      const getAuthHeaderIfAvailable = async (): Promise<AuthHeader | null> => {
        try {
          const authStore = await this.getAuthStore()
          return authStore ? await authStore.getAuthHeader() : null
        } catch (error) {
          console.warn('Failed to get auth header:', error)
          return null
        }
      }

      const authHeader = await getAuthHeaderIfAvailable()

      if (authHeader) {
        for (const [key, value] of Object.entries(authHeader)) {
          addHeaderEntry(headers, key, value)
        }
      }
    }

    addHeaderEntry(headers, 'Comfy-User', this.user)
    return fetch(this.apiURL(route), {
      cache: 'no-cache',
      ...options,
      headers
    })
  }

  override addEventListener<TEvent extends keyof ApiEvents>(
    type: TEvent,
    callback: ((event: ApiEvents[TEvent]) => void) | null,
    options?: AddEventListenerOptions | boolean
  ) {
    // Type assertion: strictFunctionTypes.  So long as we emit events in a type-safe fashion, this is safe.
    super.addEventListener(type, callback as EventListener, options)
    this._registered.add(type)
  }

  override removeEventListener<TEvent extends keyof ApiEvents>(
    type: TEvent,
    callback: ((event: ApiEvents[TEvent]) => void) | null,
    options?: EventListenerOptions | boolean
  ): void {
    super.removeEventListener(type, callback as EventListener, options)
  }

  addCustomEventListener(
    type: string,
    callback: ((event: CustomEvent<unknown>) => void) | null,
    options?: AddEventListenerOptions | boolean
  ) {
    super.addEventListener(type, callback as EventListener, options)
    this._registered.add(type)
  }

  removeCustomEventListener(
    type: string,
    callback: ((event: CustomEvent<unknown>) => void) | null,
    options?: EventListenerOptions | boolean
  ) {
    super.removeEventListener(type, callback as EventListener, options)
  }

  /**
   * Dispatches a custom event.
   * Provides type safety for the contravariance issue with EventTarget (last checked TS 5.6).
   * @param type The type of event to emit
   * @param detail The detail property used for a custom event ({@link CustomEventInit.detail})
   */
  dispatchCustomEvent<T extends SimpleApiEvents>(type: T): boolean
  dispatchCustomEvent<T extends ComplexApiEvents>(
    type: T,
    detail: ApiEventTypes[T] | null
  ): boolean
  dispatchCustomEvent<T extends keyof ApiEventTypes>(
    type: T,
    detail?: ApiEventTypes[T]
  ): boolean {
    const event =
      detail === undefined
        ? new CustomEvent(type)
        : new CustomEvent(type, { detail })
    return super.dispatchEvent(event)
  }

  /** @deprecated Use {@link dispatchCustomEvent}. */
  override dispatchEvent(event: never): boolean {
    return super.dispatchEvent(event)
  }

  /**
   * Poll status  for colab and other things that don't support websockets.
   */
  private _pollQueue() {
    setInterval(async () => {
      try {
        const resp = await this.fetchApi('/prompt')
        const status = (await resp.json()) as StatusWsMessageStatus
        this.dispatchCustomEvent('status', status)
      } catch (error) {
        this.dispatchCustomEvent('status', null)
      }
    }, 1000)
  }

  /**
   * Creates and connects a WebSocket for realtime updates
   * @param {boolean} isReconnect If the socket is connection is a reconnect attempt
   */
  private async createSocket(isReconnect?: boolean) {
    if (this.socket) {
      return
    }

    let opened = false
    let existingSession = window.name

    // Build WebSocket URL with query parameters
    const params = new URLSearchParams()

    if (existingSession) {
      params.set('clientId', existingSession)
    }

    // Get auth token and set cloud params if available
    // Uses workspace token (if enabled) or Firebase token
    if (isCloud) {
      try {
        const authStore = await this.getAuthStore()
        const authToken = await authStore?.getAuthToken()
        if (authToken) {
          params.set('token', authToken)
        }
      } catch (error) {
        // Continue without auth token if there's an error
        console.warn(
          'Could not get auth token for WebSocket connection:',
          error
        )
      }
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const baseUrl = `${protocol}://${this.api_host}${this.api_base}/ws`
    const query = params.toString()
    const wsUrl = query ? `${baseUrl}?${query}` : baseUrl

    this.socket = new WebSocket(wsUrl)
    this.socket.binaryType = 'arraybuffer'

    this.socket.addEventListener('open', () => {
      opened = true

      // Send feature flags as the first message
      this.socket!.send(
        JSON.stringify({
          type: 'feature_flags',
          data: this.getClientFeatureFlags()
        })
      )

      if (isReconnect) {
        this.dispatchCustomEvent('reconnected')
      }
    })

    this.socket.addEventListener('error', () => {
      if (this.socket) this.socket.close()
      if (!isReconnect && !opened) {
        this._pollQueue()
      }
    })

    this.socket.addEventListener('close', () => {
      setTimeout(async () => {
        this.socket = null
        await this.createSocket(true)
      }, 300)
      if (opened) {
        this.dispatchCustomEvent('status', null)
        this.dispatchCustomEvent('reconnecting')
      }
    })

    this.socket.addEventListener('message', (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          const view = new DataView(event.data)
          const eventType = view.getUint32(0)

          let imageMime
          switch (eventType) {
            case 3:
              const decoder = new TextDecoder()
              const data = event.data.slice(4)
              const nodeIdLength = view.getUint32(4)
              this.dispatchCustomEvent('progress_text', {
                nodeId: decoder.decode(data.slice(4, 4 + nodeIdLength)),
                text: decoder.decode(data.slice(4 + nodeIdLength))
              })
              break
            case 1:
              const imageType = view.getUint32(4)
              const imageData = event.data.slice(8)
              switch (imageType) {
                case 2:
                  imageMime = 'image/png'
                  break
                case 1:
                default:
                  imageMime = 'image/jpeg'
                  break
              }
              const imageBlob = new Blob([imageData], {
                type: imageMime
              })
              this.dispatchCustomEvent('b_preview', imageBlob)
              break
            case 4:
              // PREVIEW_IMAGE_WITH_METADATA
              const decoder4 = new TextDecoder()
              const metadataLength = view.getUint32(4)
              const metadataBytes = event.data.slice(8, 8 + metadataLength)
              const metadata = JSON.parse(decoder4.decode(metadataBytes))
              const imageData4 = event.data.slice(8 + metadataLength)

              let imageMime4 = metadata.image_type

              const imageBlob4 = new Blob([imageData4], {
                type: imageMime4
              })

              // Dispatch enhanced preview event with metadata
              this.dispatchCustomEvent('b_preview_with_metadata', {
                blob: imageBlob4,
                nodeId: metadata.node_id,
                displayNodeId: metadata.display_node_id,
                parentNodeId: metadata.parent_node_id,
                realNodeId: metadata.real_node_id,
                jobId: metadata.prompt_id
              })

              // Also dispatch legacy b_preview for backward compatibility
              this.dispatchCustomEvent('b_preview', imageBlob4)
              break
            default:
              throw new Error(
                `Unknown binary websocket message of type ${eventType}`
              )
          }
        } else {
          const msg = JSON.parse(event.data) as ApiMessageUnion
          switch (msg.type) {
            case 'status':
              if (msg.data.sid) {
                const clientId = msg.data.sid
                this.clientId = clientId
                window.name = clientId // use window name so it isn't reused when duplicating tabs
                sessionStorage.setItem('clientId', clientId) // store in session storage so duplicate tab can load correct workflow
              }
              this.dispatchCustomEvent('status', msg.data.status ?? null)
              break
            case 'executing':
              this.dispatchCustomEvent(
                'executing',
                msg.data.display_node || msg.data.node
              )
              break
            case 'execution_start':
            case 'execution_error':
            case 'execution_interrupted':
            case 'execution_cached':
            case 'execution_success':
            case 'progress':
            case 'progress_state':
            case 'executed':
            case 'graphChanged':
            case 'promptQueued':
            case 'logs':
            case 'b_preview':
            case 'notification':
              this.dispatchCustomEvent(msg.type, msg.data)
              break
            case 'feature_flags':
              // Store server feature flags
              this.serverFeatureFlags.value = msg.data
              console.log(
                'Server feature flags received:',
                this.serverFeatureFlags.value
              )
              this.dispatchCustomEvent('feature_flags', msg.data)
              break
            default:
              if (this._registered.has(msg.type)) {
                // Fallback for custom types - calls super direct.
                super.dispatchEvent(
                  new CustomEvent(msg.type, { detail: msg.data })
                )
              } else if (!this.reportedUnknownMessageTypes.has(msg.type)) {
                this.reportedUnknownMessageTypes.add(msg.type)
                throw new Error(`Unknown message type ${msg.type}`)
              }
          }
        }
      } catch (error) {
        console.warn('Unhandled message:', event.data, error)
      }
    })
  }

  /**
   * Initialises sockets and realtime updates
   */
  init() {
    this.createSocket()
  }

  /**
   * Gets a list of extension urls
   */
  async getExtensions(): Promise<ExtensionsResponse> {
    const resp = await this.fetchApi('/extensions', { cache: 'no-store' })
    return await resp.json()
  }

  /**
   * Gets the available workflow templates from custom nodes.
   * @returns A map of custom_node names and associated template workflow names.
   */
  async getWorkflowTemplates(): Promise<{
    [customNodesName: string]: string[]
  }> {
    const res = await this.fetchApi('/workflow_templates')
    return await res.json()
  }

  /**
   * Gets the index of core workflow templates.
   * @param locale Optional locale code (e.g., 'en', 'fr', 'zh') to load localized templates
   */
  async getCoreWorkflowTemplates(
    locale?: string
  ): Promise<WorkflowTemplates[]> {
    const fileName =
      locale && locale !== 'en' ? `index.${locale}.json` : 'index.json'
    try {
      const res = await axios.get(this.fileURL(`/templates/${fileName}`))
      const contentType = res.headers['content-type']
      return contentType?.includes('application/json') ? res.data : []
    } catch (error) {
      // Fallback to default English version if localized version doesn't exist
      if (locale && locale !== 'en') {
        console.warn(
          `Localized templates for '${locale}' not found, falling back to English`
        )
        return this.getCoreWorkflowTemplates()
      }
      console.error('Error loading core workflow templates:', error)
      return []
    }
  }

  /**
   * Gets a list of embedding names
   */
  async getEmbeddings(): Promise<EmbeddingsResponse> {
    const resp = await this.fetchApi('/embeddings', { cache: 'no-store' })
    return await resp.json()
  }

  /**
   * Loads node object definitions for the graph
   * @returns The node definitions
   */
  async getNodeDefs(): Promise<Record<string, ComfyNodeDef>> {
    const resp = await this.fetchApi('/object_info', { cache: 'no-store' })
    return await resp.json()
  }

  /**
   * Queues a prompt to be executed
   * @param {number} number The index at which to queue the prompt, passing -1 will insert the prompt at the front of the queue
   * @param {object} data The prompt data to queue
   * @param {QueuePromptOptions} options Optional execution options
   * @throws {PromptExecutionError} If the prompt fails to execute
   */
  async queuePrompt(
    number: number,
    data: { output: ComfyApiWorkflow; workflow: ComfyWorkflowJSON },
    options?: QueuePromptOptions
  ): Promise<PromptResponse> {
    const { output: prompt, workflow } = data

    const body: QueuePromptRequestBody = {
      client_id: this.clientId ?? '', // TODO: Unify clientId access
      prompt,
      ...(options?.partialExecutionTargets && {
        partial_execution_targets: options.partialExecutionTargets
      }),
      extra_data: {
        auth_token_comfy_org: this.authToken,
        api_key_comfy_org: this.apiKey,
        extra_pnginfo: { workflow },
        ...(options?.previewMethod &&
          options.previewMethod !== 'default' && {
            preview_method: options.previewMethod
          })
      }
    }

    if (number === -1) {
      body.front = true
    } else if (number != 0) {
      body.number = number
    }

    const res = await this.fetchApi('/prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (res.status !== 200) {
      const text = await res.text()
      let errorResponse
      try {
        errorResponse = JSON.parse(text)
      } catch {
        errorResponse = {
          error: {
            type: 'server_error',
            message: `${res.status} ${res.statusText}`,
            details: text
          }
        }
      }
      throw new PromptExecutionError(errorResponse)
    }

    return await res.json()
  }

  /**
   * Gets a list of model folder keys (eg ['checkpoints', 'loras', ...])
   * @returns The list of model folder keys
   */
  async getModelFolders(): Promise<ModelFolderInfo[]> {
    const res = await this.fetchApi(`/experiment/models`)
    if (res.status === 404) {
      return []
    }
    const folderBlacklist = ['configs', 'custom_nodes']
    return (await res.json()).filter(
      (folder: ModelFolderInfo) => !folderBlacklist.includes(folder.name)
    )
  }

  /**
   * Gets a list of models in the specified folder
   * @param {string} folder The folder to list models from, such as 'checkpoints'
   * @returns The list of model filenames within the specified folder
   */
  async getModels(folder: string): Promise<ModelFile[]> {
    const res = await this.fetchApi(`/experiment/models/${folder}`)
    if (res.status === 404) {
      return []
    }
    return await res.json()
  }

  /**
   * Gets the metadata for a model
   * @param {string} folder The folder containing the model
   * @param {string} model The model to get metadata for
   * @returns The metadata for the model
   */
  async viewMetadata(folder: string, model: string) {
    const res = await this.fetchApi(
      `/view_metadata/${folder}?filename=${encodeURIComponent(model)}`
    )
    const rawResponse = await res.text()
    if (!rawResponse) {
      return null
    }
    try {
      return JSON.parse(rawResponse)
    } catch (error) {
      console.error(
        'Error viewing metadata',
        res.status,
        res.statusText,
        rawResponse,
        error
      )
      return null
    }
  }

  /**
   * Loads a list of items (queue or history)
   * @param {string} type The type of items to load, queue or history
   * @returns The items of the specified type grouped by their status
   */
  async getItems(type: 'queue' | 'history') {
    if (type === 'queue') {
      return this.getQueue()
    }
    return this.getHistory()
  }

  /**
   * Gets the current state of the queue
   * @returns The currently running and queued items
   */
  async getQueue(): Promise<{
    Running: JobListItem[]
    Pending: JobListItem[]
  }> {
    try {
      return await fetchQueue(this.fetchApi.bind(this))
    } catch (error) {
      console.error('Failed to fetch queue:', error)
      return { Running: [], Pending: [] }
    }
  }

  /**
   * Gets the prompt execution history
   * @returns Prompt history including node outputs
   */
  async getHistory(
    max_items: number = 200,
    options?: { offset?: number }
  ): Promise<JobListItem[]> {
    try {
      return await fetchHistory(
        this.fetchApi.bind(this),
        max_items,
        options?.offset
      )
    } catch (error) {
      console.error(error)
      return []
    }
  }

  /**
   * Gets detailed job info including outputs and workflow
   * @param jobId The job ID
   * @returns Full job details or undefined if not found
   */
  async getJobDetail(jobId: string): Promise<JobDetail | undefined> {
    return fetchJobDetail(this.fetchApi.bind(this), jobId)
  }

  /**
   * Gets system & device stats
   * @returns System stats such as python version, OS, per device info
   */
  async getSystemStats(): Promise<SystemStats> {
    const res = await this.fetchApi('/system_stats')
    return await res.json()
  }

  /**
   * Sends a POST request to the API
   * @param {*} type The endpoint to post to
   * @param {*} body Optional POST data
   */
  private async _postItem(type: string, body?: Record<string, unknown>) {
    try {
      await this.fetchApi('/' + type, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: body ? JSON.stringify(body) : undefined
      })
    } catch (error) {
      console.error(error)
    }
  }

  /**
   * Deletes an item from the specified list
   * @param {string} type The type of item to delete, queue or history
   * @param {number} id The id of the item to delete
   */
  async deleteItem(type: string, id: string) {
    await this._postItem(type, { delete: [id] })
  }

  /**
   * Clears the specified list
   * @param {string} type The type of list to clear, queue or history
   */
  async clearItems(type: string) {
    await this._postItem(type, { clear: true })
  }

  /**
   * Interrupts the execution of the running job. If runningJobId is provided,
   * it is included in the payload as a helpful hint to the backend.
   * @param {string | null} [runningJobId] Optional Running Job ID to interrupt
   */
  async interrupt(runningJobId: string | null) {
    await this._postItem(
      'interrupt',
      runningJobId ? { prompt_id: runningJobId } : undefined
    )
  }

  /**
   * Gets user configuration data and where data should be stored
   */
  async getUserConfig(): Promise<User> {
    return (await this.fetchApi('/users')).json()
  }

  /**
   * Creates a new user
   * @param { string } username
   * @returns The fetch response
   */
  createUser(username: string) {
    return this.fetchApi('/users', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username })
    })
  }

  /**
   * Gets all setting values for the current user
   * @returns { Promise<string, unknown> } A dictionary of id -> value
   */
  async getSettings(): Promise<Settings> {
    const resp = await this.fetchApi('/settings')

    if (resp.status == 401) {
      throw new UnauthorizedError(resp.statusText)
    }
    return await resp.json()
  }

  /**
   * Gets a setting for the current user
   * @param { string } id The id of the setting to fetch
   * @returns { Promise<unknown> } The setting value
   */
  async getSetting(id: keyof Settings): Promise<Settings[keyof Settings]> {
    return (await this.fetchApi(`/settings/${encodeURIComponent(id)}`)).json()
  }

  /**
   * Stores a dictionary of settings for the current user
   */
  async storeSettings(settings: Partial<Settings>) {
    return this.fetchApi(`/settings`, {
      method: 'POST',
      body: JSON.stringify(settings)
    })
  }

  /**
   * Stores a setting for the current user
   */
  async storeSetting(id: keyof Settings, value: Settings[keyof Settings]) {
    return this.fetchApi(`/settings/${encodeURIComponent(id)}`, {
      method: 'POST',
      body: JSON.stringify(value)
    })
  }

  /**
   * Gets a user data file for the current user
   */
  async getUserData(file: string, options?: RequestInit) {
    return this.fetchApi(`/userdata/${encodeURIComponent(file)}`, options)
  }

  /**
   * Stores a user data file for the current user
   * @param { string } file The name of the userdata file to save
   * @param { unknown } data The data to save to the file
   * @param { RequestInit & { stringify?: boolean, throwOnError?: boolean } } [options]
   * @returns { Promise<Response> }
   */
  async storeUserData(
    file: string,
    data: unknown,
    options: RequestInit & {
      overwrite?: boolean
      stringify?: boolean
      throwOnError?: boolean
      full_info?: boolean
    } = {
      overwrite: true,
      stringify: true,
      throwOnError: true,
      full_info: false
    }
  ): Promise<Response> {
    const resp = await this.fetchApi(
      `/userdata/${encodeURIComponent(file)}?overwrite=${options.overwrite}&full_info=${options.full_info}`,
      {
        method: 'POST',
        body: options?.stringify ? JSON.stringify(data) : (data as BodyInit),
        ...options
      }
    )
    if (resp.status !== 200 && options.throwOnError !== false) {
      throw new Error(
        `Error storing user data file '${file}': ${resp.status} ${(await resp).statusText}`
      )
    }

    return resp
  }

  /**
   * Deletes a user data file for the current user
   * @param { string } file The name of the userdata file to delete
   */
  async deleteUserData(file: string) {
    const resp = await this.fetchApi(`/userdata/${encodeURIComponent(file)}`, {
      method: 'DELETE'
    })
    return resp
  }

  /**
   * Move a user data file for the current user
   * @param { string } source The userdata file to move
   * @param { string } dest The destination for the file
   */
  async moveUserData(
    source: string,
    dest: string,
    options = { overwrite: false }
  ) {
    const resp = await this.fetchApi(
      `/userdata/${encodeURIComponent(source)}/move/${encodeURIComponent(dest)}?overwrite=${options?.overwrite}`,
      {
        method: 'POST'
      }
    )
    return resp
  }

  async listUserDataFullInfo(dir: string): Promise<UserDataFullInfo[]> {
    const trimmedDir = trimEnd(dir, '/')
    const resp = await this.fetchApi(
      `/userdata?dir=${encodeURIComponent(trimmedDir)}&recurse=true&split=false&full_info=true`
    )
    if (resp.status === 404) return []
    if (resp.status !== 200) {
      throw new Error(
        `Error getting user data list '${trimmedDir}': ${resp.status} ${resp.statusText}`
      )
    }
    return resp.json()
  }

  async getGlobalSubgraphData(id: string): Promise<string> {
    const resp = await api.fetchApi('/global_subgraphs/' + id)
    if (resp.status !== 200) {
      throw new Error(
        `Failed to fetch global subgraph '${id}': ${resp.status} ${resp.statusText}`
      )
    }
    const subgraph: GlobalSubgraphData = await resp.json()
    if (!subgraph?.data) {
      throw new Error(`Global subgraph '${id}' returned empty data`)
    }
    return subgraph.data as string
  }
  async getGlobalSubgraphs(): Promise<Record<string, GlobalSubgraphData>> {
    const resp = await api.fetchApi('/global_subgraphs')
    if (resp.status !== 200) return {}
    const subgraphs: Record<string, GlobalSubgraphData> = await resp.json()
    for (const [k, v] of Object.entries(subgraphs)) {
      if (!v.data) v.data = this.getGlobalSubgraphData(k)
    }
    return subgraphs
  }

  async getLogs(): Promise<string> {
    const url = isCloud ? this.apiURL('/logs') : this.internalURL('/logs')
    return (await axios.get(url)).data
  }

  async getRawLogs(): Promise<LogsRawResponse> {
    const url = isCloud
      ? this.apiURL('/logs/raw')
      : this.internalURL('/logs/raw')
    return (await axios.get(url)).data
  }

  async subscribeLogs(enabled: boolean): Promise<void> {
    const url = isCloud
      ? this.apiURL('/logs/subscribe')
      : this.internalURL('/logs/subscribe')
    return await axios.patch(url, {
      enabled,
      clientId: this.clientId
    })
  }

  async getFolderPaths(): Promise<Record<string, string[]>> {
    const response = await axios
      .get(this.internalURL('/folder_paths'))
      .catch(() => null)
    if (!response) {
      return {} // Fallback: no filesystem paths known when API unavailable
    }
    return response.data
  }

  /* Frees memory by unloading models and optionally freeing execution cache
   * @param {Object} options - The options object
   * @param {boolean} options.freeExecutionCache - If true, also frees execution cache
   */
  async freeMemory(options: { freeExecutionCache: boolean }) {
    try {
      let mode = ''
      if (options.freeExecutionCache) {
        mode = '{"unload_models": true, "free_memory": true}'
      } else {
        mode = '{"unload_models": true}'
      }

      const res = await this.fetchApi(`/free`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: mode
      })

      if (res.status === 200) {
        if (options.freeExecutionCache) {
          useToastStore().add({
            severity: 'success',
            summary: 'Models and Execution Cache have been cleared.',
            life: 3000
          })
        } else {
          useToastStore().add({
            severity: 'success',
            summary: 'Models have been unloaded.',
            life: 3000
          })
        }
      } else {
        useToastStore().add({
          severity: 'error',
          summary:
            'Unloading of models failed. Installed ComfyUI may be an outdated version.',
          life: 5000
        })
      }
    } catch (error) {
      useToastStore().add({
        severity: 'error',
        summary: 'An error occurred while trying to unload models.',
        life: 5000
      })
    }
  }

  /**
   * Gets the custom nodes i18n data from the server.
   *
   * @returns The custom nodes i18n data
   */
  async getCustomNodesI18n(): Promise<CustomNodesI18n> {
    return (await axios.get(this.apiURL('/i18n'))).data
  }

  /**
   * Checks if the server supports a specific feature.
   * @param featureName The name of the feature to check (supports dot notation for nested values)
   * @returns true if the feature is supported, false otherwise
   */
  serverSupportsFeature(featureName: string): boolean {
    const override = getDevOverride<boolean>(featureName)
    if (override !== undefined) return override
    return get(this.serverFeatureFlags.value, featureName) === true
  }

  /**
   * Gets a server feature flag value.
   * @param featureName The name of the feature to get (supports dot notation for nested values)
   * @param defaultValue The default value if the feature is not found
   * @returns The feature value or default
   */
  getServerFeature<T = unknown>(featureName: string, defaultValue?: T): T {
    const override = getDevOverride<T>(featureName)
    if (override !== undefined) return override
    return get(this.serverFeatureFlags.value, featureName, defaultValue) as T
  }

  /**
   * Gets all server feature flags.
   * @returns Copy of all server feature flags
   */
  getServerFeatures(): Record<string, unknown> {
    return { ...this.serverFeatureFlags.value }
  }

  async getFuseOptions(): Promise<IFuseOptions<TemplateInfo> | null> {
    try {
      const res = await axios.get(
        this.fileURL('/templates/fuse_options.json'),
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )
      const contentType = res.headers['content-type']
      return contentType?.includes('application/json') ? res.data : null
    } catch (error) {
      console.error('Error loading fuse options:', error)
      return null
    }
  }
}

export const api = new ComfyApi()
