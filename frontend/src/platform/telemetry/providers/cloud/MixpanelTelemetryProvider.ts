import type { OverridedMixpanel } from 'mixpanel-browser'
import { watch } from 'vue'

import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import {
  TOOLKIT_BLUEPRINT_MODULES,
  TOOLKIT_NODE_NAMES
} from '@/constants/toolkitNodes'
import {
  checkForCompletedTopup as checkTopupUtil,
  clearTopupTracking as clearTopupUtil,
  startTopupTracking as startTopupUtil
} from '@/platform/telemetry/topupTracker'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { AuditLog } from '@/services/customerEventsService'
import { useWorkflowTemplatesStore } from '@/platform/workflow/templates/repositories/workflowTemplatesStore'
import { app } from '@/scripts/app'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { NodeSourceType } from '@/types/nodeSource'
import { reduceAllNodes } from '@/utils/graphTraversalUtil'

import type {
  AuthMetadata,
  CreditTopupMetadata,
  EnterLinearMetadata,
  ExecutionContext,
  ExecutionTriggerSource,
  ExecutionErrorMetadata,
  ExecutionSuccessMetadata,
  HelpCenterClosedMetadata,
  HelpCenterOpenedMetadata,
  HelpResourceClickedMetadata,
  NodeSearchMetadata,
  NodeSearchResultMetadata,
  PageVisibilityMetadata,
  RunButtonProperties,
  SettingChangedMetadata,
  SubscriptionMetadata,
  SurveyResponses,
  TabCountMetadata,
  TelemetryEventName,
  TelemetryEventProperties,
  TelemetryProvider,
  TemplateFilterMetadata,
  TemplateLibraryClosedMetadata,
  TemplateLibraryMetadata,
  TemplateMetadata,
  UiButtonClickMetadata,
  WorkflowCreatedMetadata,
  WorkflowImportMetadata
} from '../../types'
import { remoteConfig } from '@/platform/remoteConfig/remoteConfig'
import type { RemoteConfig } from '@/platform/remoteConfig/types'
import { TelemetryEvents } from '../../types'
import { normalizeSurveyResponses } from '../../utils/surveyNormalization'

const DEFAULT_DISABLED_EVENTS = [
  TelemetryEvents.WORKFLOW_OPENED,
  TelemetryEvents.PAGE_VISIBILITY_CHANGED,
  TelemetryEvents.TAB_COUNT_TRACKING,
  TelemetryEvents.NODE_SEARCH,
  TelemetryEvents.NODE_SEARCH_RESULT_SELECTED,
  TelemetryEvents.TEMPLATE_FILTER_CHANGED,
  TelemetryEvents.SETTING_CHANGED,
  TelemetryEvents.HELP_CENTER_OPENED,
  TelemetryEvents.HELP_RESOURCE_CLICKED,
  TelemetryEvents.HELP_CENTER_CLOSED,
  TelemetryEvents.WORKFLOW_CREATED,
  TelemetryEvents.UI_BUTTON_CLICKED
] as const satisfies TelemetryEventName[]

const TELEMETRY_EVENT_SET = new Set<TelemetryEventName>(
  Object.values(TelemetryEvents) as TelemetryEventName[]
)

interface QueuedEvent {
  eventName: TelemetryEventName
  properties?: TelemetryEventProperties
}

/**
 * Mixpanel Telemetry Provider - Cloud Build Implementation
 *
 * CRITICAL: OSS Build Safety
 * This provider integrates with Mixpanel for cloud telemetry tracking.
 * Entire file is tree-shaken away in OSS builds (DISTRIBUTION unset).
 *
 * To verify OSS builds exclude this code:
 * 1. `DISTRIBUTION= pnpm build` (OSS build)
 * 2. `grep -RinE --include='*.js' 'trackWorkflow|trackEvent|mixpanel' dist/` (should find nothing)
 * 3. Check dist/assets/*.js files contain no tracking code
 */
export class MixpanelTelemetryProvider implements TelemetryProvider {
  private isEnabled = true
  private mixpanel: OverridedMixpanel | null = null
  private eventQueue: QueuedEvent[] = []
  private isInitialized = false
  private lastTriggerSource: ExecutionTriggerSource | undefined
  private disabledEvents = new Set<TelemetryEventName>(DEFAULT_DISABLED_EVENTS)

  constructor() {
    this.configureDisabledEvents(
      (window.__CONFIG__ as Partial<RemoteConfig> | undefined) ?? null
    )
    watch(
      remoteConfig,
      (config) => {
        this.configureDisabledEvents(config)
      },
      { immediate: true }
    )
    const token = window.__CONFIG__?.mixpanel_token

    if (token) {
      try {
        // Dynamic import to avoid bundling mixpanel in OSS builds
        void import('mixpanel-browser')
          .then((mixpanelModule) => {
            this.mixpanel = mixpanelModule.default
            this.mixpanel.init(token, {
              debug: import.meta.env.DEV,
              track_pageview: true,
              api_host: 'https://mp.comfy.org',
              cross_subdomain_cookie: true,
              persistence: 'cookie',
              loaded: () => {
                this.isInitialized = true
                this.flushEventQueue() // flush events that were queued while initializing
                useCurrentUser().onUserResolved((user) => {
                  if (this.mixpanel && user.id) {
                    this.mixpanel.identify(user.id)
                  }
                })
              }
            })
          })
          .catch((error) => {
            console.error('Failed to load Mixpanel:', error)
            this.isEnabled = false
          })
      } catch (error) {
        console.error('Failed to initialize Mixpanel:', error)
        this.isEnabled = false
      }
    } else {
      console.warn('Mixpanel token not provided in runtime config')
      this.isEnabled = false
    }
  }

  private flushEventQueue(): void {
    if (!this.isInitialized || !this.mixpanel) {
      return
    }

    while (this.eventQueue.length > 0) {
      const event = this.eventQueue.shift()!
      try {
        this.mixpanel.track(event.eventName, event.properties || {})
      } catch (error) {
        console.error('Failed to track queued event:', error)
      }
    }
  }

  private trackEvent(
    eventName: TelemetryEventName,
    properties?: TelemetryEventProperties
  ): void {
    if (!this.isEnabled) {
      return
    }

    if (this.disabledEvents.has(eventName)) {
      return
    }

    const event: QueuedEvent = { eventName, properties }

    if (this.isInitialized && this.mixpanel) {
      // Mixpanel is ready, track immediately
      try {
        this.mixpanel.track(eventName, properties || {})
      } catch (error) {
        console.error('Failed to track event:', error)
      }
    } else {
      // Mixpanel not ready yet, queue the event
      this.eventQueue.push(event)
    }
  }

  private configureDisabledEvents(config: Partial<RemoteConfig> | null): void {
    const disabledSource =
      config?.telemetry_disabled_events ?? DEFAULT_DISABLED_EVENTS

    this.disabledEvents = this.buildEventSet(disabledSource)
  }

  private buildEventSet(values: TelemetryEventName[]): Set<TelemetryEventName> {
    return new Set(
      values.filter((value) => {
        const isValid = TELEMETRY_EVENT_SET.has(value)
        if (!isValid && import.meta.env.DEV) {
          console.warn(
            `Unknown telemetry event name in disabled list: ${value}`
          )
        }
        return isValid
      })
    )
  }

  trackSignupOpened(): void {
    this.trackEvent(TelemetryEvents.USER_SIGN_UP_OPENED)
  }

  trackAuth(metadata: AuthMetadata): void {
    this.trackEvent(TelemetryEvents.USER_AUTH_COMPLETED, metadata)
  }

  trackUserLoggedIn(): void {
    this.trackEvent(TelemetryEvents.USER_LOGGED_IN)
  }

  trackSubscription(
    event: 'modal_opened' | 'subscribe_clicked',
    metadata?: SubscriptionMetadata
  ): void {
    const eventName =
      event === 'modal_opened'
        ? TelemetryEvents.SUBSCRIPTION_REQUIRED_MODAL_OPENED
        : TelemetryEvents.SUBSCRIBE_NOW_BUTTON_CLICKED

    this.trackEvent(eventName, metadata)
  }

  trackAddApiCreditButtonClicked(): void {
    this.trackEvent(TelemetryEvents.ADD_API_CREDIT_BUTTON_CLICKED)
  }

  trackMonthlySubscriptionSucceeded(): void {
    this.trackEvent(TelemetryEvents.MONTHLY_SUBSCRIPTION_SUCCEEDED)
  }

  /**
   * Track when a user completes a subscription cancellation flow.
   * Fired after we detect the backend reports `is_active: false` and the UI stops polling.
   */
  trackMonthlySubscriptionCancelled(): void {
    this.trackEvent(TelemetryEvents.MONTHLY_SUBSCRIPTION_CANCELLED)
  }

  trackApiCreditTopupButtonPurchaseClicked(amount: number): void {
    const metadata: CreditTopupMetadata = {
      credit_amount: amount
    }
    this.trackEvent(
      TelemetryEvents.API_CREDIT_TOPUP_BUTTON_PURCHASE_CLICKED,
      metadata
    )
  }

  trackApiCreditTopupSucceeded(): void {
    this.trackEvent(TelemetryEvents.API_CREDIT_TOPUP_SUCCEEDED)
  }

  // Credit top-up tracking methods (composition with utility functions)
  startTopupTracking(): void {
    startTopupUtil()
  }

  checkForCompletedTopup(events: AuditLog[] | undefined | null): boolean {
    return checkTopupUtil(events)
  }

  clearTopupTracking(): void {
    clearTopupUtil()
  }

  trackRunButton(options?: {
    subscribe_to_run?: boolean
    trigger_source?: ExecutionTriggerSource
  }): void {
    const executionContext = this.getExecutionContext()

    const runButtonProperties: RunButtonProperties = {
      subscribe_to_run: options?.subscribe_to_run || false,
      workflow_type: executionContext.is_template ? 'template' : 'custom',
      workflow_name: executionContext.workflow_name ?? 'untitled',
      custom_node_count: executionContext.custom_node_count,
      total_node_count: executionContext.total_node_count,
      subgraph_count: executionContext.subgraph_count,
      has_api_nodes: executionContext.has_api_nodes,
      api_node_names: executionContext.api_node_names,
      has_toolkit_nodes: executionContext.has_toolkit_nodes,
      toolkit_node_names: executionContext.toolkit_node_names,
      trigger_source: options?.trigger_source
    }

    this.lastTriggerSource = options?.trigger_source
    this.trackEvent(TelemetryEvents.RUN_BUTTON_CLICKED, runButtonProperties)
  }

  trackSurvey(
    stage: 'opened' | 'submitted',
    responses?: SurveyResponses
  ): void {
    const eventName =
      stage === 'opened'
        ? TelemetryEvents.USER_SURVEY_OPENED
        : TelemetryEvents.USER_SURVEY_SUBMITTED

    // Apply normalization to survey responses
    const normalizedResponses = responses
      ? normalizeSurveyResponses(responses)
      : undefined

    this.trackEvent(eventName, normalizedResponses)

    // If this is a survey submission, also set user properties with normalized data
    if (stage === 'submitted' && normalizedResponses && this.mixpanel) {
      try {
        this.mixpanel.people.set(normalizedResponses)
      } catch (error) {
        console.error('Failed to set survey user properties:', error)
      }
    }
  }

  trackEmailVerification(stage: 'opened' | 'requested' | 'completed'): void {
    let eventName: TelemetryEventName

    switch (stage) {
      case 'opened':
        eventName = TelemetryEvents.USER_EMAIL_VERIFY_OPENED
        break
      case 'requested':
        eventName = TelemetryEvents.USER_EMAIL_VERIFY_REQUESTED
        break
      case 'completed':
        eventName = TelemetryEvents.USER_EMAIL_VERIFY_COMPLETED
        break
    }

    this.trackEvent(eventName)
  }

  trackTemplate(metadata: TemplateMetadata): void {
    this.trackEvent(TelemetryEvents.TEMPLATE_WORKFLOW_OPENED, metadata)
  }

  trackTemplateLibraryOpened(metadata: TemplateLibraryMetadata): void {
    this.trackEvent(TelemetryEvents.TEMPLATE_LIBRARY_OPENED, metadata)
  }

  trackTemplateLibraryClosed(metadata: TemplateLibraryClosedMetadata): void {
    this.trackEvent(TelemetryEvents.TEMPLATE_LIBRARY_CLOSED, metadata)
  }

  trackWorkflowImported(metadata: WorkflowImportMetadata): void {
    this.trackEvent(TelemetryEvents.WORKFLOW_IMPORTED, metadata)
  }

  trackWorkflowOpened(metadata: WorkflowImportMetadata): void {
    this.trackEvent(TelemetryEvents.WORKFLOW_OPENED, metadata)
  }

  trackEnterLinear(metadata: EnterLinearMetadata): void {
    this.trackEvent(TelemetryEvents.ENTER_LINEAR_MODE, metadata)
  }

  trackPageVisibilityChanged(metadata: PageVisibilityMetadata): void {
    this.trackEvent(TelemetryEvents.PAGE_VISIBILITY_CHANGED, metadata)
  }

  trackTabCount(metadata: TabCountMetadata): void {
    this.trackEvent(TelemetryEvents.TAB_COUNT_TRACKING, metadata)
  }

  trackNodeSearch(metadata: NodeSearchMetadata): void {
    this.trackEvent(TelemetryEvents.NODE_SEARCH, metadata)
  }

  trackNodeSearchResultSelected(metadata: NodeSearchResultMetadata): void {
    this.trackEvent(TelemetryEvents.NODE_SEARCH_RESULT_SELECTED, metadata)
  }

  trackTemplateFilterChanged(metadata: TemplateFilterMetadata): void {
    this.trackEvent(TelemetryEvents.TEMPLATE_FILTER_CHANGED, metadata)
  }

  trackHelpCenterOpened(metadata: HelpCenterOpenedMetadata): void {
    this.trackEvent(TelemetryEvents.HELP_CENTER_OPENED, metadata)
  }

  trackHelpResourceClicked(metadata: HelpResourceClickedMetadata): void {
    this.trackEvent(TelemetryEvents.HELP_RESOURCE_CLICKED, metadata)
  }

  trackHelpCenterClosed(metadata: HelpCenterClosedMetadata): void {
    this.trackEvent(TelemetryEvents.HELP_CENTER_CLOSED, metadata)
  }

  trackWorkflowCreated(metadata: WorkflowCreatedMetadata): void {
    this.trackEvent(TelemetryEvents.WORKFLOW_CREATED, metadata)
  }

  trackWorkflowExecution(): void {
    const context = this.getExecutionContext()
    const eventContext: ExecutionContext = {
      ...context,
      trigger_source: this.lastTriggerSource ?? 'unknown'
    }
    this.trackEvent(TelemetryEvents.EXECUTION_START, eventContext)
    this.lastTriggerSource = undefined
  }

  trackExecutionError(metadata: ExecutionErrorMetadata): void {
    this.trackEvent(TelemetryEvents.EXECUTION_ERROR, metadata)
  }

  trackExecutionSuccess(metadata: ExecutionSuccessMetadata): void {
    this.trackEvent(TelemetryEvents.EXECUTION_SUCCESS, metadata)
  }

  trackSettingChanged(metadata: SettingChangedMetadata): void {
    this.trackEvent(TelemetryEvents.SETTING_CHANGED, metadata)
  }

  trackUiButtonClicked(metadata: UiButtonClickMetadata): void {
    this.trackEvent(TelemetryEvents.UI_BUTTON_CLICKED, metadata)
  }

  getExecutionContext(): ExecutionContext {
    const workflowStore = useWorkflowStore()
    const templatesStore = useWorkflowTemplatesStore()
    const nodeDefStore = useNodeDefStore()
    const activeWorkflow = workflowStore.activeWorkflow

    // Calculate node metrics in a single traversal
    type NodeMetrics = {
      custom_node_count: number
      api_node_count: number
      toolkit_node_count: number
      subgraph_count: number
      total_node_count: number
      has_api_nodes: boolean
      api_node_names: string[]
      has_toolkit_nodes: boolean
      toolkit_node_names: string[]
    }

    const nodeCounts = reduceAllNodes<NodeMetrics>(
      app.rootGraph,
      (metrics, node) => {
        const nodeDef = nodeDefStore.nodeDefsByName[node.type]
        const isCustomNode =
          nodeDef?.nodeSource?.type === NodeSourceType.CustomNodes
        const isApiNode = nodeDef?.api_node === true
        const isSubgraph = node.isSubgraphNode?.() === true

        if (isApiNode) {
          metrics.has_api_nodes = true
          const canonicalName = nodeDef?.name
          if (
            canonicalName &&
            !metrics.api_node_names.includes(canonicalName)
          ) {
            metrics.api_node_names.push(canonicalName)
          }
        }

        const isToolkitNode =
          TOOLKIT_NODE_NAMES.has(node.type) ||
          (nodeDef?.python_module !== undefined &&
            TOOLKIT_BLUEPRINT_MODULES.has(nodeDef.python_module))
        if (isToolkitNode) {
          metrics.has_toolkit_nodes = true
          const trackingName = nodeDef?.name ?? node.type
          if (!metrics.toolkit_node_names.includes(trackingName)) {
            metrics.toolkit_node_names.push(trackingName)
          }
        }

        metrics.custom_node_count += isCustomNode ? 1 : 0
        metrics.api_node_count += isApiNode ? 1 : 0
        metrics.toolkit_node_count += isToolkitNode ? 1 : 0
        metrics.subgraph_count += isSubgraph ? 1 : 0
        metrics.total_node_count += 1

        return metrics
      },
      {
        custom_node_count: 0,
        api_node_count: 0,
        toolkit_node_count: 0,
        subgraph_count: 0,
        total_node_count: 0,
        has_api_nodes: false,
        api_node_names: [],
        has_toolkit_nodes: false,
        toolkit_node_names: []
      }
    )

    if (activeWorkflow?.filename) {
      const isTemplate = templatesStore.knownTemplateNames.has(
        activeWorkflow.filename
      )

      if (isTemplate) {
        const template = templatesStore.getTemplateByName(
          activeWorkflow.filename
        )

        const englishMetadata = templatesStore.getEnglishMetadata(
          activeWorkflow.filename
        )

        return {
          is_template: true,
          workflow_name: activeWorkflow.filename,
          template_source: template?.sourceModule,
          template_category: englishMetadata?.category ?? template?.category,
          template_tags: englishMetadata?.tags ?? template?.tags,
          template_models: englishMetadata?.models ?? template?.models,
          template_use_case: englishMetadata?.useCase ?? template?.useCase,
          template_license: englishMetadata?.license ?? template?.license,
          ...nodeCounts
        }
      }

      return {
        is_template: false,
        workflow_name: activeWorkflow.filename,
        ...nodeCounts
      }
    }

    return {
      is_template: false,
      workflow_name: undefined,
      ...nodeCounts
    }
  }
}
