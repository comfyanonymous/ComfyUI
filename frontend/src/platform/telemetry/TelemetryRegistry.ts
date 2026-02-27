import type { AuditLog } from '@/services/customerEventsService'

import type {
  AuthMetadata,
  BeginCheckoutMetadata,
  EnterLinearMetadata,
  ExecutionErrorMetadata,
  ExecutionSuccessMetadata,
  ExecutionTriggerSource,
  HelpCenterClosedMetadata,
  HelpCenterOpenedMetadata,
  HelpResourceClickedMetadata,
  NodeSearchMetadata,
  NodeSearchResultMetadata,
  PageViewMetadata,
  PageVisibilityMetadata,
  SettingChangedMetadata,
  SubscriptionMetadata,
  SurveyResponses,
  TabCountMetadata,
  TelemetryDispatcher,
  TelemetryProvider,
  TemplateFilterMetadata,
  TemplateLibraryClosedMetadata,
  TemplateLibraryMetadata,
  TemplateMetadata,
  UiButtonClickMetadata,
  WorkflowCreatedMetadata,
  WorkflowImportMetadata
} from './types'

/**
 * Registry that holds multiple telemetry providers and dispatches
 * all tracking calls to each registered provider.
 *
 * Implements TelemetryDispatcher (all methods required) while dispatching
 * to TelemetryProvider instances using optional chaining since providers
 * only implement the methods they care about.
 */
export class TelemetryRegistry implements TelemetryDispatcher {
  private providers: TelemetryProvider[] = []

  registerProvider(provider: TelemetryProvider): void {
    this.providers.push(provider)
  }

  private dispatch(action: (provider: TelemetryProvider) => void): void {
    this.providers.forEach((provider) => {
      try {
        action(provider)
      } catch (error) {
        console.error('[Telemetry] Provider dispatch failed', error)
      }
    })
  }

  trackSignupOpened(): void {
    this.dispatch((provider) => provider.trackSignupOpened?.())
  }

  trackAuth(metadata: AuthMetadata): void {
    this.dispatch((provider) => provider.trackAuth?.(metadata))
  }

  trackUserLoggedIn(): void {
    this.dispatch((provider) => provider.trackUserLoggedIn?.())
  }

  trackSubscription(
    event: 'modal_opened' | 'subscribe_clicked',
    metadata?: SubscriptionMetadata
  ): void {
    this.dispatch((provider) => provider.trackSubscription?.(event, metadata))
  }

  trackBeginCheckout(metadata: BeginCheckoutMetadata): void {
    this.dispatch((provider) => provider.trackBeginCheckout?.(metadata))
  }

  trackMonthlySubscriptionSucceeded(): void {
    this.dispatch((provider) => provider.trackMonthlySubscriptionSucceeded?.())
  }

  trackMonthlySubscriptionCancelled(): void {
    this.dispatch((provider) => provider.trackMonthlySubscriptionCancelled?.())
  }

  trackAddApiCreditButtonClicked(): void {
    this.dispatch((provider) => provider.trackAddApiCreditButtonClicked?.())
  }

  trackApiCreditTopupButtonPurchaseClicked(amount: number): void {
    this.dispatch((provider) =>
      provider.trackApiCreditTopupButtonPurchaseClicked?.(amount)
    )
  }

  trackApiCreditTopupSucceeded(): void {
    this.dispatch((provider) => provider.trackApiCreditTopupSucceeded?.())
  }

  trackRunButton(options?: {
    subscribe_to_run?: boolean
    trigger_source?: ExecutionTriggerSource
  }): void {
    this.dispatch((provider) => provider.trackRunButton?.(options))
  }

  startTopupTracking(): void {
    this.dispatch((provider) => provider.startTopupTracking?.())
  }

  checkForCompletedTopup(events: AuditLog[] | undefined | null): boolean {
    return this.providers.some((provider) => {
      try {
        return provider.checkForCompletedTopup?.(events) ?? false
      } catch (error) {
        console.error('[Telemetry] Provider dispatch failed', error)
        return false
      }
    })
  }

  clearTopupTracking(): void {
    this.dispatch((provider) => provider.clearTopupTracking?.())
  }

  trackSurvey(
    stage: 'opened' | 'submitted',
    responses?: SurveyResponses
  ): void {
    this.dispatch((provider) => provider.trackSurvey?.(stage, responses))
  }

  trackEmailVerification(stage: 'opened' | 'requested' | 'completed'): void {
    this.dispatch((provider) => provider.trackEmailVerification?.(stage))
  }

  trackTemplate(metadata: TemplateMetadata): void {
    this.dispatch((provider) => provider.trackTemplate?.(metadata))
  }

  trackTemplateLibraryOpened(metadata: TemplateLibraryMetadata): void {
    this.dispatch((provider) => provider.trackTemplateLibraryOpened?.(metadata))
  }

  trackTemplateLibraryClosed(metadata: TemplateLibraryClosedMetadata): void {
    this.dispatch((provider) => provider.trackTemplateLibraryClosed?.(metadata))
  }

  trackWorkflowImported(metadata: WorkflowImportMetadata): void {
    this.dispatch((provider) => provider.trackWorkflowImported?.(metadata))
  }

  trackWorkflowOpened(metadata: WorkflowImportMetadata): void {
    this.dispatch((provider) => provider.trackWorkflowOpened?.(metadata))
  }

  trackEnterLinear(metadata: EnterLinearMetadata): void {
    this.dispatch((provider) => provider.trackEnterLinear?.(metadata))
  }

  trackPageVisibilityChanged(metadata: PageVisibilityMetadata): void {
    this.dispatch((provider) => provider.trackPageVisibilityChanged?.(metadata))
  }

  trackTabCount(metadata: TabCountMetadata): void {
    this.dispatch((provider) => provider.trackTabCount?.(metadata))
  }

  trackNodeSearch(metadata: NodeSearchMetadata): void {
    this.dispatch((provider) => provider.trackNodeSearch?.(metadata))
  }

  trackNodeSearchResultSelected(metadata: NodeSearchResultMetadata): void {
    this.dispatch((provider) =>
      provider.trackNodeSearchResultSelected?.(metadata)
    )
  }

  trackTemplateFilterChanged(metadata: TemplateFilterMetadata): void {
    this.dispatch((provider) => provider.trackTemplateFilterChanged?.(metadata))
  }

  trackHelpCenterOpened(metadata: HelpCenterOpenedMetadata): void {
    this.dispatch((provider) => provider.trackHelpCenterOpened?.(metadata))
  }

  trackHelpResourceClicked(metadata: HelpResourceClickedMetadata): void {
    this.dispatch((provider) => provider.trackHelpResourceClicked?.(metadata))
  }

  trackHelpCenterClosed(metadata: HelpCenterClosedMetadata): void {
    this.dispatch((provider) => provider.trackHelpCenterClosed?.(metadata))
  }

  trackWorkflowCreated(metadata: WorkflowCreatedMetadata): void {
    this.dispatch((provider) => provider.trackWorkflowCreated?.(metadata))
  }

  trackWorkflowExecution(): void {
    this.dispatch((provider) => provider.trackWorkflowExecution?.())
  }

  trackExecutionError(metadata: ExecutionErrorMetadata): void {
    this.dispatch((provider) => provider.trackExecutionError?.(metadata))
  }

  trackExecutionSuccess(metadata: ExecutionSuccessMetadata): void {
    this.dispatch((provider) => provider.trackExecutionSuccess?.(metadata))
  }

  trackSettingChanged(metadata: SettingChangedMetadata): void {
    this.dispatch((provider) => provider.trackSettingChanged?.(metadata))
  }

  trackUiButtonClicked(metadata: UiButtonClickMetadata): void {
    this.dispatch((provider) => provider.trackUiButtonClicked?.(metadata))
  }

  trackPageView(pageName: string, properties?: PageViewMetadata): void {
    this.dispatch((provider) => provider.trackPageView?.(pageName, properties))
  }
}
