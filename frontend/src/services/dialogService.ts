import { merge } from 'es-toolkit/compat'
import type { Component } from 'vue'

import ConfirmationDialogContent from '@/components/dialog/content/ConfirmationDialogContent.vue'
import ErrorDialogContent from '@/components/dialog/content/ErrorDialogContent.vue'
import PromptDialogContent from '@/components/dialog/content/PromptDialogContent.vue'
import TopUpCreditsDialogContentLegacy from '@/components/dialog/content/TopUpCreditsDialogContentLegacy.vue'
import TopUpCreditsDialogContentWorkspace from '@/platform/workspace/components/TopUpCreditsDialogContentWorkspace.vue'
import { t } from '@/i18n'
import { useTelemetry } from '@/platform/telemetry'
import { isCloud } from '@/platform/distribution/types'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useDialogStore } from '@/stores/dialogStore'
import type {
  DialogComponentProps,
  ShowDialogOptions
} from '@/stores/dialogStore'

import type { ComponentAttrs } from 'vue-component-type-helpers'
import type { SubscriptionDialogReason } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'

// Lazy loaders for dialogs - components are loaded on first use
const lazyApiNodesSignInContent = () =>
  import('@/components/dialog/content/ApiNodesSignInContent.vue')
const lazySignInContent = () =>
  import('@/components/dialog/content/SignInContent.vue')
const lazyUpdatePasswordContent = () =>
  import('@/components/dialog/content/UpdatePasswordContent.vue')
const lazyComfyOrgHeader = () =>
  import('@/components/dialog/header/ComfyOrgHeader.vue')

export type ConfirmationDialogType =
  | 'default'
  | 'overwrite'
  | 'overwriteBlueprint'
  | 'delete'
  | 'dirtyClose'
  | 'reinstall'
  | 'info'

/**
 * Minimal interface for execution error dialogs.
 * Satisfied by both ExecutionErrorWsMessage (WebSocket) and ExecutionError (Jobs API).
 */
export interface ExecutionErrorDialogInput {
  exception_type: string
  exception_message: string
  node_id: string | number
  node_type: string
  traceback: string[]
}

export const useDialogService = () => {
  const dialogStore = useDialogStore()

  function showExecutionErrorDialog(executionError: ExecutionErrorDialogInput) {
    const props: ComponentAttrs<typeof ErrorDialogContent> = {
      error: {
        exceptionType: executionError.exception_type,
        exceptionMessage: executionError.exception_message,
        nodeId: executionError.node_id?.toString(),
        nodeType: executionError.node_type,
        traceback: executionError.traceback.join('\n'),
        reportType: 'graphExecutionError'
      }
    }

    dialogStore.showDialog({
      key: 'global-execution-error',
      component: ErrorDialogContent,
      props,
      dialogComponentProps: {
        onClose: () => {
          useTelemetry()?.trackUiButtonClicked({
            button_id: 'error_dialog_closed'
          })
        }
      }
    })
  }

  function parseError(error: Error) {
    const filename =
      'fileName' in error
        ? (error.fileName as string)
        : error.stack?.match(/(\/extensions\/.*\.js)/)?.[1]

    const extensionFile = filename
      ? filename.substring(filename.indexOf('/extensions/'))
      : undefined

    return {
      errorMessage: error.toString(),
      stackTrace: error.stack,
      extensionFile
    }
  }

  /**
   * Show a error dialog to the user when an error occurs.
   * @param error The error to show
   * @param options The options for the dialog
   */
  function showErrorDialog(
    error: unknown,
    options: {
      title?: string
      reportType?: string
    } = {}
  ) {
    const errorProps: {
      errorMessage: string
      stackTrace?: string
      extensionFile?: string
    } =
      error instanceof Error
        ? parseError(error)
        : {
            errorMessage: String(error)
          }

    const props: ComponentAttrs<typeof ErrorDialogContent> = {
      error: {
        exceptionType: options.title ?? 'Unknown Error',
        exceptionMessage: errorProps.errorMessage,
        traceback: errorProps.stackTrace ?? t('errorDialog.noStackTrace'),
        reportType: options.reportType
      }
    }

    dialogStore.showDialog({
      key: 'global-error',
      component: ErrorDialogContent,
      props,
      dialogComponentProps: {
        onClose: () => {
          useTelemetry()?.trackUiButtonClicked({
            button_id: 'error_dialog_closed'
          })
        }
      }
    })
  }

  /**
   * Shows a dialog requiring sign in for API nodes
   * @returns Promise that resolves to true if user clicks login, false if cancelled
   */
  async function showApiNodesSignInDialog(
    apiNodeNames: string[]
  ): Promise<boolean> {
    const [{ default: ApiNodesSignInContent }, { default: ComfyOrgHeader }] =
      await Promise.all([lazyApiNodesSignInContent(), lazyComfyOrgHeader()])

    return new Promise<boolean>((resolve) => {
      dialogStore.showDialog({
        key: 'api-nodes-signin',
        component: ApiNodesSignInContent,
        props: {
          apiNodeNames,
          onLogin: () => showSignInDialog().then((result) => resolve(result)),
          onCancel: () => resolve(false)
        },
        headerComponent: ComfyOrgHeader,
        dialogComponentProps: {
          closable: false,
          onClose: () => resolve(false)
        }
      })
    }).then((result) => {
      dialogStore.closeDialog({ key: 'api-nodes-signin' })
      return result
    })
  }

  async function showSignInDialog(): Promise<boolean> {
    const [{ default: SignInContent }, { default: ComfyOrgHeader }] =
      await Promise.all([lazySignInContent(), lazyComfyOrgHeader()])

    return new Promise<boolean>((resolve) => {
      dialogStore.showDialog({
        key: 'global-signin',
        component: SignInContent,
        headerComponent: ComfyOrgHeader,
        props: {
          onSuccess: () => resolve(true)
        },
        dialogComponentProps: {
          closable: true,
          onClose: () => resolve(false)
        }
      })
    }).then((result) => {
      dialogStore.closeDialog({ key: 'global-signin' })
      return result
    })
  }

  async function prompt({
    title,
    message,
    defaultValue = '',
    placeholder
  }: {
    title: string
    message: string
    defaultValue?: string
    placeholder?: string
  }): Promise<string | null> {
    return new Promise((resolve) => {
      dialogStore.showDialog({
        key: 'global-prompt',
        title,
        component: PromptDialogContent,
        props: {
          message,
          defaultValue,
          onConfirm: (value: string) => {
            resolve(value)
          },
          placeholder
        },
        dialogComponentProps: {
          onClose: () => {
            resolve(null)
          }
        }
      })
    })
  }

  /**
   * @returns `true` if the user confirms the dialog,
   * `false` if denied (e.g. no in yes/no/cancel), or
   * `null` if the dialog is cancelled or closed
   */
  async function confirm({
    title,
    message,
    type = 'default',
    itemList = [],
    hint
  }: {
    /** Dialog heading */
    title: string
    /** The main message body */
    message: string
    /** Pre-configured dialog type */
    type?: ConfirmationDialogType
    /** Displayed as an unordered list immediately below the message body */
    itemList?: string[]
    hint?: string
  }): Promise<boolean | null> {
    return new Promise((resolve) => {
      const options: ShowDialogOptions = {
        key: 'global-prompt',
        title,
        component: ConfirmationDialogContent,
        props: {
          message,
          type,
          itemList,
          onConfirm: resolve,
          hint
        },
        dialogComponentProps: {
          onClose: () => resolve(null)
        }
      }

      dialogStore.showDialog(options)
    })
  }

  async function showTopUpCreditsDialog(options?: {
    isInsufficientCredits?: boolean
  }) {
    const { isActiveSubscription, isFreeTier, type } = useBillingContext()
    if (!isActiveSubscription.value || isFreeTier.value) {
      await showSubscriptionRequiredDialog({
        reason: options?.isInsufficientCredits
          ? 'out_of_credits'
          : 'top_up_blocked'
      })
      return
    }

    const component =
      type.value === 'workspace'
        ? TopUpCreditsDialogContentWorkspace
        : TopUpCreditsDialogContentLegacy

    return dialogStore.showDialog({
      key: 'top-up-credits',
      component,
      props: options,
      dialogComponentProps: {
        headless: true,
        pt: {
          header: { class: 'p-0! hidden' },
          content: { class: 'p-0! m-0! rounded-2xl' },
          root: { class: 'rounded-2xl' }
        }
      }
    })
  }

  /**
   * Shows a dialog for updating the current user's password.
   */
  async function showUpdatePasswordDialog() {
    const [{ default: UpdatePasswordContent }, { default: ComfyOrgHeader }] =
      await Promise.all([lazyUpdatePasswordContent(), lazyComfyOrgHeader()])

    return dialogStore.showDialog({
      key: 'global-update-password',
      component: UpdatePasswordContent,
      headerComponent: ComfyOrgHeader,
      props: {
        onSuccess: () =>
          dialogStore.closeDialog({ key: 'global-update-password' })
      }
    })
  }

  /**
   * Shows a dialog from a third party extension.
   * @param options - The dialog options.
   * @param options.key - The dialog key.
   * @param options.title - The dialog title.
   * @param options.headerComponent - The dialog header component.
   * @param options.footerComponent - The dialog footer component.
   * @param options.component - The dialog component.
   * @param options.props - The dialog props.
   * @returns The dialog instance and a function to close the dialog.
   */
  function showExtensionDialog(options: ShowDialogOptions & { key: string }) {
    return {
      dialog: dialogStore.showExtensionDialog(options),
      closeDialog: () => dialogStore.closeDialog({ key: options.key })
    }
  }

  function showLayoutDialog<C extends Component>(options: {
    key: string
    component: C
    props: ComponentAttrs<C>
    dialogComponentProps?: DialogComponentProps
  }) {
    const layoutDefaultProps: DialogComponentProps = {
      headless: true,
      modal: true,
      closable: true,
      pt: {
        root: {
          class: 'rounded-2xl overflow-hidden'
        },
        header: {
          class: 'p-0! hidden'
        },
        content: {
          class: 'p-0! m-0!'
        }
      }
    }

    return dialogStore.showDialog({
      ...options,
      dialogComponentProps: merge(
        layoutDefaultProps,
        options.dialogComponentProps || {}
      )
    })
  }

  function showSmallLayoutDialog(
    options: Omit<ShowDialogOptions, 'dialogComponentProps'> & {
      dialogComponentProps?: Omit<DialogComponentProps, 'pt'>
    }
  ) {
    const { dialogComponentProps: callerProps, ...rest } = options

    return dialogStore.showDialog({
      ...rest,
      dialogComponentProps: {
        closable: true,
        pt: {
          root: { class: 'bg-base-background border-border-default' },
          header: { class: '!p-0 !m-0' },
          content: { class: '!p-0 overflow-y-hidden' },
          footer: { class: '!p-0' },
          pcCloseButton: {
            root: {
              class: '!w-7 !h-7 !border-none !outline-none !p-2 !m-1.5'
            }
          }
        },
        ...callerProps
      }
    })
  }

  async function showSubscriptionRequiredDialog(options?: {
    reason?: SubscriptionDialogReason
  }) {
    if (!isCloud || !window.__CONFIG__?.subscription_required) {
      return
    }

    const { useSubscriptionDialog } =
      await import('@/platform/cloud/subscription/composables/useSubscriptionDialog')
    const { show } = useSubscriptionDialog()
    show(options)
  }

  // Workspace dialogs - dynamically imported to avoid bundling when feature flag is off
  const workspaceDialogPt = {
    headless: true,
    pt: {
      header: { class: 'p-0! hidden' },
      content: { class: 'p-0! m-0! rounded-2xl' },
      root: { class: 'rounded-2xl' }
    }
  } as const

  async function showDeleteWorkspaceDialog(options?: {
    workspaceId?: string
    workspaceName?: string
  }) {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/DeleteWorkspaceDialogContent.vue')
    return dialogStore.showDialog({
      key: 'delete-workspace',
      component,
      props: options,
      dialogComponentProps: workspaceDialogPt
    })
  }

  async function showCreateWorkspaceDialog(
    onConfirm?: (name: string) => void | Promise<void>
  ) {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/CreateWorkspaceDialogContent.vue')
    return dialogStore.showDialog({
      key: 'create-workspace',
      component,
      props: { onConfirm },
      dialogComponentProps: {
        ...workspaceDialogPt
      }
    })
  }

  async function showLeaveWorkspaceDialog() {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/LeaveWorkspaceDialogContent.vue')
    return dialogStore.showDialog({
      key: 'leave-workspace',
      component,
      dialogComponentProps: workspaceDialogPt
    })
  }

  async function showEditWorkspaceDialog() {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/EditWorkspaceDialogContent.vue')
    return dialogStore.showDialog({
      key: 'edit-workspace',
      component,
      dialogComponentProps: {
        ...workspaceDialogPt
      }
    })
  }

  async function showRemoveMemberDialog(memberId: string) {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/RemoveMemberDialogContent.vue')
    return dialogStore.showDialog({
      key: 'remove-member',
      component,
      props: { memberId },
      dialogComponentProps: workspaceDialogPt
    })
  }

  async function showInviteMemberDialog() {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/InviteMemberDialogContent.vue')
    return dialogStore.showDialog({
      key: 'invite-member',
      component,
      dialogComponentProps: {
        ...workspaceDialogPt
      }
    })
  }

  async function showInviteMemberUpsellDialog() {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/InviteMemberUpsellDialogContent.vue')
    return dialogStore.showDialog({
      key: 'invite-member-upsell',
      component,
      dialogComponentProps: {
        ...workspaceDialogPt
      }
    })
  }

  async function showRevokeInviteDialog(inviteId: string) {
    const { default: component } =
      await import('@/platform/workspace/components/dialogs/RevokeInviteDialogContent.vue')
    return dialogStore.showDialog({
      key: 'revoke-invite',
      component,
      props: { inviteId },
      dialogComponentProps: workspaceDialogPt
    })
  }

  function showBillingComingSoonDialog() {
    return dialogStore.showDialog({
      key: 'billing-coming-soon',
      title: t('subscription.billingComingSoon.title'),
      component: ConfirmationDialogContent,
      props: {
        message: t('subscription.billingComingSoon.message'),
        type: 'info' as ConfirmationDialogType,
        onConfirm: () => {}
      },
      dialogComponentProps: {
        pt: {
          root: { class: 'max-w-[360px]' }
        }
      }
    })
  }

  async function showCancelSubscriptionDialog(cancelAt?: string) {
    const { default: component } =
      await import('@/components/dialog/content/subscription/CancelSubscriptionDialogContent.vue')
    return dialogStore.showDialog({
      key: 'cancel-subscription',
      component,
      props: { cancelAt },
      dialogComponentProps: {
        ...workspaceDialogPt
      }
    })
  }

  return {
    showExecutionErrorDialog,
    showApiNodesSignInDialog,
    showSignInDialog,
    showSubscriptionRequiredDialog,
    showTopUpCreditsDialog,
    showUpdatePasswordDialog,
    showExtensionDialog,
    prompt,
    showErrorDialog,
    confirm,
    showLayoutDialog,
    showSmallLayoutDialog,
    showDeleteWorkspaceDialog,
    showCreateWorkspaceDialog,
    showLeaveWorkspaceDialog,
    showEditWorkspaceDialog,
    showRemoveMemberDialog,
    showRevokeInviteDialog,
    showInviteMemberDialog,
    showInviteMemberUpsellDialog,
    showBillingComingSoonDialog,
    showCancelSubscriptionDialog
  }
}
