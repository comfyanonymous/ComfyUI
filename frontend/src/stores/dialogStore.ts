// We should consider moving to https://primevue.org/dynamicdialog/ once everything is in Vue.
// Currently we need to bridge between legacy app code and Vue app with a Pinia store.
import { merge } from 'es-toolkit/compat'
import { defineStore } from 'pinia'
import type { DialogPassThroughOptions } from 'primevue/dialog'
import { markRaw, ref } from 'vue'
import type { Component } from 'vue'

import type GlobalDialog from '@/components/dialog/GlobalDialog.vue'
import type { ComponentAttrs } from 'vue-component-type-helpers'

type DialogPosition =
  | 'center'
  | 'top'
  | 'bottom'
  | 'left'
  | 'right'
  | 'topleft'
  | 'topright'
  | 'bottomleft'
  | 'bottomright'

interface CustomDialogComponentProps {
  maximizable?: boolean
  maximized?: boolean
  onClose?: () => void
  closable?: boolean
  modal?: boolean
  position?: DialogPosition
  pt?: DialogPassThroughOptions
  closeOnEscape?: boolean
  dismissableMask?: boolean
  unstyled?: boolean
  headless?: boolean
}

export type DialogComponentProps = ComponentAttrs<typeof GlobalDialog> &
  CustomDialogComponentProps

export interface DialogInstance<
  H extends Component = Component,
  B extends Component = Component,
  F extends Component = Component
> {
  key: string
  visible: boolean
  title?: string
  headerComponent?: H
  headerProps?: ComponentAttrs<H>
  component: B
  contentProps: ComponentAttrs<B>
  footerComponent?: F
  footerProps?: ComponentAttrs<F>
  dialogComponentProps: DialogComponentProps
  priority: number
}

export interface ShowDialogOptions<
  H extends Component = Component,
  B extends Component = Component,
  F extends Component = Component
> {
  key?: string
  title?: string
  headerComponent?: H
  footerComponent?: F
  component: B
  props?: ComponentAttrs<B>
  headerProps?: ComponentAttrs<H>
  footerProps?: ComponentAttrs<F>
  dialogComponentProps?: DialogComponentProps
  /**
   * Optional priority for dialog stacking.
   * A dialog will never be shown above a dialog with a higher priority.
   * @default 1
   */
  priority?: number
}

export const useDialogStore = defineStore('dialog', () => {
  const dialogStack = ref<DialogInstance[]>([])

  /**
   * The key of the currently active (top-most) dialog.
   * Only the active dialog can be closed with the ESC key.
   */
  const activeKey = ref<string | null>(null)

  const genDialogKey = () => `dialog-${Math.random().toString(36).slice(2, 9)}`

  /**
   * Inserts a dialog into the stack at the correct position based on priority.
   * Higher priority dialogs are placed before lower priority ones.
   */
  function insertDialogByPriority(dialog: DialogInstance) {
    const insertIndex = dialogStack.value.findIndex(
      (d) => d.priority <= dialog.priority
    )

    dialogStack.value.splice(
      insertIndex === -1 ? dialogStack.value.length : insertIndex,
      0,
      dialog
    )
  }

  function riseDialog(options: { key: string }) {
    const dialogKey = options.key

    const index = dialogStack.value.findIndex((d) => d.key === dialogKey)
    if (index !== -1) {
      const [dialog] = dialogStack.value.splice(index, 1)
      insertDialogByPriority(dialog)
      activeKey.value = dialogKey
      updateCloseOnEscapeStates()
    }
  }

  function closeDialog(options?: { key: string }) {
    const targetDialog = options
      ? dialogStack.value.find((d) => d.key === options.key)
      : dialogStack.value.find((d) => d.key === activeKey.value)
    if (!targetDialog) return

    targetDialog.dialogComponentProps?.onClose?.()
    const index = dialogStack.value.indexOf(targetDialog)
    dialogStack.value.splice(index, 1)

    activeKey.value =
      dialogStack.value.length > 0
        ? dialogStack.value[dialogStack.value.length - 1].key
        : null

    updateCloseOnEscapeStates()
  }

  function createDialog<
    H extends Component = Component,
    B extends Component = Component,
    F extends Component = Component
  >(options: ShowDialogOptions<H, B, F> & { key: string }) {
    if (dialogStack.value.length >= 10) {
      dialogStack.value.shift()
    }

    const dialog = {
      key: options.key,
      visible: true,
      title: options.title,
      headerComponent: options.headerComponent
        ? markRaw(options.headerComponent)
        : undefined,
      footerComponent: options.footerComponent
        ? markRaw(options.footerComponent)
        : undefined,
      component: markRaw(options.component),
      headerProps: { ...options.headerProps },
      contentProps: { ...options.props },
      footerProps: { ...options.footerProps },
      priority: options.priority ?? 1,
      dialogComponentProps: {
        maximizable: false,
        modal: true,
        closable: true,
        closeOnEscape: true,
        dismissableMask: true,
        ...options.dialogComponentProps,
        maximized: false,
        onMaximize: () => {
          dialog.dialogComponentProps.maximized = true
        },
        onUnmaximize: () => {
          dialog.dialogComponentProps.maximized = false
        },
        onAfterHide: () => {
          closeDialog(dialog)
        },
        pt: merge(options.dialogComponentProps?.pt || {}, {
          root: {
            onMousedown: () => {
              riseDialog(dialog)
            }
          }
        })
      }
    }

    insertDialogByPriority(dialog)
    activeKey.value = options.key
    updateCloseOnEscapeStates()

    return dialog
  }

  /**
   * Ensures only the top-most dialog in the stack can be closed with the Escape key.
   * This is necessary because PrimeVue Dialogs do not handle `closeOnEscape` prop
   * correctly when multiple dialogs are open.
   */
  function updateCloseOnEscapeStates() {
    const topDialog = dialogStack.value.find((d) => d.key === activeKey.value)
    const topClosable = topDialog?.dialogComponentProps.closable

    dialogStack.value.forEach((dialog) => {
      dialog.dialogComponentProps = {
        ...dialog.dialogComponentProps,
        closeOnEscape: dialog === topDialog && !!topClosable
      }
    })
  }

  function showDialog<
    H extends Component = Component,
    B extends Component = Component,
    F extends Component = Component
  >(options: ShowDialogOptions<H, B, F>) {
    const dialogKey = options.key || genDialogKey()

    let dialog = dialogStack.value.find((d) => d.key === dialogKey)

    if (dialog) {
      dialog.visible = true
      riseDialog(dialog)
    } else {
      dialog = createDialog({ ...options, key: dialogKey })
    }
    return dialog
  }

  /**
   * Shows a dialog from a third party extension.
   * Explicitly keys extension dialogs with `extension-` prefix,
   * to avoid conflicts & prevent use of internal dialogs (available via `dialogService`).
   */
  function showExtensionDialog(options: ShowDialogOptions & { key: string }) {
    const { key } = options
    if (!key) {
      console.error('Extension dialog key is required')
      return
    }

    const extKey = key.startsWith('extension-') ? key : `extension-${key}`

    const dialog = dialogStack.value.find((d) => d.key === extKey)
    if (!dialog) return createDialog({ ...options, key: extKey })

    dialog.visible = true
    riseDialog(dialog)
    return dialog
  }

  function isDialogOpen(key: string) {
    return dialogStack.value.some((d) => d.key === key)
  }

  return {
    dialogStack,
    riseDialog,
    showDialog,
    closeDialog,
    showExtensionDialog,
    isDialogOpen,
    activeKey
  }
})
