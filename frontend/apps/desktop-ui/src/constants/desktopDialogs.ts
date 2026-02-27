export interface DialogAction {
  readonly label: string
  readonly action: 'openUrl' | 'close' | 'cancel'
  readonly url?: string
  readonly severity?: 'danger' | 'primary' | 'secondary' | 'warn'
  readonly returnValue: string
}

interface DesktopDialog {
  readonly title: string
  readonly message: string
  readonly buttons: DialogAction[]
}

export const DESKTOP_DIALOGS = {
  /** Shown when a corrupt venv is detected. */
  reinstallVenv: {
    title: 'Reinstall ComfyUI (Fresh Start)?',
    message: `Sorry, we can't launch ComfyUI because some installed packages aren't compatible.

Click Reinstall to restore ComfyUI and get back up and running.

Please note: if you've added custom nodes, you'll need to reinstall them after this process.`,
    buttons: [
      {
        label: 'Learn More',
        action: 'openUrl',
        url: 'https://docs.comfy.org',
        returnValue: 'openDocs'
      },
      {
        label: 'Reinstall',
        action: 'close',
        severity: 'danger',
        returnValue: 'resetVenv'
      }
    ]
  },
  /** A dialog that is shown when an invalid dialog ID is provided. */
  invalidDialog: {
    title: 'Invalid Dialog',
    message: `Invalid dialog ID was provided.`,
    buttons: [
      {
        label: 'Close',
        action: 'cancel',
        returnValue: 'cancel'
      }
    ]
  }
} as const satisfies { [K: string]: DesktopDialog }

/** The ID of a desktop dialog. */
type DesktopDialogId = keyof typeof DESKTOP_DIALOGS

/**
 * Checks if {@link id} is a valid dialog ID.
 * @param id The string to check
 * @returns `true` if the ID is a valid dialog ID, otherwise `false`
 */
function isDialogId(id: unknown): id is DesktopDialogId {
  return typeof id === 'string' && id in DESKTOP_DIALOGS
}

/**
 * Gets the dialog with the given ID.
 * @param dialogId The ID of the dialog to get
 * @returns The dialog with the given ID
 */
export function getDialog(
  dialogId: string | string[]
): DesktopDialog & { id: DesktopDialogId } {
  const id = isDialogId(dialogId) ? dialogId : 'invalidDialog'
  return { id, ...structuredClone(DESKTOP_DIALOGS[id]) }
}
