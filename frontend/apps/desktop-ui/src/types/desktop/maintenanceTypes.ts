import type { PrimeVueSeverity } from '../primeVueTypes'

interface MaintenanceTaskButton {
  /** The text to display on the button. */
  text?: string
  /** CSS classes used for the button icon, e.g. 'pi pi-external-link' */
  icon?: string
}

/** A maintenance task, used by the maintenance page. */
export interface MaintenanceTask {
  /** ID string used as i18n key */
  id: string
  /** The display name of the task, e.g. Git */
  name: string
  /** Short description of the task. */
  shortDescription?: string
  /** Description of the task when it is in an error state. */
  errorDescription?: string
  /** Description of the task when it is in a warning state. */
  warningDescription?: string
  /** Full description of the task when it is in an OK state. */
  description?: string
  /** URL to the image to show in card mode. */
  headerImg?: string
  /** The button to display on the task card / list item. */
  button?: MaintenanceTaskButton
  /** Whether to show a confirmation dialog before running the task. */
  requireConfirm?: boolean
  /** The text to display in the confirmation dialog. */
  confirmText?: string
  /** Called by onClick to run the actual task. */
  execute: (args?: unknown[]) => boolean | Promise<boolean>
  /** Show the button with `severity="danger"` */
  severity?: PrimeVueSeverity
  /** Whether this task should display the terminal window when run. */
  usesTerminal?: boolean
  /** If `true`, successful completion of this task will refresh install validation and automatically continue if successful. */
  isInstallationFix?: boolean
}

/** The filter options for the maintenance task list. */
export interface MaintenanceFilter {
  /** CSS classes used for the filter button icon, e.g. 'pi pi-cross' */
  icon: string
  /** The text to display on the filter button. */
  value: string
  /** The tasks to display when this filter is selected. */
  tasks: ReadonlyArray<MaintenanceTask>
}
