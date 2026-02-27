import type { OverlayIconProps } from '@/components/common/OverlayIcon.vue'

export type WorkflowMenuItem = WorkflowMenuSeparator | WorkflowMenuAction

interface WorkflowMenuSeparator {
  separator: true
}

export interface WorkflowMenuAction {
  separator?: false
  label: string
  icon?: string
  command?: () => void
  disabled?: boolean
  badge?: string
  overlayIcon?: OverlayIconProps
}
