import { markRaw } from 'vue'

import WorkflowsSidebarTab from '@/components/sidebar/tabs/WorkflowsSidebarTab.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { SidebarTabExtension } from '@/types/extensionTypes'

export const useWorkflowsSidebarTab = (): SidebarTabExtension => {
  const settingStore = useSettingStore()
  const workflowStore = useWorkflowStore()
  return {
    id: 'workflows',
    icon: 'icon-[comfy--workflow]',
    iconBadge: () => {
      if (
        settingStore.get('Comfy.Workflow.WorkflowTabsPosition') !== 'Sidebar'
      ) {
        return null
      }
      const value = workflowStore.openWorkflows.length.toString()
      return value === '0' ? null : value
    },
    title: 'sideToolbar.workflows',
    tooltip: 'sideToolbar.workflows',
    label: 'sideToolbar.labels.workflows',
    component: markRaw(WorkflowsSidebarTab),
    type: 'vue'
  }
}
