import { markRaw } from 'vue'

import LogsTerminal from '@/components/bottomPanel/tabs/terminal/LogsTerminal.vue'
import CommandTerminal from '@/components/bottomPanel/tabs/terminal/CommandTerminal.vue'
import type { BottomPanelExtension } from '@/types/extensionTypes'

export function useLogsTerminalTab(): BottomPanelExtension {
  return {
    id: 'logs-terminal',
    title: 'Logs',
    titleKey: 'g.logs',
    component: markRaw(LogsTerminal),
    type: 'vue'
  }
}

export function useCommandTerminalTab(): BottomPanelExtension {
  return {
    id: 'command-terminal',
    title: 'Terminal',
    titleKey: 'g.terminal',
    component: markRaw(CommandTerminal),
    type: 'vue'
  }
}
