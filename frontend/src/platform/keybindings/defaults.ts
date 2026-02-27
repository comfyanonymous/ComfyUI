import type { Keybinding } from './types'

export const CORE_KEYBINDINGS: Keybinding[] = [
  {
    combo: {
      ctrl: true,
      key: 'Enter'
    },
    commandId: 'Comfy.QueuePrompt'
  },
  {
    combo: {
      ctrl: true,
      shift: true,
      key: 'Enter'
    },
    commandId: 'Comfy.QueuePromptFront'
  },
  {
    combo: {
      ctrl: true,
      alt: true,
      key: 'Enter'
    },
    commandId: 'Comfy.Interrupt'
  },
  {
    combo: {
      key: 'r'
    },
    commandId: 'Comfy.RefreshNodeDefinitions'
  },
  {
    combo: {
      key: 'w'
    },
    commandId: 'Workspace.ToggleSidebarTab.workflows'
  },
  {
    combo: {
      key: 'n'
    },
    commandId: 'Workspace.ToggleSidebarTab.node-library'
  },
  {
    combo: {
      key: 'm'
    },
    commandId: 'Workspace.ToggleSidebarTab.model-library'
  },
  {
    combo: {
      key: 'a'
    },
    commandId: 'Workspace.ToggleSidebarTab.assets'
  },
  {
    combo: {
      ctrl: true,
      shift: true,
      key: 'a'
    },
    commandId: 'Comfy.ToggleLinear'
  },
  {
    combo: {
      key: 's',
      ctrl: true
    },
    commandId: 'Comfy.SaveWorkflow'
  },
  {
    combo: {
      key: 'o',
      ctrl: true
    },
    commandId: 'Comfy.OpenWorkflow'
  },
  {
    combo: {
      key: 'g',
      ctrl: true
    },
    commandId: 'Comfy.Graph.GroupSelectedNodes'
  },
  {
    combo: {
      key: ',',
      ctrl: true
    },
    commandId: 'Comfy.ShowSettingsDialog'
  },
  {
    combo: {
      key: '=',
      alt: true
    },
    commandId: 'Comfy.Canvas.ZoomIn',
    targetElementId: 'graph-canvas'
  },
  {
    combo: {
      key: '+',
      alt: true,
      shift: true
    },
    commandId: 'Comfy.Canvas.ZoomIn',
    targetElementId: 'graph-canvas'
  },
  {
    combo: {
      key: '+',
      alt: true
    },
    commandId: 'Comfy.Canvas.ZoomIn',
    targetElementId: 'graph-canvas'
  },
  {
    combo: {
      key: '-',
      alt: true
    },
    commandId: 'Comfy.Canvas.ZoomOut',
    targetElementId: 'graph-canvas'
  },
  {
    combo: {
      key: '.'
    },
    commandId: 'Comfy.Canvas.FitView',
    targetElementId: 'graph-canvas-container'
  },
  {
    combo: {
      key: 'p'
    },
    commandId: 'Comfy.Canvas.ToggleSelected.Pin',
    targetElementId: 'graph-canvas-container'
  },
  {
    combo: {
      key: 'c',
      alt: true
    },
    commandId: 'Comfy.Canvas.ToggleSelectedNodes.Collapse',
    targetElementId: 'graph-canvas-container'
  },
  {
    combo: {
      key: 'b',
      ctrl: true
    },
    commandId: 'Comfy.Canvas.ToggleSelectedNodes.Bypass',
    targetElementId: 'graph-canvas-container'
  },
  {
    combo: {
      key: 'm',
      ctrl: true
    },
    commandId: 'Comfy.Canvas.ToggleSelectedNodes.Mute',
    targetElementId: 'graph-canvas-container'
  },
  {
    combo: {
      key: '`',
      ctrl: true
    },
    commandId: 'Workspace.ToggleBottomPanelTab.logs-terminal'
  },
  {
    combo: {
      key: 'e',
      ctrl: true,
      shift: true
    },
    commandId: 'Comfy.Graph.ConvertToSubgraph'
  },
  {
    combo: {
      key: 'm',
      alt: true
    },
    commandId: 'Comfy.Canvas.ToggleMinimap'
  },
  {
    combo: {
      ctrl: true,
      shift: true,
      key: 'k'
    },
    commandId: 'Workspace.ToggleBottomPanel.Shortcuts'
  },
  {
    combo: {
      key: 'v'
    },
    commandId: 'Comfy.Canvas.Unlock'
  },
  {
    combo: {
      key: 'h'
    },
    commandId: 'Comfy.Canvas.Lock'
  },
  {
    combo: {
      key: 'Escape'
    },
    commandId: 'Comfy.Graph.ExitSubgraph'
  }
]
