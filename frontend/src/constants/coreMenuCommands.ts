import { isCloud } from '@/platform/distribution/types'

export const CORE_MENU_COMMANDS = [
  [[], ['Comfy.NewBlankWorkflow']],
  [[], []], // Separator after New
  [['File'], ['Comfy.OpenWorkflow']],
  [
    ['File'],
    [
      'Comfy.SaveWorkflow',
      'Comfy.SaveWorkflowAs',
      'Comfy.ExportWorkflow',
      'Comfy.ExportWorkflowAPI'
    ]
  ],
  [['Edit'], ['Comfy.Undo', 'Comfy.Redo']],
  [
    ['Edit'],
    [
      'Comfy.Canvas.CopySelected',
      'Comfy.Canvas.PasteFromClipboard',
      'Comfy.Canvas.SelectAll'
    ]
  ],
  [['Edit'], ['Comfy.ClearWorkflow']],
  [['Edit'], ['Comfy.OpenClipspace']],
  [
    ['Edit'],
    [
      'Comfy.RefreshNodeDefinitions',
      ...(isCloud
        ? []
        : [
            'Comfy.Memory.UnloadModels',
            'Comfy.Memory.UnloadModelsAndExecutionCache'
          ])
    ]
  ],
  [['View'], []],
  [
    ['Help'],
    [
      'Comfy.Help.OpenComfyUIIssues',
      'Comfy.Help.OpenComfyUIDocs',
      'Comfy.Help.OpenComfyOrgDiscord',
      'Comfy.Help.OpenComfyUIForum'
    ]
  ],
  [['Help'], ['Comfy.Help.AboutComfyUI', 'Comfy.ContactSupport']]
]
