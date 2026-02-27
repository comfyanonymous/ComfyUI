import _ from 'es-toolkit/compat'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

import { app, ComfyApp } from '@/scripts/app'
import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { useDialogStore } from '@/stores/dialogStore'
import { useMaskEditor } from '@/composables/maskeditor/useMaskEditor'
import { useCanvasTransform } from '@/composables/maskeditor/useCanvasTransform'

function openMaskEditor(node: LGraphNode): void {
  if (!node) {
    console.error('[MaskEditor] No node provided')
    return
  }

  if (!node.imgs?.length && node.previewMediaType !== 'image') {
    console.error('[MaskEditor] Node has no images')
    return
  }

  useMaskEditor().openMaskEditor(node)
}

// Open mask editor from clipspace (for plugin compatibility)
// This is called when ComfyApp.open_maskeditor() is invoked without arguments
function openMaskEditorFromClipspace(): void {
  const node = ComfyApp.clipspace_return_node as LGraphNode | null
  if (!node) {
    console.error('[MaskEditor] No clipspace_return_node found')
    return
  }

  openMaskEditor(node)
}

// Check if the dialog is already opened
function isOpened(): boolean {
  return useDialogStore().isDialogOpen('global-mask-editor')
}

const changeBrushSize = async (sizeChanger: (oldSize: number) => number) => {
  if (!isOpened()) return

  const store = useMaskEditorStore()
  const oldBrushSize = store.brushSettings.size
  const newBrushSize = sizeChanger(oldBrushSize)
  store.setBrushSize(newBrushSize)
}

app.registerExtension({
  name: 'Comfy.MaskEditor',
  settings: [
    {
      id: 'Comfy.MaskEditor.BrushAdjustmentSpeed',
      category: ['Mask Editor', 'BrushAdjustment', 'Sensitivity'],
      name: 'Brush adjustment speed multiplier',
      tooltip:
        'Controls how quickly the brush size and hardness change when adjusting. Higher values mean faster changes.',
      type: 'slider',
      attrs: {
        min: 0.1,
        max: 2.0,
        step: 0.1
      },
      defaultValue: 1.0,
      versionAdded: '1.0.0'
    },
    {
      id: 'Comfy.MaskEditor.UseDominantAxis',
      category: ['Mask Editor', 'BrushAdjustment', 'UseDominantAxis'],
      name: 'Lock brush adjustment to dominant axis',
      tooltip:
        'When enabled, brush adjustments will only affect size OR hardness based on which direction you move more',
      type: 'boolean',
      defaultValue: true
    }
  ],
  commands: [
    {
      id: 'Comfy.MaskEditor.OpenMaskEditor',
      icon: 'pi pi-pencil',
      label: 'Open Mask Editor for Selected Node',
      function: () => {
        const selectedNodes = app.canvas.selected_nodes
        if (!selectedNodes || Object.keys(selectedNodes).length !== 1) return

        const selectedNode = selectedNodes[Object.keys(selectedNodes)[0]]
        openMaskEditor(selectedNode)
      }
    },
    {
      id: 'Comfy.MaskEditor.BrushSize.Increase',
      icon: 'pi pi-plus-circle',
      label: 'Increase Brush Size in MaskEditor',
      function: () => changeBrushSize((old) => _.clamp(old + 2, 1, 250))
    },
    {
      id: 'Comfy.MaskEditor.BrushSize.Decrease',
      icon: 'pi pi-minus-circle',
      label: 'Decrease Brush Size in MaskEditor',
      function: () => changeBrushSize((old) => _.clamp(old - 2, 1, 250))
    },
    {
      id: 'Comfy.MaskEditor.ColorPicker',
      icon: 'pi pi-palette',
      label: 'Open Color Picker in MaskEditor',
      function: () => {
        if (!isOpened()) return

        const store = useMaskEditorStore()
        store.colorInput?.click()
      }
    },
    {
      id: 'Comfy.MaskEditor.Rotate.Right',
      icon: 'pi pi-refresh',
      label: 'Rotate Right in MaskEditor',
      function: async () => {
        if (!isOpened()) return
        await useCanvasTransform().rotateClockwise()
      }
    },
    {
      id: 'Comfy.MaskEditor.Rotate.Left',
      icon: 'pi pi-undo',
      label: 'Rotate Left in MaskEditor',
      function: async () => {
        if (!isOpened()) return
        await useCanvasTransform().rotateCounterclockwise()
      }
    },
    {
      id: 'Comfy.MaskEditor.Mirror.Horizontal',
      icon: 'pi pi-arrows-h',
      label: 'Mirror Horizontal in MaskEditor',
      function: async () => {
        if (!isOpened()) return
        await useCanvasTransform().mirrorHorizontal()
      }
    },
    {
      id: 'Comfy.MaskEditor.Mirror.Vertical',
      icon: 'pi pi-arrows-v',
      label: 'Mirror Vertical in MaskEditor',
      function: async () => {
        if (!isOpened()) return
        await useCanvasTransform().mirrorVertical()
      }
    }
  ],
  init() {
    // Set up ComfyApp static methods for plugin compatibility (deprecated)
    ComfyApp.open_maskeditor = openMaskEditorFromClipspace

    console.warn(
      '[MaskEditor] ComfyApp.open_maskeditor is deprecated. ' +
        'Plugins should migrate to using the command system or direct node context menu integration.'
    )
  }
})
