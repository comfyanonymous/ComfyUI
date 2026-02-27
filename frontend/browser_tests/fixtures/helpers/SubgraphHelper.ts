import type { Page } from '@playwright/test'

import type {
  CanvasPointerEvent,
  Subgraph
} from '@/lib/litegraph/src/litegraph'

import type { ComfyPage } from '../ComfyPage'
import type { NodeReference } from '../utils/litegraphUtils'
import { SubgraphSlotReference } from '../utils/litegraphUtils'

export class SubgraphHelper {
  constructor(private readonly comfyPage: ComfyPage) {}

  private get page(): Page {
    return this.comfyPage.page
  }

  /**
   * Core helper method for interacting with subgraph I/O slots.
   * Handles both input/output slots and both right-click/double-click actions.
   *
   * @param slotType - 'input' or 'output'
   * @param action - 'rightClick' or 'doubleClick'
   * @param slotName - Optional specific slot name to target
   */
  private async interactWithSubgraphSlot(
    slotType: 'input' | 'output',
    action: 'rightClick' | 'doubleClick',
    slotName?: string
  ): Promise<void> {
    const foundSlot = await this.page.evaluate(
      async (params) => {
        const { slotType, action, targetSlotName } = params
        const app = window.app!
        const currentGraph = app.canvas!.graph!

        // Check if we're in a subgraph
        if (!('inputNode' in currentGraph)) {
          throw new Error(
            'Not in a subgraph - this method only works inside subgraphs'
          )
        }

        const subgraph = currentGraph as Subgraph

        // Get the appropriate node and slots
        const node =
          slotType === 'input' ? subgraph.inputNode : subgraph.outputNode
        const slots = slotType === 'input' ? subgraph.inputs : subgraph.outputs

        if (!node) {
          throw new Error(`No ${slotType} node found in subgraph`)
        }

        if (!slots || slots.length === 0) {
          throw new Error(`No ${slotType} slots found in subgraph`)
        }

        // Filter slots based on target name and action type
        const slotsToTry = targetSlotName
          ? slots.filter((slot) => slot.name === targetSlotName)
          : action === 'rightClick'
            ? slots
            : [slots[0]] // Right-click tries all, double-click uses first

        if (slotsToTry.length === 0) {
          throw new Error(
            targetSlotName
              ? `${slotType} slot '${targetSlotName}' not found`
              : `No ${slotType} slots available to try`
          )
        }

        // Handle the interaction based on action type
        if (action === 'rightClick') {
          // Right-click: try each slot until one works
          for (const slot of slotsToTry) {
            if (!slot.pos) continue

            const event = {
              canvasX: slot.pos[0],
              canvasY: slot.pos[1],
              button: 2, // Right mouse button
              preventDefault: () => {},
              stopPropagation: () => {}
            }

            if (node.onPointerDown) {
              node.onPointerDown(
                event as Partial<CanvasPointerEvent> as CanvasPointerEvent,
                app.canvas.pointer,
                app.canvas.linkConnector
              )
              return {
                success: true,
                slotName: slot.name,
                x: slot.pos[0],
                y: slot.pos[1]
              }
            }
          }
        } else if (action === 'doubleClick') {
          // Double-click: use first slot with bounding rect center
          const slot = slotsToTry[0]
          if (!slot.boundingRect) {
            throw new Error(`${slotType} slot bounding rect not found`)
          }

          const rect = slot.boundingRect
          const testX = rect[0] + rect[2] / 2 // x + width/2
          const testY = rect[1] + rect[3] / 2 // y + height/2

          const event = {
            canvasX: testX,
            canvasY: testY,
            button: 0, // Left mouse button
            preventDefault: () => {},
            stopPropagation: () => {}
          }

          if (node.onPointerDown) {
            node.onPointerDown(
              event as Partial<CanvasPointerEvent> as CanvasPointerEvent,
              app.canvas.pointer,
              app.canvas.linkConnector
            )

            // Trigger double-click
            if (app.canvas.pointer.onDoubleClick) {
              app.canvas.pointer.onDoubleClick(
                event as Partial<CanvasPointerEvent> as CanvasPointerEvent
              )
            }
          }

          return { success: true, slotName: slot.name, x: testX, y: testY }
        }

        return { success: false }
      },
      { slotType, action, targetSlotName: slotName }
    )

    if (!foundSlot.success) {
      const actionText =
        action === 'rightClick' ? 'open context menu for' : 'double-click'
      throw new Error(
        slotName
          ? `Could not ${actionText} ${slotType} slot '${slotName}'`
          : `Could not find any ${slotType} slot to ${actionText}`
      )
    }

    // Wait for the appropriate UI element to appear
    if (action === 'rightClick') {
      await this.page.waitForSelector('.litemenu-entry', {
        state: 'visible',
        timeout: 5000
      })
    } else {
      await this.comfyPage.nextFrame()
    }
  }

  /**
   * Right-clicks on a subgraph input slot to open the context menu.
   * Must be called when inside a subgraph.
   *
   * This method uses the actual slot positions from the subgraph.inputs array,
   * which contain the correct coordinates for each input slot. These positions
   * are different from the visual node positions and are specifically where
   * the slots are rendered on the input node.
   *
   * @param inputName Optional name of the specific input slot to target (e.g., 'text').
   *                  If not provided, tries all available input slots until one works.
   * @returns Promise that resolves when the context menu appears
   */
  async rightClickInputSlot(inputName?: string): Promise<void> {
    return this.interactWithSubgraphSlot('input', 'rightClick', inputName)
  }

  /**
   * Right-clicks on a subgraph output slot to open the context menu.
   * Must be called when inside a subgraph.
   *
   * Similar to rightClickInputSlot but for output slots.
   *
   * @param outputName Optional name of the specific output slot to target.
   *                   If not provided, tries all available output slots until one works.
   * @returns Promise that resolves when the context menu appears
   */
  async rightClickOutputSlot(outputName?: string): Promise<void> {
    return this.interactWithSubgraphSlot('output', 'rightClick', outputName)
  }

  /**
   * Double-clicks on a subgraph input slot to rename it.
   * Must be called when inside a subgraph.
   *
   * @param inputName Optional name of the specific input slot to target (e.g., 'text').
   *                  If not provided, tries the first available input slot.
   * @returns Promise that resolves when the rename dialog appears
   */
  async doubleClickInputSlot(inputName?: string): Promise<void> {
    return this.interactWithSubgraphSlot('input', 'doubleClick', inputName)
  }

  /**
   * Double-clicks on a subgraph output slot to rename it.
   * Must be called when inside a subgraph.
   *
   * @param outputName Optional name of the specific output slot to target.
   *                   If not provided, tries the first available output slot.
   * @returns Promise that resolves when the rename dialog appears
   */
  async doubleClickOutputSlot(outputName?: string): Promise<void> {
    return this.interactWithSubgraphSlot('output', 'doubleClick', outputName)
  }

  /**
   * Get a reference to a subgraph input slot
   */
  getInputSlot(slotName?: string): SubgraphSlotReference {
    return new SubgraphSlotReference('input', slotName || '', this.comfyPage)
  }

  /**
   * Get a reference to a subgraph output slot
   */
  getOutputSlot(slotName?: string): SubgraphSlotReference {
    return new SubgraphSlotReference('output', slotName || '', this.comfyPage)
  }

  /**
   * Connect a regular node output to a subgraph input.
   * This creates a new input slot on the subgraph if targetInputName is not provided.
   */
  async connectToInput(
    sourceNode: NodeReference,
    sourceSlotIndex: number,
    targetInputName?: string
  ): Promise<void> {
    const sourceSlot = await sourceNode.getOutput(sourceSlotIndex)
    const targetSlot = this.getInputSlot(targetInputName)

    const targetPosition = targetInputName
      ? await targetSlot.getPosition() // Connect to existing slot
      : await targetSlot.getOpenSlotPosition() // Create new slot

    await this.comfyPage.canvasOps.dragAndDrop(
      await sourceSlot.getPosition(),
      targetPosition
    )
    await this.comfyPage.nextFrame()
  }

  /**
   * Connect a subgraph input to a regular node input.
   * This creates a new input slot on the subgraph if sourceInputName is not provided.
   */
  async connectFromInput(
    targetNode: NodeReference,
    targetSlotIndex: number,
    sourceInputName?: string
  ): Promise<void> {
    const sourceSlot = this.getInputSlot(sourceInputName)
    const targetSlot = await targetNode.getInput(targetSlotIndex)

    const sourcePosition = sourceInputName
      ? await sourceSlot.getPosition() // Connect from existing slot
      : await sourceSlot.getOpenSlotPosition() // Create new slot

    const targetPosition = await targetSlot.getPosition()

    await this.comfyPage.canvasOps.dragAndDrop(sourcePosition, targetPosition)
    await this.comfyPage.nextFrame()
  }

  /**
   * Connect a regular node output to a subgraph output.
   * This creates a new output slot on the subgraph if targetOutputName is not provided.
   */
  async connectToOutput(
    sourceNode: NodeReference,
    sourceSlotIndex: number,
    targetOutputName?: string
  ): Promise<void> {
    const sourceSlot = await sourceNode.getOutput(sourceSlotIndex)
    const targetSlot = this.getOutputSlot(targetOutputName)

    const targetPosition = targetOutputName
      ? await targetSlot.getPosition() // Connect to existing slot
      : await targetSlot.getOpenSlotPosition() // Create new slot

    await this.comfyPage.canvasOps.dragAndDrop(
      await sourceSlot.getPosition(),
      targetPosition
    )
    await this.comfyPage.nextFrame()
  }

  /**
   * Connect a subgraph output to a regular node input.
   * This creates a new output slot on the subgraph if sourceOutputName is not provided.
   */
  async connectFromOutput(
    targetNode: NodeReference,
    targetSlotIndex: number,
    sourceOutputName?: string
  ): Promise<void> {
    const sourceSlot = this.getOutputSlot(sourceOutputName)
    const targetSlot = await targetNode.getInput(targetSlotIndex)

    const sourcePosition = sourceOutputName
      ? await sourceSlot.getPosition() // Connect from existing slot
      : await sourceSlot.getOpenSlotPosition() // Create new slot

    await this.comfyPage.canvasOps.dragAndDrop(
      sourcePosition,
      await targetSlot.getPosition()
    )
    await this.comfyPage.nextFrame()
  }
}
