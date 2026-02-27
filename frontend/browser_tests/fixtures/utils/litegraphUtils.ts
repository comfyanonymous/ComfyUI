import { expect } from '@playwright/test'
import type { Page } from '@playwright/test'

import type { NodeId } from '../../../src/platform/workflow/validation/schemas/workflowSchema'
import { ManageGroupNode } from '../../helpers/manageGroupNode'
import type { ComfyPage } from '../ComfyPage'
import type { Position, Size } from '../types'

export const getMiddlePoint = (pos1: Position, pos2: Position) => {
  return {
    x: (pos1.x + pos2.x) / 2,
    y: (pos1.y + pos2.y) / 2
  }
}

export class SubgraphSlotReference {
  constructor(
    readonly type: 'input' | 'output',
    readonly slotName: string,
    readonly comfyPage: ComfyPage
  ) {}

  async getPosition(): Promise<Position> {
    const pos: [number, number] = await this.comfyPage.page.evaluate(
      ([type, slotName]) => {
        const currentGraph = window.app!.canvas.graph!

        // Check if we're in a subgraph (subgraphs have inputNode property)
        if (!('inputNode' in currentGraph)) {
          throw new Error(
            'Not in a subgraph - this method only works inside subgraphs'
          )
        }

        const slots =
          type === 'input' ? currentGraph.inputs : currentGraph.outputs
        if (!slots || slots.length === 0) {
          throw new Error(`No ${type} slots found in subgraph`)
        }

        // Find the specific slot or use the first one if no name specified
        const slot = slotName
          ? slots.find((s) => s.name === slotName)
          : slots[0]

        if (!slot) {
          throw new Error(`${type} slot '${slotName}' not found`)
        }

        if (!slot.pos) {
          throw new Error(`${type} slot '${slotName}' has no position`)
        }

        // Convert from offset to canvas coordinates
        const canvasPos = window.app!.canvas.ds.convertOffsetToCanvas([
          slot.pos[0],
          slot.pos[1]
        ])
        return canvasPos
      },
      [this.type, this.slotName] as const
    )

    return {
      x: pos[0],
      y: pos[1]
    }
  }

  async getOpenSlotPosition(): Promise<Position> {
    const pos: [number, number] = await this.comfyPage.page.evaluate(
      ([type]) => {
        const currentGraph = window.app!.canvas.graph!

        // Check if we're in a subgraph (subgraphs have inputNode property)
        if (!('inputNode' in currentGraph)) {
          throw new Error(
            'Not in a subgraph - this method only works inside subgraphs'
          )
        }

        const node =
          type === 'input' ? currentGraph.inputNode : currentGraph.outputNode

        if (!node) {
          throw new Error(`No ${type} node found in subgraph`)
        }

        // Convert from offset to canvas coordinates
        const canvasPos = window.app!.canvas.ds.convertOffsetToCanvas([
          node.emptySlot.pos[0],
          node.emptySlot.pos[1]
        ])
        return canvasPos
      },
      [this.type] as const
    )

    return {
      x: pos[0],
      y: pos[1]
    }
  }
}

class NodeSlotReference {
  constructor(
    readonly type: 'input' | 'output',
    readonly index: number,
    readonly node: NodeReference
  ) {}
  async getPosition() {
    const pos: [number, number] = await this.node.comfyPage.page.evaluate(
      ([type, id, index]) => {
        // Use canvas.graph to get the current graph (works in both main graph and subgraphs)
        const node = window.app!.canvas.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)

        const rawPos = node.getConnectionPos(type === 'input', index)
        const convertedPos =
          window.app!.canvas.ds!.convertOffsetToCanvas(rawPos)

        // Debug logging - convert Float64Arrays to regular arrays for visibility
        console.warn(
          `NodeSlotReference debug for ${type} slot ${index} on node ${id}:`,
          {
            nodePos: [node.pos[0], node.pos[1]],
            nodeSize: [node.size[0], node.size[1]],
            rawConnectionPos: [rawPos[0], rawPos[1]],
            convertedPos: [convertedPos[0], convertedPos[1]],
            currentGraphType:
              'inputNode' in window.app!.canvas.graph! ? 'Subgraph' : 'LGraph'
          }
        )

        return convertedPos
      },
      [this.type, this.node.id, this.index] as const
    )
    return {
      x: pos[0],
      y: pos[1]
    }
  }
  async getLinkCount() {
    return await this.node.comfyPage.page.evaluate(
      ([type, id, index]) => {
        const node = window.app!.canvas.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)
        if (type === 'input') {
          return node.inputs[index].link == null ? 0 : 1
        }
        return node.outputs[index].links?.length ?? 0
      },
      [this.type, this.node.id, this.index] as const
    )
  }
  async removeLinks() {
    await this.node.comfyPage.page.evaluate(
      ([type, id, index]) => {
        const node = window.app!.canvas.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)
        if (type === 'input') {
          node.disconnectInput(index)
        } else {
          node.disconnectOutput(index)
        }
      },
      [this.type, this.node.id, this.index] as const
    )
  }
}

class NodeWidgetReference {
  constructor(
    readonly index: number,
    readonly node: NodeReference
  ) {}

  /**
   * @returns The position of the widget's center
   */
  async getPosition(): Promise<Position> {
    const pos: [number, number] = await this.node.comfyPage.page.evaluate(
      ([id, index]) => {
        const node = window.app!.canvas.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)
        const widget = node.widgets![index]
        if (!widget) throw new Error(`Widget ${index} not found.`)

        const [x, y, w, _h] = node.getBounding()
        return window.app!.canvasPosToClientPos([
          x + w / 2,
          y + window.LiteGraph!['NODE_TITLE_HEIGHT'] + widget.last_y! + 1
        ])
      },
      [this.node.id, this.index] as const
    )
    return {
      x: pos[0],
      y: pos[1]
    }
  }

  /**
   * @returns The position of the widget's associated socket
   */
  async getSocketPosition(): Promise<Position> {
    const pos: [number, number] = await this.node.comfyPage.page.evaluate(
      ([id, index]) => {
        const node = window.app!.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)
        const widget = node.widgets![index]
        if (!widget) throw new Error(`Widget ${index} not found.`)

        const slot = node.inputs.find(
          (slot) => slot.widget?.name === widget.name
        )
        if (!slot) throw new Error(`Socket ${widget.name} not found.`)

        const [x, y] = node.getBounding()
        return window.app!.canvasPosToClientPos([
          x + slot.pos![0],
          y + slot.pos![1] + window.LiteGraph!['NODE_TITLE_HEIGHT']
        ])
      },
      [this.node.id, this.index] as const
    )
    return {
      x: pos[0],
      y: pos[1]
    }
  }

  async click() {
    await this.node.comfyPage.canvas.click({
      position: await this.getPosition()
    })
  }

  async dragHorizontal(delta: number) {
    const pos = await this.getPosition()
    const canvas = this.node.comfyPage.canvas
    const canvasPos = (await canvas.boundingBox())!
    await this.node.comfyPage.canvasOps.dragAndDrop(
      {
        x: canvasPos.x + pos.x,
        y: canvasPos.y + pos.y
      },
      {
        x: canvasPos.x + pos.x + delta,
        y: canvasPos.y + pos.y
      }
    )
  }

  async getValue() {
    return await this.node.comfyPage.page.evaluate(
      ([id, index]) => {
        const node = window.app!.graph!.getNodeById(id)
        if (!node) throw new Error(`Node ${id} not found.`)
        const widget = node.widgets![index]
        if (!widget) throw new Error(`Widget ${index} not found.`)
        return widget.value
      },
      [this.node.id, this.index] as const
    )
  }
}
export class NodeReference {
  constructor(
    readonly id: NodeId,
    readonly comfyPage: ComfyPage
  ) {}
  async exists(): Promise<boolean> {
    return await this.comfyPage.page.evaluate((id) => {
      const node = window.app!.canvas.graph!.getNodeById(id)
      return !!node
    }, this.id)
  }
  getType(): Promise<string> {
    return this.getProperty('type')
  }
  async getPosition(): Promise<Position> {
    const pos = await this.comfyPage.canvasOps.convertOffsetToCanvas(
      await this.getProperty<[number, number]>('pos')
    )
    return {
      x: pos[0],
      y: pos[1]
    }
  }
  async getBounding(): Promise<Position & Size> {
    const [x, y, width, height] = await this.comfyPage.page.evaluate((id) => {
      const node = window.app!.canvas.graph!.getNodeById(id)
      if (!node) throw new Error('Node not found')
      return [...node.getBounding()] as [number, number, number, number]
    }, this.id)
    return {
      x,
      y,
      width,
      height
    }
  }
  async getSize(): Promise<Size> {
    const size = await this.getProperty<[number, number]>('size')
    return {
      width: size[0],
      height: size[1]
    }
  }
  async getFlags(): Promise<{ collapsed?: boolean; pinned?: boolean }> {
    return await this.getProperty('flags')
  }
  async getTitlePosition(): Promise<Position> {
    const nodePos = await this.getPosition()
    const nodeSize = await this.getSize()
    return { x: nodePos.x + nodeSize.width / 2, y: nodePos.y - 15 }
  }
  async isPinned() {
    return !!(await this.getFlags()).pinned
  }
  async isCollapsed() {
    return !!(await this.getFlags()).collapsed
  }
  async isBypassed() {
    return (await this.getProperty<number | null | undefined>('mode')) === 4
  }
  async getProperty<T>(prop: string): Promise<T> {
    return await this.comfyPage.page.evaluate(
      ([id, prop]) => {
        const node = window.app!.canvas.graph!.getNodeById(id)
        if (!node) throw new Error('Node not found')
        return (node as unknown as Record<string, T>)[prop]
      },
      [this.id, prop] as const
    )
  }
  async getOutput(index: number) {
    return new NodeSlotReference('output', index, this)
  }
  async getInput(index: number) {
    return new NodeSlotReference('input', index, this)
  }
  async getWidget(index: number) {
    return new NodeWidgetReference(index, this)
  }
  async click(
    position: 'title' | 'collapse',
    options?: Parameters<Page['click']>[1] & { moveMouseToEmptyArea?: boolean }
  ) {
    let clickPos: Position
    switch (position) {
      case 'title':
        clickPos = await this.getTitlePosition()
        break
      case 'collapse': {
        const nodePos = await this.getPosition()
        clickPos = { x: nodePos.x + 5, y: nodePos.y - 10 }
        break
      }
      default:
        throw new Error(`Invalid click position ${position}`)
    }

    const moveMouseToEmptyArea = options?.moveMouseToEmptyArea
    if (options) {
      delete options.moveMouseToEmptyArea
    }

    await this.comfyPage.canvas.click({
      ...options,
      position: clickPos,
      force: true
    })
    await this.comfyPage.nextFrame()
    if (moveMouseToEmptyArea) {
      await this.comfyPage.canvasOps.moveMouseToEmptyArea()
    }
  }
  async copy() {
    await this.click('title')
    await this.comfyPage.clipboard.copy()
    await this.comfyPage.nextFrame()
  }
  async connectWidget(
    originSlotIndex: number,
    targetNode: NodeReference,
    targetWidgetIndex: number
  ) {
    const originSlot = await this.getOutput(originSlotIndex)
    const targetWidget = await targetNode.getWidget(targetWidgetIndex)
    await this.comfyPage.canvasOps.dragAndDrop(
      await originSlot.getPosition(),
      await targetWidget.getSocketPosition()
    )
    return originSlot
  }
  async connectOutput(
    originSlotIndex: number,
    targetNode: NodeReference,
    targetSlotIndex: number
  ) {
    const originSlot = await this.getOutput(originSlotIndex)
    const targetSlot = await targetNode.getInput(targetSlotIndex)
    await this.comfyPage.canvasOps.dragAndDrop(
      await originSlot.getPosition(),
      await targetSlot.getPosition()
    )
    return originSlot
  }
  async getContextMenuOptionNames() {
    await this.click('title', { button: 'right' })
    const ctx = this.comfyPage.page.locator('.litecontextmenu')
    return await ctx.locator('.litemenu-entry').allInnerTexts()
  }
  async clickContextMenuOption(optionText: string) {
    await this.click('title', { button: 'right' })
    const ctx = this.comfyPage.page.locator('.litecontextmenu')
    await ctx.getByText(optionText).click()
  }
  async convertToGroupNode(groupNodeName: string = 'GroupNode') {
    await this.clickContextMenuOption('Convert to Group Node')
    await this.comfyPage.nodeOps.fillPromptDialog(groupNodeName)
    await this.comfyPage.nextFrame()
    const nodes = await this.comfyPage.nodeOps.getNodeRefsByType(
      `workflow>${groupNodeName}`
    )
    if (nodes.length !== 1) {
      throw new Error(`Did not find single group node (found=${nodes.length})`)
    }
    return nodes[0]
  }
  async convertToSubgraph() {
    await this.clickContextMenuOption('Convert to Subgraph')
    await this.comfyPage.nextFrame()
    const nodes =
      await this.comfyPage.nodeOps.getNodeRefsByTitle('New Subgraph')
    if (nodes.length !== 1) {
      throw new Error(
        `Did not find single subgraph node (found=${nodes.length})`
      )
    }
    return nodes[0]
  }
  async manageGroupNode() {
    await this.clickContextMenuOption('Manage Group Node')
    await this.comfyPage.nextFrame()
    return new ManageGroupNode(
      this.comfyPage.page,
      this.comfyPage.page.locator('.comfy-group-manage')
    )
  }
  async navigateIntoSubgraph() {
    const titleHeight = await this.comfyPage.page.evaluate(() => {
      return window.LiteGraph!['NODE_TITLE_HEIGHT']
    })
    const nodePos = await this.getPosition()
    const nodeSize = await this.getSize()

    // Try multiple positions to avoid DOM widget interference
    const clickPositions = [
      { x: nodePos.x + nodeSize.width / 2, y: nodePos.y + titleHeight + 5 },
      {
        x: nodePos.x + nodeSize.width / 2,
        y: nodePos.y + nodeSize.height / 2
      },
      { x: nodePos.x + 20, y: nodePos.y + titleHeight + 5 }
    ]

    // Click the enter_subgraph title button (top-right of title bar).
    // This is more reliable than dblclick on the node body because
    // promoted DOM widgets can overlay the body and intercept events.
    const subgraphButtonPos = {
      x: nodePos.x + nodeSize.width - 15,
      y: nodePos.y - titleHeight / 2
    }

    const checkIsInSubgraph = async () => {
      return this.comfyPage.page.evaluate(() => {
        const graph = window.app!.canvas.graph
        return !!graph && 'inputNode' in graph
      })
    }

    await expect(async () => {
      // Try just clicking the enter button first
      await this.comfyPage.canvas.click({
        position: { x: 250, y: 250 },
        force: true
      })
      await this.comfyPage.nextFrame()

      await this.comfyPage.canvas.click({
        position: subgraphButtonPos,
        force: true
      })
      await this.comfyPage.nextFrame()

      if (await checkIsInSubgraph()) return

      for (const position of clickPositions) {
        // Clear any selection first
        await this.comfyPage.canvas.click({
          position: { x: 250, y: 250 },
          force: true
        })
        await this.comfyPage.nextFrame()

        // Double-click to enter subgraph
        await this.comfyPage.canvas.dblclick({ position, force: true })
        await this.comfyPage.nextFrame()

        if (await checkIsInSubgraph()) return
      }
      throw new Error('Not in subgraph yet')
    }).toPass({ timeout: 5000, intervals: [100, 200, 500] })
  }
}
