import type {
  IContextMenuValue,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import {
  LGraphCanvas,
  LGraphGroup,
  type LGraphNode
} from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { ComfyExtension } from '@/types/comfy'

import { app } from '../../scripts/app'

function setNodeMode(node: LGraphNode, mode: number) {
  node.mode = mode
  node.graph?.change()
}

function addNodesToGroup(group: LGraphGroup, items: Iterable<Positionable>) {
  const padding = useSettingStore().get('Comfy.GroupSelectedNodes.Padding')
  group.resizeTo([...group.children, ...items], padding)
}

const ext: ComfyExtension = {
  name: 'Comfy.GroupOptions',

  getCanvasMenuItems(canvas: LGraphCanvas): IContextMenuValue[] {
    const items: IContextMenuValue[] = []

    // @ts-expect-error fixme ts strict error
    const group = canvas.graph.getGroupOnPos(
      canvas.graph_mouse[0],
      canvas.graph_mouse[1]
    )

    if (!group) {
      if (canvas.selectedItems.size > 0) {
        items.push({
          content: 'Add Group For Selected Nodes',
          callback: () => {
            const group = new LGraphGroup()
            addNodesToGroup(group, canvas.selectedItems)
            // @ts-expect-error fixme ts strict error
            canvas.graph.add(group)
            // @ts-expect-error fixme ts strict error
            canvas.graph.change()

            group.recomputeInsideNodes()
          }
        })
      }

      return items
    }

    // Group nodes aren't recomputed until the group is moved, this ensures the nodes are up-to-date
    group.recomputeInsideNodes()
    const nodesInGroup = group.nodes

    items.push({
      content: 'Add Selected Nodes To Group',
      disabled: !canvas.selectedItems?.size,
      callback: () => {
        addNodesToGroup(group, canvas.selectedItems)
        // @ts-expect-error fixme ts strict error
        canvas.graph.change()
      }
    })

    // No nodes in group, return default options
    if (nodesInGroup.length === 0) {
      return items
    } else {
      // Add a separator between the default options and the group options
      // @ts-expect-error fixme ts strict error
      items.push(null)
    }

    // Check if all nodes are the same mode
    let allNodesAreSameMode = true
    for (let i = 1; i < nodesInGroup.length; i++) {
      if (nodesInGroup[i].mode !== nodesInGroup[0].mode) {
        allNodesAreSameMode = false
        break
      }
    }

    items.push({
      content: 'Fit Group To Nodes',
      callback: () => {
        group.recomputeInsideNodes()
        const padding = useSettingStore().get(
          'Comfy.GroupSelectedNodes.Padding'
        )
        group.resizeTo(group.children, padding)
        // @ts-expect-error fixme ts strict error
        canvas.graph.change()
      }
    })

    items.push({
      content: 'Select Nodes',
      callback: () => {
        canvas.selectNodes(nodesInGroup)
        // @ts-expect-error fixme ts strict error
        canvas.graph.change()
        canvas.canvas.focus()
      }
    })

    // Modes
    // 0: Always
    // 1: On Event
    // 2: Never
    // 3: On Trigger
    // 4: Bypass
    // If all nodes are the same mode, add a menu option to change the mode
    if (allNodesAreSameMode) {
      const mode = nodesInGroup[0].mode
      switch (mode) {
        case 0:
          // All nodes are always, option to disable, and bypass
          items.push({
            content: 'Set Group Nodes to Never',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 2)
              }
            }
          })
          items.push({
            content: 'Bypass Group Nodes',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 4)
              }
            }
          })
          break
        case 2:
          // All nodes are never, option to enable, and bypass
          items.push({
            content: 'Set Group Nodes to Always',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 0)
              }
            }
          })
          items.push({
            content: 'Bypass Group Nodes',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 4)
              }
            }
          })
          break
        case 4:
          // All nodes are bypass, option to enable, and disable
          items.push({
            content: 'Set Group Nodes to Always',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 0)
              }
            }
          })
          items.push({
            content: 'Set Group Nodes to Never',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 2)
              }
            }
          })
          break
        default:
          // All nodes are On Trigger or On Event(Or other?), option to disable, set to always, or bypass
          items.push({
            content: 'Set Group Nodes to Always',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 0)
              }
            }
          })
          items.push({
            content: 'Set Group Nodes to Never',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 2)
              }
            }
          })
          items.push({
            content: 'Bypass Group Nodes',
            callback: () => {
              for (const node of nodesInGroup) {
                setNodeMode(node, 4)
              }
            }
          })
          break
      }
    } else {
      // Nodes are not all the same mode, add a menu option to change the mode to always, never, or bypass
      items.push({
        content: 'Set Group Nodes to Always',
        callback: () => {
          for (const node of nodesInGroup) {
            setNodeMode(node, 0)
          }
        }
      })
      items.push({
        content: 'Set Group Nodes to Never',
        callback: () => {
          for (const node of nodesInGroup) {
            setNodeMode(node, 2)
          }
        }
      })
      items.push({
        content: 'Bypass Group Nodes',
        callback: () => {
          for (const node of nodesInGroup) {
            setNodeMode(node, 4)
          }
        }
      })
    }

    return items
  }
}

app.registerExtension(ext)
