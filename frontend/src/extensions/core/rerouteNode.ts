import type { IContextMenuValue } from '@/lib/litegraph/src/litegraph'
import {
  LGraphCanvas,
  LGraphNode,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'
import type { ISlotType } from '@/lib/litegraph/src/interfaces'

import { app } from '../../scripts/app'
import { getWidgetConfig, mergeIfValid, setWidgetConfig } from './widgetInputs'

// Node that allows you to redirect connections for cleaner graphs

app.registerExtension({
  name: 'Comfy.RerouteNode',
  registerCustomNodes() {
    interface RerouteNode extends LGraphNode {
      __outputType?: string | number
    }

    class RerouteNode extends LGraphNode {
      static override category: string | undefined
      static defaultVisibility = false

      constructor(title?: string) {
        super(title ?? '')
        if (!this.properties) {
          this.properties = {}
        }
        this.properties.showOutputText = RerouteNode.defaultVisibility
        this.properties.horizontal = false

        this.addInput('', '*')
        this.addOutput(this.properties.showOutputText ? '*' : '', '*')

        // This node is purely frontend and does not impact the resulting prompt so should not be serialized
        this.isVirtualNode = true
      }
      override onAfterGraphConfigured() {
        requestAnimationFrame(() => {
          this.onConnectionsChange(LiteGraph.INPUT, undefined, true)
        })
      }
      override clone(): LGraphNode | null {
        const cloned = super.clone()
        if (!cloned) return cloned
        cloned.removeOutput(0)
        cloned.addOutput(this.properties.showOutputText ? '*' : '', '*')
        cloned.setSize(cloned.computeSize())
        return cloned
      }
      override onConnectionsChange(
        type: ISlotType,
        _index: number | undefined,
        connected: boolean
      ) {
        const { graph } = this
        if (!graph) return
        if (app.configuringGraph) return

        // Prevent multiple connections to different types when we have no input
        if (connected && type === LiteGraph.OUTPUT) {
          // Ignore wildcard nodes as these will be updated to real types
          const types = new Set(
            this.outputs[0].links
              ?.map((l) => graph.links[l]?.type)
              ?.filter((t) => t && t !== '*') ?? []
          )
          if (types.size > 1) {
            const linksToDisconnect = []
            for (const linkId of this.outputs[0].links ?? []) {
              const link = graph.links[linkId]
              linksToDisconnect.push(link)
            }
            linksToDisconnect.pop()
            for (const link of linksToDisconnect) {
              const node = graph.getNodeById(link.target_id)
              node?.disconnectInput(link.target_slot)
            }
          }
        }

        // Find root input
        let currentNode: RerouteNode | null = this
        let updateNodes: RerouteNode[] = []
        let inputType = null
        let inputNode = null
        while (currentNode) {
          updateNodes.unshift(currentNode)
          const linkId = currentNode.inputs[0].link
          if (linkId !== null) {
            const link = graph.links[linkId]
            if (!link) return
            const node = graph.getNodeById(link.origin_id)
            if (!node) return
            if (node instanceof RerouteNode) {
              if (node === this) {
                // We've found a circle
                currentNode.disconnectInput(link.target_slot)
                currentNode = null
              } else {
                // Move the previous node
                currentNode = node
              }
            } else {
              // We've found the end
              inputNode = currentNode
              inputType = node.outputs[link.origin_slot]?.type ?? null
              break
            }
          } else {
            // This path has no input node
            currentNode = null
            break
          }
        }

        // Find all outputs
        const nodes: RerouteNode[] = [this]
        let outputType = null
        while (nodes.length) {
          currentNode = nodes.pop()!
          const outputs = currentNode.outputs?.[0]?.links ?? []
          for (const linkId of outputs) {
            const link = graph.links[linkId]

            // When disconnecting sometimes the link is still registered
            if (!link) continue

            const node = graph.getNodeById(link.target_id)
            if (!node) continue
            if (node instanceof RerouteNode) {
              // Follow reroute nodes
              nodes.push(node)
              updateNodes.push(node)
            } else {
              // We've found an output
              const nodeInput = node.inputs[link.target_slot]
              const nodeOutType = nodeInput.type
              const keep =
                !inputType ||
                !nodeOutType ||
                LiteGraph.isValidConnection(inputType, nodeOutType)
              if (!keep) {
                // The output doesnt match our input so disconnect it
                node.disconnectInput(link.target_slot)
                continue
              }
              node.onConnectionsChange?.(
                LiteGraph.INPUT,
                link.target_slot,
                keep,
                link,
                nodeInput
              )
              outputType = node.inputs[link.target_slot].type
            }
          }
        }

        const displayType = inputType || outputType || '*'
        const color = LGraphCanvas.link_type_colors[displayType]

        let widgetConfig
        let widgetType
        // Update the types of each node
        for (const node of updateNodes) {
          // If we dont have an input type we are always wildcard but we'll show the output type
          // This lets you change the output link to a different type and all nodes will update
          node.outputs[0].type = inputType || '*'
          node.__outputType = displayType
          node.outputs[0].name = node.properties.showOutputText
            ? `${displayType}`
            : ''
          node.setSize(node.computeSize())

          for (const l of node.outputs[0].links || []) {
            const link = graph.links[l]
            if (!link) continue
            link.color = color

            if (app.configuringGraph) continue
            const targetNode = graph.getNodeById(link.target_id)
            if (!targetNode) continue
            const targetInput = targetNode.inputs?.[link.target_slot]
            if (targetInput?.widget) {
              const config = getWidgetConfig(targetInput)
              if (!widgetConfig) {
                widgetConfig = config[1] ?? {}
                widgetType = config[0]
              }

              const merged = mergeIfValid(targetInput, [
                config[0],
                widgetConfig
              ])
              if (merged.customConfig) {
                widgetConfig = merged.customConfig
              }
            }
          }
        }

        for (const node of updateNodes) {
          if (widgetConfig && outputType) {
            node.inputs[0].widget = { name: 'value' }
            setWidgetConfig(node.inputs[0], [
              widgetType ?? `${displayType}`,
              widgetConfig
            ])
          } else {
            setWidgetConfig(node.inputs[0], undefined)
          }
        }

        if (inputNode?.inputs?.[0]?.link) {
          const link = graph.links[inputNode.inputs[0].link]
          if (link) {
            link.color = color
          }
        }
      }

      override getExtraMenuOptions(
        _: unknown,
        options: (IContextMenuValue | null)[]
      ): IContextMenuValue[] {
        options.unshift(
          {
            content:
              (this.properties.showOutputText ? 'Hide' : 'Show') + ' Type',
            callback: () => {
              this.properties.showOutputText = !this.properties.showOutputText
              if (this.properties.showOutputText) {
                this.outputs[0].name = `${this.__outputType || this.outputs[0].type}`
              } else {
                this.outputs[0].name = ''
              }
              this.setSize(this.computeSize())
              app.canvas.setDirty(true, true)
            }
          },
          {
            content:
              (RerouteNode.defaultVisibility ? 'Hide' : 'Show') +
              ' Type By Default',
            callback: () => {
              RerouteNode.setDefaultTextVisibility(
                !RerouteNode.defaultVisibility
              )
            }
          }
        )
        return []
      }
      override computeSize(): [number, number] {
        return [
          this.properties.showOutputText && this.outputs && this.outputs.length
            ? Math.max(
                75,
                LiteGraph.NODE_TEXT_SIZE * this.outputs[0].name.length * 0.6 +
                  40
              )
            : 75,
          26
        ]
      }

      static setDefaultTextVisibility(visible: boolean) {
        RerouteNode.defaultVisibility = visible
        if (visible) {
          localStorage['Comfy.RerouteNode.DefaultVisibility'] = 'true'
        } else {
          delete localStorage['Comfy.RerouteNode.DefaultVisibility']
        }
      }
    }

    // Load default visibility
    RerouteNode.setDefaultTextVisibility(
      !!localStorage['Comfy.RerouteNode.DefaultVisibility']
    )

    LiteGraph.registerNodeType(
      'Reroute',
      Object.assign(RerouteNode, {
        title_mode: LiteGraph.NO_TITLE,
        title: 'Reroute',
        collapsable: false
      })
    )

    RerouteNode.category = 'utils'
  }
})
