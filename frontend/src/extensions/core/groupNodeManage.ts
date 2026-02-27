import { PREFIX, SEPARATOR } from '@/constants/groupNodeConstants'
import type {
  GroupNodeConfigEntry,
  GroupNodeWorkflowData,
  LGraphNode,
  LGraphNodeConstructor
} from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'

import { type ComfyApp, app } from '../../scripts/app'
import { $el } from '../../scripts/ui'
import { ComfyDialog } from '../../scripts/ui/dialog'
import { DraggableList } from '../../scripts/ui/draggableList'
import { GroupNodeConfig, GroupNodeHandler } from './groupNode'
import './groupNodeManage.css'

const ORDER: symbol = Symbol()

function merge(
  target: Record<string, unknown>,
  source: Record<string, unknown>
): Record<string, unknown> {
  for (const key in source) {
    const sv = source[key]
    if (typeof sv === 'object' && sv !== null) {
      let tv = target[key] as Record<string, unknown> | undefined
      if (!tv) {
        tv = target[key] = {}
      }
      merge(tv, sv as Record<string, unknown>)
    } else {
      target[key] = sv
    }
  }

  return target
}

export class ManageGroupDialog extends ComfyDialog<HTMLDialogElement> {
  tabs!: Record<
    'Inputs' | 'Outputs' | 'Widgets',
    { tab: HTMLAnchorElement; page: HTMLElement }
  >
  selectedNodeIndex: number | null | undefined
  selectedTab: keyof ManageGroupDialog['tabs'] = 'Inputs'
  selectedGroup: string | undefined
  modifications: Record<
    string,
    Record<
      string,
      Record<
        string,
        { name?: string | undefined; visible?: boolean | undefined }
      >
    >
  > = {}
  nodeItems!: HTMLLIElement[]
  app: ComfyApp
  groupNodeType!: LGraphNodeConstructor<LGraphNode>
  groupData!: GroupNodeConfig

  innerNodesList!: HTMLUListElement
  widgetsPage!: HTMLElement
  inputsPage!: HTMLElement
  outputsPage!: HTMLElement
  draggable: DraggableList | undefined

  get selectedNodeInnerIndex(): number {
    const index = this.selectedNodeIndex
    if (index == null) throw new Error('No node selected')
    const item = this.nodeItems[index]
    if (!item?.dataset.nodeindex) throw new Error('Invalid node item')
    return +item.dataset.nodeindex
  }

  constructor(app: ComfyApp) {
    super()
    this.app = app
    this.element = $el('dialog.comfy-group-manage', {
      parent: document.body
    }) as HTMLDialogElement
  }

  changeTab(tab: keyof ManageGroupDialog['tabs']): void {
    this.tabs[this.selectedTab].tab.classList.remove('active')
    this.tabs[this.selectedTab].page.classList.remove('active')
    this.tabs[tab].tab.classList.add('active')
    this.tabs[tab].page.classList.add('active')
    this.selectedTab = tab
  }

  changeNode(index: number, force?: boolean): void {
    if (!force && this.selectedNodeIndex === index) return

    if (this.selectedNodeIndex != null) {
      this.nodeItems[this.selectedNodeIndex].classList.remove('selected')
    }
    this.nodeItems[index].classList.add('selected')
    this.selectedNodeIndex = index

    if (!this.buildInputsPage() && this.selectedTab === 'Inputs') {
      this.changeTab('Widgets')
    }
    if (!this.buildWidgetsPage() && this.selectedTab === 'Widgets') {
      this.changeTab('Outputs')
    }
    if (!this.buildOutputsPage() && this.selectedTab === 'Outputs') {
      this.changeTab('Inputs')
    }

    this.changeTab(this.selectedTab)
  }

  getGroupData() {
    this.groupNodeType = LiteGraph.registered_node_types[
      `${PREFIX}${SEPARATOR}` + this.selectedGroup
    ] as unknown as LGraphNodeConstructor<LGraphNode>
    this.groupData = GroupNodeHandler.getGroupData(this.groupNodeType)!
  }

  changeGroup(group: string, reset = true): void {
    this.selectedGroup = group
    this.getGroupData()

    const nodes = this.groupData.nodeData.nodes
    this.nodeItems = nodes.map(
      (n, i) =>
        $el(
          'li.draggable-item',
          {
            dataset: {
              nodeindex: n.index + ''
            },
            onclick: () => {
              this.changeNode(i)
            }
          },
          [
            $el('span.drag-handle'),
            $el(
              'div',
              {
                textContent: n.title ?? n.type
              },
              n.title
                ? $el('span', {
                    textContent: n.type
                  })
                : []
            )
          ]
        ) as HTMLLIElement
    )

    this.innerNodesList.replaceChildren(...this.nodeItems)

    if (reset) {
      this.selectedNodeIndex = null
      this.changeNode(0)
    } else {
      const items = this.draggable!.getAllItems()
      let index = items.findIndex((item: Element) =>
        item.classList.contains('selected')
      )
      if (index === -1) index = this.selectedNodeIndex!
      this.changeNode(index, true)
    }

    const ordered = [...nodes]
    this.draggable?.dispose()
    this.draggable = new DraggableList(this.innerNodesList, 'li')
    this.draggable.addEventListener('dragend', (e: Event) => {
      const { oldPosition, newPosition } = (e as CustomEvent).detail
      if (oldPosition === newPosition) return
      ordered.splice(newPosition, 0, ordered.splice(oldPosition, 1)[0])
      for (let i = 0; i < ordered.length; i++) {
        this.storeModification({
          nodeIndex: ordered[i].index,
          section: ORDER,
          prop: 'order',
          value: i
        })
      }
    })
  }

  storeModification(props: {
    nodeIndex?: number
    section: string | symbol
    prop: string
    value: unknown
  }) {
    const { nodeIndex, section, prop, value } = props
    const groupKey = this.selectedGroup!
    const groupMod = (this.modifications[groupKey] ??= {})
    const nodesMod = ((groupMod as Record<string, unknown>).nodes ??=
      {}) as Record<string, Record<symbol | string, Record<string, unknown>>>
    const nodeMod = (nodesMod[nodeIndex ?? this.selectedNodeInnerIndex] ??= {})
    const typeMod = (nodeMod[section] ??= {})
    if (typeof value === 'object' && value !== null) {
      const objMod = (typeMod[prop] ??= {})
      Object.assign(objMod, value)
    } else {
      typeMod[prop] = value
    }
  }

  getEditElement(
    section: string,
    prop: string | number,
    value: unknown,
    placeholder: string,
    checked: boolean,
    checkable = true
  ): HTMLDivElement {
    let displayValue = value === placeholder ? '' : value

    const groupKey = this.selectedGroup!
    const mods = (
      this.modifications[groupKey] as Record<string, unknown> | undefined
    )?.nodes as
      | Record<
          number,
          Record<string, Record<string, { name?: string; visible?: boolean }>>
        >
      | undefined
    const modEntry = mods?.[this.selectedNodeInnerIndex]?.[section]?.[prop]
    if (modEntry) {
      if (modEntry.name != null) {
        displayValue = modEntry.name
      }
      if (modEntry.visible != null) {
        checked = modEntry.visible
      }
    }

    return $el('div', [
      $el('input', {
        value: displayValue as string,
        placeholder,
        type: 'text',
        onchange: (e: Event) => {
          this.storeModification({
            section,
            prop: String(prop),
            value: { name: (e.target as HTMLInputElement).value }
          })
        }
      }),
      $el('label', { textContent: 'Visible' }, [
        $el('input', {
          type: 'checkbox',
          checked,
          disabled: !checkable,
          onchange: (e: Event) => {
            this.storeModification({
              section,
              prop: String(prop),
              value: { visible: !!(e.target as HTMLInputElement).checked }
            })
          }
        })
      ])
    ]) as HTMLDivElement
  }

  buildWidgetsPage() {
    const widgets =
      this.groupData.oldToNewWidgetMap[this.selectedNodeInnerIndex]
    const items = Object.keys(widgets ?? {})
    const type = app.rootGraph.extra.groupNodes![this.selectedGroup!]!
    const config = type.config?.[this.selectedNodeInnerIndex]?.input
    this.widgetsPage.replaceChildren(
      ...items.map((oldName) => {
        return this.getEditElement(
          'input',
          oldName,
          widgets[oldName],
          oldName,
          config?.[oldName]?.visible !== false
        )
      })
    )
    return !!items.length
  }

  buildInputsPage() {
    const inputs = this.groupData.nodeInputs[this.selectedNodeInnerIndex]
    const items = Object.keys(inputs ?? {})
    const type = app.rootGraph.extra.groupNodes![this.selectedGroup!]!
    const config = type.config?.[this.selectedNodeInnerIndex]?.input
    const elements = items
      .map((oldName) => {
        const value = inputs[oldName]
        if (!value) {
          return null
        }

        return this.getEditElement(
          'input',
          oldName,
          value,
          oldName,
          config?.[oldName]?.visible !== false
        )
      })
      .filter((el): el is HTMLDivElement => el !== null)
    this.inputsPage.replaceChildren(...elements)
    return !!items.length
  }

  buildOutputsPage() {
    const nodes = this.groupData.nodeData.nodes
    const innerNodeDef = this.groupData.getNodeDef(
      nodes[this.selectedNodeInnerIndex]
    )
    const outputs = innerNodeDef?.output ?? []
    const groupOutputs =
      this.groupData.oldToNewOutputMap[this.selectedNodeInnerIndex]

    const type = app.rootGraph.extra.groupNodes![this.selectedGroup!]!
    const config = type.config?.[this.selectedNodeInnerIndex]?.output
    const node = this.groupData.nodeData.nodes[this.selectedNodeInnerIndex]
    const checkable = node.type !== 'PrimitiveNode'
    const elements = outputs.map((outputType: unknown, slot: number) => {
      const groupOutputIndex = groupOutputs?.[slot]
      const oldName = innerNodeDef?.output_name?.[slot] ?? String(outputType)
      let value = config?.[slot]?.name
      const visible = config?.[slot]?.visible || groupOutputIndex != null
      if (!value || value === oldName) {
        value = ''
      }
      return this.getEditElement(
        'output',
        slot,
        value,
        oldName,
        visible,
        checkable
      )
    })
    this.outputsPage.replaceChildren(...elements)
    return !!outputs.length
  }

  override show(groupNodeType?: string | HTMLElement | HTMLElement[]): void {
    // Extract string type - this method repurposes the show signature
    const nodeType =
      typeof groupNodeType === 'string' ? groupNodeType : undefined
    const groupNodes = Object.keys(app.rootGraph.extra?.groupNodes ?? {}).sort(
      (a, b) => a.localeCompare(b)
    )

    this.innerNodesList = $el(
      'ul.comfy-group-manage-list-items'
    ) as HTMLUListElement
    this.widgetsPage = $el('section.comfy-group-manage-node-page')
    this.inputsPage = $el('section.comfy-group-manage-node-page')
    this.outputsPage = $el('section.comfy-group-manage-node-page')
    const pages = $el('div', [
      this.widgetsPage,
      this.inputsPage,
      this.outputsPage
    ])

    type TabName = 'Inputs' | 'Widgets' | 'Outputs'
    const tabEntries: [TabName, HTMLElement][] = [
      ['Inputs', this.inputsPage],
      ['Widgets', this.widgetsPage],
      ['Outputs', this.outputsPage]
    ]
    this.tabs = tabEntries.reduce(
      (p, [name, page]) => {
        p[name] = {
          tab: $el('a', {
            onclick: () => {
              this.changeTab(name)
            },
            textContent: name
          }) as HTMLAnchorElement,
          page
        }
        return p
      },
      {} as ManageGroupDialog['tabs']
    )

    const outer = $el('div.comfy-group-manage-outer', [
      $el('header', [
        $el('h2', 'Group Nodes'),
        $el(
          'select',
          {
            onchange: (e: Event) => {
              this.changeGroup((e.target as HTMLSelectElement).value)
            }
          },
          groupNodes.map((g) =>
            $el('option', {
              textContent: g,
              selected: `${PREFIX}${SEPARATOR}${g}` === nodeType,
              value: g
            })
          )
        )
      ]),
      $el('main', [
        $el('section.comfy-group-manage-list', this.innerNodesList),
        $el('section.comfy-group-manage-node', [
          $el(
            'header',
            Object.values(this.tabs).map((t) => t.tab)
          ),
          pages
        ])
      ]),
      $el('footer', [
        $el(
          'button.comfy-btn',
          {
            onclick: () => {
              const node = app.rootGraph.nodes.find(
                (n) => n.type === `${PREFIX}${SEPARATOR}` + this.selectedGroup
              )
              if (node) {
                useToastStore().addAlert(
                  'This group node is in use in the current workflow, please first remove these.'
                )
                return
              }
              if (
                confirm(
                  `Are you sure you want to remove the node: "${this.selectedGroup}"`
                )
              ) {
                delete app.rootGraph.extra.groupNodes![this.selectedGroup!]
                LiteGraph.unregisterNodeType(
                  `${PREFIX}${SEPARATOR}` + this.selectedGroup
                )
              }
              this.show()
            }
          },
          'Delete Group Node'
        ),
        $el(
          'button.comfy-btn',
          {
            onclick: async () => {
              type NodesByType = Record<string, LGraphNode[]>
              let nodesByType: NodesByType | undefined
              const recreateNodes: LGraphNode[] = []
              const types: Record<string, GroupNodeWorkflowData> = {}
              for (const g in this.modifications) {
                const groupNodeData = app.rootGraph.extra.groupNodes![g]!
                let config = (groupNodeData.config ??= {})

                type NodeMods = Record<
                  string,
                  Record<symbol | string, Record<string, unknown>>
                >
                let nodeMods = this.modifications[g]?.nodes as
                  | NodeMods
                  | undefined
                if (nodeMods) {
                  const keys = Object.keys(nodeMods)
                  if (nodeMods[keys[0]]?.[ORDER]) {
                    // If any node is reordered, they will all need sequencing
                    const orderedNodes: GroupNodeWorkflowData['nodes'] = []
                    const orderedMods: NodeMods = {}
                    const orderedConfig: Record<number, GroupNodeConfigEntry> =
                      {}

                    for (const n of keys) {
                      const order = (nodeMods[n][ORDER] as { order: number })
                        .order
                      orderedNodes[order] = groupNodeData.nodes[+n]
                      orderedMods[order] = nodeMods[n]
                      orderedNodes[order].index = order
                    }

                    // Rewrite links
                    const nodesLen = groupNodeData.nodes.length
                    for (const l of groupNodeData.links) {
                      const srcIdx = l[0] as number
                      const dstIdx = l[2] as number
                      if (srcIdx != null && srcIdx < nodesLen)
                        l[0] = groupNodeData.nodes[srcIdx].index!
                      if (dstIdx != null && dstIdx < nodesLen)
                        l[2] = groupNodeData.nodes[dstIdx].index!
                    }

                    // Rewrite externals
                    if (groupNodeData.external) {
                      for (const ext of groupNodeData.external) {
                        const extIdx = ext[0] as number
                        if (extIdx != null && extIdx < nodesLen) {
                          ext[0] = groupNodeData.nodes[extIdx].index!
                        }
                      }
                    }

                    // Rewrite modifications
                    for (const id of keys) {
                      if (config[+id]) {
                        orderedConfig[groupNodeData.nodes[+id].index!] =
                          config[+id]
                      }
                      delete config[+id]
                    }

                    groupNodeData.nodes = orderedNodes
                    nodeMods = orderedMods
                    groupNodeData.config = config = orderedConfig
                  }

                  merge(
                    config as Record<string, unknown>,
                    nodeMods as Record<string, unknown>
                  )
                }

                types[g] = groupNodeData

                if (!nodesByType) {
                  nodesByType = app.rootGraph.nodes.reduce<NodesByType>(
                    (p, n) => {
                      const nodeType = n.type ?? ''
                      p[nodeType] ??= []
                      p[nodeType].push(n)
                      return p
                    },
                    {}
                  )
                }

                const groupTypeNodes = nodesByType[`${PREFIX}${SEPARATOR}` + g]
                if (groupTypeNodes) recreateNodes.push(...groupTypeNodes)
              }

              await GroupNodeConfig.registerFromWorkflow(types, [])

              for (const node of recreateNodes) {
                node.recreate?.()
              }

              this.modifications = {}
              this.app.canvas.setDirty(true, true)
              this.changeGroup(this.selectedGroup!, false)
            }
          },
          'Save'
        ),
        $el(
          'button.comfy-btn',
          { onclick: () => this.element.close() },
          'Close'
        )
      ])
    ])

    this.element.replaceChildren(outer)
    this.changeGroup(
      nodeType
        ? (groupNodes.find((g) => `${PREFIX}${SEPARATOR}${g}` === nodeType) ??
            groupNodes[0])
        : groupNodes[0]
    )
    this.element.showModal()

    this.element.addEventListener('close', () => {
      this.draggable?.dispose()
      this.element.remove()
    })
  }
}
