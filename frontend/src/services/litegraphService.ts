import _ from 'es-toolkit/compat'

import { downloadFile, openFileInNewTab } from '@/base/common/downloadUtil'
import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems'
import { useSubgraphOperations } from '@/composables/graph/useSubgraphOperations'
import { useNodeAnimatedImage } from '@/composables/node/useNodeAnimatedImage'
import { useNodeCanvasImagePreview } from '@/composables/node/useNodeCanvasImagePreview'
import { useNodeImage, useNodeVideo } from '@/composables/node/useNodeImage'
import {
  addWidgetPromotionOptions,
  isPreviewPseudoWidget
} from '@/core/graph/subgraph/promotionUtils'
import { applyDynamicInputs } from '@/core/graph/widgets/dynamicWidgets'
import { st, t } from '@/i18n'
import {
  LGraphCanvas,
  LGraphEventMode,
  LGraphNode,
  LiteGraph,
  RenderShape,
  SubgraphNode,
  createBounds
} from '@/lib/litegraph/src/litegraph'
import type {
  GraphAddOptions,
  IContextMenuValue,
  Point,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import type {
  ExportedSubgraphInstance,
  ISerialisableNodeInput,
  ISerialisableNodeOutput,
  ISerialisedNode
} from '@/lib/litegraph/src/types/serialisation'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useDialogService } from '@/services/dialogService'
import { resolveSubgraphPseudoWidgetCache } from '@/services/subgraphPseudoWidgetCache'
import type { SubgraphPseudoWidgetCache } from '@/services/subgraphPseudoWidgetCache'
import { transformInputSpecV2ToV1 } from '@/schemas/nodeDef/migration'
import type {
  ComfyNodeDef as ComfyNodeDefV2,
  InputSpec,
  OutputSpec
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import { ComfyApp, app } from '@/scripts/app'
import { isComponentWidget, isDOMWidget } from '@/scripts/domWidget'
import { $el } from '@/scripts/ui'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'
import { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { usePromotionStore } from '@/stores/promotionStore'
import { useSubgraphStore } from '@/stores/subgraphStore'
import { useFavoritedWidgetsStore } from '@/stores/workspace/favoritedWidgetsStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { useWidgetStore } from '@/stores/widgetStore'
import { normalizeI18nKey } from '@/utils/formatUtil'
import {
  isAnimatedOutput,
  isImageNode,
  isVideoNode,
  isVideoOutput,
  migrateWidgetsValues
} from '@/utils/litegraphUtil'
import { getOrderedInputSpecs } from '@/workbench/utils/nodeDefOrderingUtil'

import { useExtensionService } from './extensionService'
import { useMaskEditor } from '@/composables/maskeditor/useMaskEditor'

export interface HasInitialMinSize {
  _initialMinSize: { width: number; height: number }
}

export const CONFIG = Symbol()
export const GET_CONFIG = Symbol()

export function getExtraOptionsForWidget(
  node: LGraphNode,
  widget: IBaseWidget
) {
  const options: IContextMenuValue[] = []
  const input = node.inputs.find((inp) => inp.widget?.name === widget.name)

  if (input) {
    options.unshift({
      content: `${t('contextMenu.RenameWidget')}: ${widget.label ?? widget.name}`,
      callback: async () => {
        const newLabel = await useDialogService().prompt({
          title: t('g.rename'),
          message: t('g.enterNewNamePrompt'),
          defaultValue: widget.label,
          placeholder: widget.name
        })
        if (newLabel === null) return
        widget.label = newLabel || undefined
        input.label = newLabel || undefined
        widget.callback?.(widget.value)
        useCanvasStore().canvas?.setDirty(true)
      }
    })
  }

  const favoritedWidgetsStore = useFavoritedWidgetsStore()
  const isFavorited = favoritedWidgetsStore.isFavorited(node, widget.name)
  options.unshift({
    content: isFavorited
      ? `${t('contextMenu.UnfavoriteWidget')}: ${widget.label ?? widget.name}`
      : `${t('contextMenu.FavoriteWidget')}: ${widget.label ?? widget.name}`,
    callback: () => {
      favoritedWidgetsStore.toggleFavorite(node, widget.name)
    }
  })

  if (node.graph && !node.graph.isRootGraph) {
    addWidgetPromotionOptions(options, widget, node)
  }
  return options
}

/**
 * Service that augments litegraph with ComfyUI specific functionality.
 */
export const useLitegraphService = () => {
  const extensionService = useExtensionService()
  const toastStore = useToastStore()
  const widgetStore = useWidgetStore()
  const canvasStore = useCanvasStore()
  const { toggleSelectedNodesMode } = useSelectedLiteGraphItems()
  const subgraphPseudoWidgetCache = new WeakMap<
    SubgraphNode,
    SubgraphPseudoWidgetCache<LGraphNode, IBaseWidget>
  >()

  function invalidateSubgraphPseudoWidgetCache(node: SubgraphNode) {
    subgraphPseudoWidgetCache.delete(node)
  }

  function getPseudoWidgetPreviewTargets(node: SubgraphNode): LGraphNode[] {
    const promotionStore = usePromotionStore()
    const promotions = promotionStore.getPromotionsRef(
      node.rootGraph.id,
      node.id
    )
    const resolved = resolveSubgraphPseudoWidgetCache({
      cache: subgraphPseudoWidgetCache.get(node) ?? null,
      promotions,
      getNodeById: (nodeId) => node.subgraph.getNodeById(nodeId) ?? undefined,
      isPreviewPseudoWidget
    })
    subgraphPseudoWidgetCache.set(node, resolved.cache)
    return resolved.nodes
  }

  /**
   * @internal The key for the node definition in the i18n file.
   */
  function nodeKey(node: LGraphNode): string {
    return `nodeDefs.${normalizeI18nKey(node.constructor.nodeData!.name)}`
  }
  /**
   * @internal Add input sockets to the node. (No widget)
   */
  function addInputSocket(node: LGraphNode, inputSpec: InputSpec) {
    const inputName = inputSpec.name
    const nameKey = `${nodeKey(node)}.inputs.${normalizeI18nKey(inputName)}.name`
    const widgetConstructor = widgetStore.widgets.get(
      inputSpec.widgetType ?? inputSpec.type
    )
    if (
      (widgetConstructor && !inputSpec.forceInput) ||
      applyDynamicInputs(node, inputSpec)
    )
      return

    const input = node.addInput(inputName, inputSpec.type, {
      shape: inputSpec.isOptional ? RenderShape.HollowCircle : undefined,
      localized_name: st(nameKey, inputName)
    })
    input.label ??= inputSpec.display_name
  }
  /**
   * @internal Setup stroke styles for the node under various conditions.
   */
  function setupStrokeStyles(node: LGraphNode) {
    node.strokeStyles['running'] = function (this: LGraphNode) {
      const nodeId = String(this.id)
      const nodeLocatorId = useWorkflowStore().nodeIdToNodeLocatorId(nodeId)
      const state =
        useExecutionStore().nodeLocationProgressStates[nodeLocatorId]?.state
      if (state === 'running') {
        return { color: '#0f0', lineWidth: 3 }
      }
    }
    node.strokeStyles['dragOver'] = function (this: LGraphNode) {
      if (app.dragOverNode?.id == this.id) {
        return { color: 'dodgerblue' }
      }
    }
    node.strokeStyles['executionError'] = function (this: LGraphNode) {
      if (app.lastExecutionError?.node_id == this.id) {
        return { color: '#f0f', lineWidth: 3 }
      }
    }
  }

  /**
   * Utility function. Implemented for use with dynamic widgets
   */
  function addNodeInput(node: LGraphNode, inputSpec: InputSpec) {
    addInputSocket(node, inputSpec)
    addInputWidget(node, inputSpec)
  }

  /**
   * @internal Add a widget to the node. For both primitive types and custom widgets
   * (unless `socketless`), an input socket is also added.
   */
  function addInputWidget(node: LGraphNode, inputSpec: InputSpec) {
    const widgetInputSpec = { ...inputSpec }
    if (inputSpec.widgetType) {
      widgetInputSpec.type = inputSpec.widgetType
    }
    const inputName = inputSpec.name
    const nameKey = `${nodeKey(node)}.inputs.${normalizeI18nKey(inputName)}.name`
    const widgetConstructor = widgetStore.widgets.get(widgetInputSpec.type)
    if (!widgetConstructor || inputSpec.forceInput) return

    const {
      widget,
      minWidth = 1,
      minHeight = 1
    } = widgetConstructor(
      node,
      inputName,
      transformInputSpecV2ToV1(widgetInputSpec),
      app
    ) ?? {}

    if (widget) {
      widget.label = st(
        nameKey,
        widget.label ?? widgetInputSpec.display_name ?? inputName
      )
      widget.options ??= {}
      Object.assign(widget.options, {
        advanced: inputSpec.advanced,
        hidden: inputSpec.hidden
      })
    }

    if (!widget?.options?.socketless) {
      const inputSpecV1 = transformInputSpecV2ToV1(widgetInputSpec)
      node.addInput(inputName, inputSpec.type, {
        shape: inputSpec.isOptional ? RenderShape.HollowCircle : undefined,
        localized_name: st(nameKey, inputName),
        widget: { name: inputName, [GET_CONFIG]: () => inputSpecV1 }
      })
    }
    const castedNode = node as LGraphNode & HasInitialMinSize
    castedNode._initialMinSize.width = Math.max(
      castedNode._initialMinSize.width,
      minWidth
    )
    castedNode._initialMinSize.height = Math.max(
      castedNode._initialMinSize.height,
      minHeight
    )
  }

  /**
   * @internal Add inputs to the node.
   */
  function addInputs(node: LGraphNode, inputs: Record<string, InputSpec>) {
    // Use input_order if available to ensure consistent widget ordering
    //@ts-expect-error was ComfyNode.nodeData as ComfyNodeDefImpl
    const nodeDefImpl = node.constructor.nodeData as ComfyNodeDefImpl
    const orderedInputSpecs = getOrderedInputSpecs(nodeDefImpl, inputs)

    // Create sockets and widgets in the determined order
    for (const inputSpec of orderedInputSpecs) addInputSocket(node, inputSpec)
    for (const inputSpec of orderedInputSpecs) addInputWidget(node, inputSpec)
  }

  /**
   * @internal Add outputs to the node.
   */
  function addOutputs(node: LGraphNode, outputs: OutputSpec[]) {
    for (const output of outputs) {
      const { name, is_list } = output
      // TODO: Fix the typing at the node spec level
      const type = output.type === 'COMFY_MATCHTYPE_V3' ? '*' : output.type
      const shapeOptions = is_list ? { shape: LiteGraph.GRID_SHAPE } : {}
      const nameKey = `${nodeKey(node)}.outputs.${output.index}.name`
      const typeKey = `dataTypes.${normalizeI18nKey(type)}`
      const outputOptions = {
        ...shapeOptions,
        // If the output name is different from the output type, use the output name.
        // e.g.
        // - type ("INT"); name ("Positive") => translate name
        // - type ("FLOAT"); name ("FLOAT") => translate type
        localized_name: type !== name ? st(nameKey, name) : st(typeKey, name)
      }
      node.addOutput(name, type, outputOptions)
    }
  }

  /**
   * @internal Set the initial size of the node.
   */
  function setInitialSize(node: LGraphNode) {
    const s = node.computeSize()
    // Expand the width a little to fit widget values on screen.
    const pad =
      node.widgets?.length &&
      !useSettingStore().get('LiteGraph.Node.DefaultPadding')
    const castedNode = node as LGraphNode & HasInitialMinSize
    s[0] = Math.max(castedNode._initialMinSize.width, s[0] + (pad ? 60 : 0))
    s[1] = Math.max(castedNode._initialMinSize.height, s[1])
    node.setSize(s)
  }

  function registerSubgraphNodeDef(
    nodeDefV1: ComfyNodeDefV1,
    subgraph: Subgraph,
    instanceData: ExportedSubgraphInstance
  ) {
    const node = class ComfyNode
      extends SubgraphNode
      implements HasInitialMinSize
    {
      static comfyClass: string
      static override title: string
      static override category: string
      static override nodeData: ComfyNodeDefV1 & ComfyNodeDefV2

      _initialMinSize = { width: 1, height: 1 }

      constructor() {
        super(app.rootGraph, subgraph, instanceData)

        // Set up event listener for promoted widget registration
        subgraph.events.addEventListener('widget-promoted', (event) => {
          invalidateSubgraphPseudoWidgetCache(this)
          const { widget } = event.detail
          // Only handle DOM widgets
          if (!isDOMWidget(widget) && !isComponentWidget(widget)) return

          const domWidgetStore = useDomWidgetStore()
          if (!domWidgetStore.widgetStates.has(widget.id)) {
            domWidgetStore.registerWidget(widget)
            // Set initial visibility based on whether the widget's node is in the current graph
            const widgetState = domWidgetStore.widgetStates.get(widget.id)
            if (widgetState) {
              const currentGraph = canvasStore.getCanvas().graph
              widgetState.visible =
                currentGraph?.nodes.includes(widget.node) ?? false
            }
          }
        })

        // Set up event listener for promoted widget removal
        subgraph.events.addEventListener('widget-demoted', (event) => {
          invalidateSubgraphPseudoWidgetCache(this)
          const { widget } = event.detail
          // Only handle DOM widgets
          if (!isDOMWidget(widget) && !isComponentWidget(widget)) return

          const domWidgetStore = useDomWidgetStore()
          if (domWidgetStore.widgetStates.has(widget.id)) {
            domWidgetStore.unregisterWidget(widget.id)
          }
        })

        setupStrokeStyles(this)
        addInputs(this, ComfyNode.nodeData.inputs)
        addOutputs(this, ComfyNode.nodeData.outputs)
        setInitialSize(this)
        this.serialize_widgets = true
        void extensionService.invokeExtensionsAsync('nodeCreated', this)
      }

      /**
       * Configure the node from a serialised node. Keep 'name', 'type', 'shape',
       * and 'localized_name' information from the original node definition.
       */
      override configure(data: ISerialisedNode): void {
        const RESERVED_KEYS = ['name', 'type', 'shape', 'localized_name']

        // Note: input name is unique in a node definition, so we can lookup
        // input by name.
        const inputByName = new Map<string, ISerialisableNodeInput>(
          data.inputs?.map((input) => [input.name, input]) ?? []
        )
        // Inputs defined by the node definition.
        const definedInputNames = new Set(
          this.inputs.map((input) => input.name)
        )
        const definedInputs = this.inputs.map((input) => {
          const inputData = inputByName.get(input.name)
          return inputData
            ? {
                ...inputData,
                // Whether the input has associated widget follows the
                // original node definition.
                ..._.pick(input, RESERVED_KEYS.concat('widget'))
              }
            : input
        })
        // Extra inputs that potentially dynamically added by custom js logic.
        const extraInputs = data.inputs?.filter(
          (input) => !definedInputNames.has(input.name)
        )
        data.inputs = [...definedInputs, ...(extraInputs ?? [])]

        // Note: output name is not unique, so we cannot lookup output by name.
        // Use index instead.
        data.outputs = _.zip(this.outputs, data.outputs).map(
          ([output, outputData]) => {
            // If there are extra outputs in the serialised node, use them directly.
            // There are currently custom nodes that dynamically add outputs via
            // js logic.
            if (!output) return outputData as ISerialisableNodeOutput

            return outputData
              ? {
                  ...outputData,
                  ..._.pick(output, RESERVED_KEYS)
                }
              : output
          }
        )

        data.widgets_values = migrateWidgetsValues(
          ComfyNode.nodeData.inputs,
          this.widgets ?? [],
          data.widgets_values ?? []
        )

        super.configure(data)
      }
    }

    addNodeContextMenuHandler(node)
    addDrawBackgroundHandler(node)
    addNodeKeyHandler(node)
    // Note: Some extensions expects node.comfyClass to be set in
    // `beforeRegisterNodeDef`.
    node.prototype.comfyClass = nodeDefV1.name
    node.comfyClass = nodeDefV1.name

    const nodeDef = new ComfyNodeDefImpl(nodeDefV1)
    node.nodeData = nodeDef
    LiteGraph.registerNodeType(subgraph.id, node)
    // Note: Do not following assignments before `LiteGraph.registerNodeType`
    // because `registerNodeType` will overwrite the assignments.
    node.category = nodeDef.category
    node.skip_list = true
    node.title = nodeDef.display_name || nodeDef.name
  }

  async function registerNodeDef(nodeId: string, nodeDefV1: ComfyNodeDefV1) {
    const node = class ComfyNode
      extends LGraphNode
      implements HasInitialMinSize
    {
      static comfyClass: string
      static override title: string
      static override category: string
      static override nodeData: ComfyNodeDefV1 & ComfyNodeDefV2

      _initialMinSize = { width: 1, height: 1 }

      constructor(title: string) {
        super(title)
        setupStrokeStyles(this)
        addInputs(this, ComfyNode.nodeData.inputs)
        addOutputs(this, ComfyNode.nodeData.outputs)
        setInitialSize(this)
        this.serialize_widgets = true

        // Mark API Nodes yellow by default to distinguish with other nodes.
        if (ComfyNode.nodeData.api_node) {
          this.color = LGraphCanvas.node_colors.yellow.color
          this.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
        }

        void extensionService.invokeExtensionsAsync('nodeCreated', this)
      }

      /**
       * Configure the node from a serialised node. Keep 'name', 'type', 'shape',
       * and 'localized_name' information from the original node definition.
       */
      override configure(data: ISerialisedNode): void {
        const RESERVED_KEYS = ['name', 'type', 'shape', 'localized_name']

        // Note: input name is unique in a node definition, so we can lookup
        // input by name.
        const inputByName = new Map<string, ISerialisableNodeInput>(
          data.inputs?.map((input) => [input.name, input]) ?? []
        )
        // Inputs defined by the node definition.
        const definedInputNames = new Set(
          this.inputs.map((input) => input.name)
        )
        const definedInputs = this.inputs.map((input) => {
          const inputData = inputByName.get(input.name)
          return inputData
            ? {
                ...inputData,
                // Whether the input has associated widget follows the
                // original node definition.
                ..._.pick(input, RESERVED_KEYS.concat('widget'))
              }
            : input
        })
        // Extra inputs that potentially dynamically added by custom js logic.
        const extraInputs = data.inputs?.filter(
          (input) => !definedInputNames.has(input.name)
        )
        data.inputs = [...definedInputs, ...(extraInputs ?? [])]

        // Note: output name is not unique, so we cannot lookup output by name.
        // Use index instead.
        data.outputs = _.zip(this.outputs, data.outputs).map(
          ([output, outputData]) => {
            // If there are extra outputs in the serialised node, use them directly.
            // There are currently custom nodes that dynamically add outputs via
            // js logic.
            if (!output) return outputData as ISerialisableNodeOutput

            return outputData
              ? {
                  ...outputData,
                  ..._.pick(output, RESERVED_KEYS)
                }
              : output
          }
        )

        data.widgets_values = migrateWidgetsValues(
          ComfyNode.nodeData.inputs,
          this.widgets ?? [],
          data.widgets_values ?? []
        )

        super.configure(data)
      }
    }

    addNodeContextMenuHandler(node)
    addDrawBackgroundHandler(node)
    addNodeKeyHandler(node)
    // Note: Some extensions expects node.comfyClass to be set in
    // `beforeRegisterNodeDef`.
    node.prototype.comfyClass = nodeDefV1.name
    node.comfyClass = nodeDefV1.name
    await extensionService.invokeExtensionsAsync(
      'beforeRegisterNodeDef',
      node,
      nodeDefV1 // Receives V1 NodeDef, and potentially make modifications to it
    )

    const nodeDef = new ComfyNodeDefImpl(nodeDefV1)
    node.nodeData = nodeDef
    LiteGraph.registerNodeType(nodeId, node)
    // Note: Do not following assignments before `LiteGraph.registerNodeType`
    // because `registerNodeType` will overwrite the assignments.
    node.category = nodeDef.category
    node.title = nodeDef.display_name || nodeDef.name

    // Set skip_list for dev-only nodes based on current DevMode setting
    // This ensures nodes registered after initial load respect the current setting
    if (nodeDef.dev_only) {
      const settingStore = useSettingStore()
      node.skip_list = !settingStore.get('Comfy.DevMode')
    }
  }

  /**
   * Adds special context menu handling for nodes
   * e.g. this adds Open Image functionality for nodes that show images
   * @param {*} node The node to add the menu handler
   */
  function addNodeContextMenuHandler(node: typeof LGraphNode) {
    function getCopyImageOption(img: HTMLImageElement): IContextMenuValue[] {
      if (typeof window.ClipboardItem === 'undefined') return []
      return [
        {
          content: 'Copy Image',
          callback: async () => {
            const url = new URL(img.src)
            url.searchParams.delete('preview')

            // @ts-expect-error fixme ts strict error
            const writeImage = async (blob) => {
              await navigator.clipboard.write([
                new ClipboardItem({
                  [blob.type]: blob
                })
              ])
            }

            try {
              const data = await fetch(url)
              const blob = await data.blob()
              try {
                await writeImage(blob)
              } catch (error) {
                // Chrome seems to only support PNG on write, convert and try again
                if (blob.type !== 'image/png') {
                  const canvas = $el('canvas', {
                    width: img.naturalWidth,
                    height: img.naturalHeight
                  }) as HTMLCanvasElement
                  const ctx = canvas.getContext('2d')
                  // @ts-expect-error fixme ts strict error
                  let image
                  if (typeof window.createImageBitmap === 'undefined') {
                    image = new Image()
                    const p = new Promise((resolve, reject) => {
                      // @ts-expect-error fixme ts strict error
                      image.onload = resolve
                      // @ts-expect-error fixme ts strict error
                      image.onerror = reject
                    }).finally(() => {
                      // @ts-expect-error fixme ts strict error
                      URL.revokeObjectURL(image.src)
                    })
                    image.src = URL.createObjectURL(blob)
                    await p
                  } else {
                    image = await createImageBitmap(blob)
                  }
                  try {
                    // @ts-expect-error fixme ts strict error
                    ctx.drawImage(image, 0, 0)
                    canvas.toBlob(writeImage, 'image/png')
                  } finally {
                    // @ts-expect-error fixme ts strict error
                    if (typeof image.close === 'function') {
                      // @ts-expect-error fixme ts strict error
                      image.close()
                    }
                  }

                  return
                }
                throw error
              }
            } catch (error) {
              toastStore.addAlert(
                t('toastMessages.errorCopyImage', {
                  // @ts-expect-error fixme ts strict error
                  error: error.message ?? error
                })
              )
            }
          }
        }
      ]
    }

    node.prototype.getExtraMenuOptions = function (canvas, options) {
      if (this.imgs) {
        // If this node has images then we add an open in new tab item
        let img
        if (this.imageIndex != null) {
          // An image is selected so select that
          img = this.imgs[this.imageIndex]
        } else if (this.overIndex != null) {
          // No image is selected but one is hovered
          img = this.imgs[this.overIndex]
        }
        if (img) {
          options.unshift(
            {
              content: 'Open Image',
              callback: () => {
                const url = new URL(img.src)
                url.searchParams.delete('preview')
                void openFileInNewTab(url.toString())
              }
            },
            ...getCopyImageOption(img),
            {
              content: 'Save Image',
              callback: () => {
                const url = new URL(img.src)
                url.searchParams.delete('preview')
                const filename = new URLSearchParams(url.search).get('filename')
                downloadFile(url.toString(), filename ?? undefined)
              }
            }
          )
        }
      }

      options.push({
        content: 'Bypass',
        callback: () => {
          toggleSelectedNodesMode(LGraphEventMode.BYPASS)
          canvas.setDirty(true, true)
        }
      })

      // prevent conflict of clipspace content
      if (!ComfyApp.clipspace_return_node) {
        options.push({
          content: 'Copy (Clipspace)',
          callback: () => {
            ComfyApp.copyToClipspace(this)
          }
        })

        if (ComfyApp.clipspace != null) {
          options.push({
            content: 'Paste (Clipspace)',
            callback: () => {
              ComfyApp.pasteFromClipspace(this)
            }
          })
        }

        if (isImageNode(this)) {
          options.push({
            content: 'Open in MaskEditor | Image Canvas',
            callback: () => {
              useMaskEditor().openMaskEditor(this)
            }
          })
        }
      }
      if (this instanceof SubgraphNode) {
        options.unshift(
          {
            content: 'Edit Subgraph Widgets',
            callback: () => {
              useRightSidePanelStore().openPanel('subgraph')
            }
          },
          {
            content: 'Unpack Subgraph',
            callback: () => {
              const { unpackSubgraph } = useSubgraphOperations()
              unpackSubgraph()
            }
          }
        )
      }
      const [x, y] = canvas.graph_mouse
      const overWidget = this.getWidgetOnPos(x, y, true)
      if (overWidget)
        options.unshift(...getExtraOptionsForWidget(this, overWidget))
      return []
    }
  }
  function updatePreviews(node: LGraphNode, callback?: () => void) {
    try {
      unsafeUpdatePreviews.call(node, callback)
    } catch (error) {
      console.error('Error drawing node background', error)
    }
  }
  function unsafeUpdatePreviews(this: LGraphNode, callback?: () => void) {
    if (this.flags.collapsed) return

    const nodeOutputStore = useNodeOutputStore()
    const { showAnimatedPreview, removeAnimatedPreview } =
      useNodeAnimatedImage()
    const { showCanvasImagePreview, removeCanvasImagePreview } =
      useNodeCanvasImagePreview()

    const output = nodeOutputStore.getNodeOutputs(this)
    const preview = nodeOutputStore.getNodePreviews(this)

    const isNewOutput = output && this.images !== output.images
    const isNewPreview = preview && this.preview !== preview

    if (isNewPreview) this.preview = preview
    if (isNewOutput) this.images = output.images

    if (isNewOutput || isNewPreview) {
      this.animatedImages = isAnimatedOutput(output)

      const isVideo = isVideoOutput(output) || isVideoNode(this)
      if (isVideo) {
        useNodeVideo(this, callback).showPreview()
      } else {
        useNodeImage(this, callback).showPreview()
      }
    }

    // Nothing to do
    if (!this.imgs?.length) return

    if (this.animatedImages) {
      removeCanvasImagePreview(this)
      showAnimatedPreview(this)
    } else {
      removeAnimatedPreview(this)
      showCanvasImagePreview(this)
    }
  }

  /**
   * Adds Custom drawing logic for nodes
   * e.g. Draws images and handles thumbnail navigation on nodes that output images
   * @param {*} node The node to add the draw handler
   */
  function addDrawBackgroundHandler(node: typeof LGraphNode) {
    /**
     * @deprecated No longer needed as we use {@link useImagePreviewWidget}
     */
    node.prototype.setSizeForImage = function (this: LGraphNode) {
      console.warn(
        'node.setSizeForImage is deprecated. Now it has no effect. Please remove the call to it.'
      )
    }
    node.prototype.onDrawBackground = function () {
      updatePreviews(this)

      if (this instanceof SubgraphNode) {
        const parentGraph = this.graph
        for (const interiorNode of getPseudoWidgetPreviewTargets(this)) {
          updatePreviews(interiorNode, () => {
            parentGraph?.setDirtyCanvas(true)
          })
        }
      }
    }
  }

  function addNodeKeyHandler(node: typeof LGraphNode) {
    const origNodeOnKeyDown = node.prototype.onKeyDown

    node.prototype.onKeyDown = function (e) {
      // @ts-expect-error fixme ts strict error
      if (origNodeOnKeyDown && origNodeOnKeyDown.apply(this, e) === false) {
        return false
      }

      if (this.flags.collapsed || !this.imgs || this.imageIndex === null) {
        return
      }

      let handled = false

      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (e.key === 'ArrowLeft') {
          // @ts-expect-error fixme ts strict error
          this.imageIndex -= 1
        } else if (e.key === 'ArrowRight') {
          // @ts-expect-error fixme ts strict error
          this.imageIndex += 1
        }
        // @ts-expect-error fixme ts strict error
        this.imageIndex %= this.imgs.length

        // @ts-expect-error fixme ts strict error
        if (this.imageIndex < 0) {
          // @ts-expect-error fixme ts strict error
          this.imageIndex = this.imgs.length + this.imageIndex
        }
        handled = true
      } else if (e.key === 'Escape') {
        this.imageIndex = null
        handled = true
      }

      if (handled === true) {
        e.preventDefault()
        e.stopImmediatePropagation()
        return false
      }
    }
  }

  function addNodeOnGraph(
    nodeDef: ComfyNodeDefV1 | ComfyNodeDefV2,
    options: Record<string, unknown> & { pos?: Point } = {},
    addOptions?: GraphAddOptions
  ): LGraphNode | null {
    options.pos ??= getCanvasCenter()

    if (nodeDef.name.startsWith(useSubgraphStore().typePrefix)) {
      const canvas = canvasStore.getCanvas()
      const bp = useSubgraphStore().getBlueprint(nodeDef.name)
      const items: object = {
        nodes: bp.nodes,
        subgraphs: bp.definitions?.subgraphs
      }
      const results = canvas._deserializeItems(items, {
        position: options.pos
      })
      if (!results) throw new Error('Failed to add subgraph blueprint')
      const node = results.nodes.values().next().value
      if (!node)
        throw new Error(
          'Subgraph blueprint was added, but failed to resolve a subgraph Node'
        )
      return node
    }

    const node = LiteGraph.createNode(
      nodeDef.name,
      nodeDef.display_name,
      options
    )

    const graph = useWorkflowStore().activeSubgraph ?? app.graph
    if (!graph || !node) return null

    graph.add(node, addOptions)
    return node
  }

  function getCanvasCenter(): Point {
    const dpi = Math.max(window.devicePixelRatio ?? 1, 1)
    const visibleArea = app.canvas?.ds?.visible_area
    if (!visibleArea) {
      return [0, 0]
    }
    const [x, y, w, h] = visibleArea
    return [x + w / dpi / 2, y + h / dpi / 2]
  }

  function goToNode(nodeId: NodeId) {
    const graphNode = app.canvas.graph?.getNodeById(nodeId)
    if (!graphNode) return
    app.canvas.animateToBounds(graphNode.boundingRect)
  }

  function ensureBounds(nodes: LGraphNode[]) {
    for (const node of nodes) {
      if (!node.boundingRect.every((i) => i === 0)) continue
      node.updateArea()
    }
  }

  /**
   * Resets the canvas view to the default
   */
  function resetView() {
    const canvas = canvasStore.canvas
    if (!canvas) return

    canvas.ds.scale = 1
    canvas.ds.offset = [0, 0]
    canvas.setDirty(true, true)
  }

  function fitView() {
    const canvas = canvasStore.getCanvas()
    const nodes = canvas.graph?.nodes
    if (!nodes) return
    ensureBounds(nodes)
    const bounds = createBounds(nodes)
    if (!bounds) return

    canvas.ds.fitToBounds(bounds)
    canvas.setDirty(true, true)
  }

  return {
    registerNodeDef,
    registerSubgraphNodeDef,
    addNodeOnGraph,
    addNodeInput,
    getCanvasCenter,
    getExtraOptionsForWidget,
    goToNode,
    resetView,
    fitView,
    updatePreviews
  }
}
