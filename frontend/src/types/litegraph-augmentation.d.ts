import '@/lib/litegraph/src/litegraph'
import type {
  ExecutableLGraphNode,
  ExecutionId,
  LLink,
  Size
} from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  TWidgetValue
} from '@/lib/litegraph/src/types/widgets'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { NodeExecutionOutput } from '@/schemas/apiSchema'
import type { ComfyNodeDef as ComfyNodeDefV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import type { DOMWidget, DOMWidgetOptions } from '@/scripts/domWidget'

/** ComfyUI extensions of litegraph */
declare module '@/lib/litegraph/src/types/widgets' {
  interface IWidgetOptions {
    /** Currently used by DOM widgets only.  Declaring here reduces complexity. */
    onHide?: (widget: DOMWidget) => void
    /**
     * Controls whether the widget's value is included in the API workflow/prompt.
     * - If false, the value will be excluded from the API workflow but still serialized as part of the graph state
     * - If true or undefined, the value will be included in both the API workflow and graph state
     * @default true
     * @use {@link IBaseWidget.serialize} if you don't want the widget value to be included in both
     * the API workflow and graph state.
     */
    serialize?: boolean
    /**
     * Rounding value for numeric float widgets.
     */
    round?: number
    /**
     * The minimum size of the node if the widget is present.
     */
    minNodeSize?: Size

    /** If the widget is advanced, this will be set to true. */
    advanced?: boolean

    /** If the widget is hidden, this will be set to true. */
    hidden?: boolean
  }

  interface WidgetCallbackOptions {
    isPartialExecution?: boolean
  }

  interface IBaseWidget {
    onRemove?(): void
    beforeQueued?(options?: WidgetCallbackOptions): unknown
    afterQueued?(options?: WidgetCallbackOptions): unknown
    serializeValue?(node: LGraphNode, index: number): Promise<unknown> | unknown

    /**
     * Refreshes the widget's value or options from its remote source.
     */
    refresh?(): unknown

    /**
     * If the widget supports dynamic prompts, this will be set to true.
     * See extensions/core/dynamicPrompts.ts
     */
    dynamicPrompts?: boolean
  }
}

/**
 * ComfyUI extensions of litegraph interfaces
 */
declare module '@/lib/litegraph/src/interfaces' {
  interface IWidgetLocator {
    [key: symbol]: unknown
  }
}

/**
 *  ComfyUI extensions of litegraph
 */
declare module '@/lib/litegraph/src/litegraph' {
  interface LGraphNodeConstructor<T extends LGraphNode = LGraphNode> {
    type?: string
    comfyClass: string
    title: string
    nodeData?: ComfyNodeDefV1 & ComfyNodeDefV2 & { [key: symbol]: unknown }
    category?: string
    new (): T
  }

  // Add interface augmentations into the class itself

  interface BaseWidget extends IBaseWidget {}

  interface LGraphNode {
    constructor: LGraphNodeConstructor

    /**
     * Callback fired on each node after the graph is configured
     */
    onAfterGraphConfigured?(): void
    onGraphConfigured?(): void
    /**
     * Callback fired when node execution completes.
     * Output contains known media properties (images, audio, video) plus
     * arbitrary node-specific outputs (text, ui, custom properties).
     */
    onExecuted?(output: NodeExecutionOutput): void
    onNodeCreated?(this: LGraphNode): void
    /** @deprecated groupNode */
    setInnerNodes?(nodes: LGraphNode[]): void
    /** Originally a group node API. */
    getInnerNodes?(
      nodesByExecutionId: Map<ExecutionId, ExecutableLGraphNode>,
      subgraphNodePath?: readonly NodeId[],
      nodes?: ExecutableLGraphNode[],
      subgraphs?: Set<LGraphNode>
    ): ExecutableLGraphNode[]
    /** @deprecated groupNode */
    convertToNodes?(): LGraphNode[]
    recreate?(): Promise<LGraphNode>
    refreshComboInNode?(defs: Record<string, ComfyNodeDef>)
    /** @deprecated groupNode */
    updateLink?(link: LLink): LLink | null
    /**
     * @deprecated primitive node.
     * Used by virtual nodes (primitives) to insert their values into the graph prior to queueing.
     * Externally used by
     * - https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/bbda5e52ad580c13ceaa53136d9c2bed9137bd2e/web/js/presetText.js#L160-L182
     * - https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/4c7858ddd5126f7293dc3c9f6e0fc4c263cde079/web/js/VHS.core.js#L1889-L1889
     */
    applyToGraph?(extraLinks?: LLink[]): void
    onExecutionStart?(): unknown
    /**
     * Callback invoked when the node is dragged over from an external source, i.e.
     * a file or another HTML element.
     * @param e The drag event
     * @returns {boolean} True if the drag event should be handled by this node, false otherwise
     */
    onDragOver?(e: DragEvent): boolean
    /**
     * Callback invoked when the node is dropped from an external source, i.e.
     * a file or another HTML element.
     * @param e The drag event
     * @returns {boolean} True if the drag event should be handled by this node, false otherwise
     */
    onDragDrop?(e: DragEvent): Promise<boolean> | boolean

    index?: number
    runningInternalNodeId?: NodeId

    comfyClass?: string

    /**
     * If the node is a frontend only node and should not be serialized into the prompt.
     */
    isVirtualNode?: boolean

    addDOMWidget<
      T extends HTMLElement = HTMLElement,
      V extends object | string = string
    >(
      name: string,
      type: string,
      element: T,
      options?: DOMWidgetOptions<V>
    ): DOMWidget<T, V>

    animatedImages?: boolean
    imgs?: HTMLImageElement[]
    images?: ExecutedWsMessage['output']
    /** Container for the node's video preview */
    videoContainer?: HTMLElement
    /** Whether the node's preview media is loading */
    isLoading?: boolean
    /** Whether a file is being uploaded to this node */
    isUploading?: boolean
    /** The content type of the node's preview media */
    previewMediaType?: 'image' | 'video' | 'audio' | 'model'
    /** If true, output images are stored but not rendered below the node */
    hideOutputImages?: boolean

    preview: string[]
    /** Index of the currently selected image on a multi-image node such as Preview Image */
    imageIndex?: number | null
    imageRects: Rect[]
    overIndex?: number | null
    pointerDown?: { index: number | null; pos: Point } | null
    /**
     * @deprecated No longer needed as we use {@link useImagePreviewWidget}
     */
    setSizeForImage?(force?: boolean): void
    /** @deprecated Unused */
    inputHeight?: unknown

    /** The y offset of the image preview to the top of the node body. */
    imageOffset?: number
    /** Callback for pasting an image file into the node */
    pasteFile?(file: File): void
    /** Callback for pasting multiple files into the node */
    pasteFiles?(files: File[]): void
  }
  /**
   * Only used by the Primitive node. Primitive node is using the widget property
   * to store/access the widget config.
   * We should remove this hacky solution once we have a proper solution.
   */
  interface INodeOutputSlot {
    widget?: { name: string; [key: symbol]: unknown }
  }
}

/**
 * Extended types for litegraph, to be merged upstream once it has stabilized.
 */
declare module '@/lib/litegraph/src/litegraph' {
  /**
   * widgets_values is set to LGraphNode by `LGraphNode.configure`, but it is not
   * used by litegraph internally. We should remove the dependency on it later.
   */
  interface LGraphNode {
    widgets_values?: TWidgetValue[]
  }
}
