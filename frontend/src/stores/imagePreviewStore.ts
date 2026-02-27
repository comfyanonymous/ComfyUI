import { useTimeoutFn } from '@vueuse/core'
import { defineStore } from 'pinia'
import { ref } from 'vue'

import type { LGraphNode, SubgraphNode } from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type {
  ExecutedWsMessage,
  ResultItem,
  ResultItemType
} from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import type { NodeLocatorId } from '@/types/nodeIdentification'
import { parseFilePath } from '@/utils/formatUtil'
import { isAnimatedOutput, isVideoNode } from '@/utils/litegraphUtil'
import {
  releaseSharedObjectUrl,
  retainSharedObjectUrl
} from '@/utils/objectUrlUtil'
import { executionIdToNodeLocatorId } from '@/utils/graphTraversalUtil'

const PREVIEW_REVOKE_DELAY_MS = 400

const createOutputs = (
  filenames: string[],
  type: ResultItemType,
  isAnimated: boolean
): ExecutedWsMessage['output'] => {
  return {
    images: filenames.map((image) => ({ type, ...parseFilePath(image) })),
    animated: filenames.map(
      (image) =>
        isAnimated && (image.endsWith('.webp') || image.endsWith('.png'))
    )
  }
}

interface SetOutputOptions {
  merge?: boolean
}

export const useNodeOutputStore = defineStore('nodeOutput', () => {
  const { nodeIdToNodeLocatorId, nodeToNodeLocatorId } = useWorkflowStore()
  const scheduledRevoke: Record<NodeLocatorId, { stop: () => void }> = {}
  const latestPreview = ref<string[]>([])

  function scheduleRevoke(locator: NodeLocatorId, cb: () => void) {
    scheduledRevoke[locator]?.stop()

    const { stop } = useTimeoutFn(() => {
      delete scheduledRevoke[locator]
      cb()
    }, PREVIEW_REVOKE_DELAY_MS)

    scheduledRevoke[locator] = { stop }
  }

  const nodeOutputs = ref<Record<string, ExecutedWsMessage['output']>>({})

  // Reactive state for node preview images - mirrors app.nodePreviewImages
  const nodePreviewImages = ref<Record<string, string[]>>(
    app.nodePreviewImages || {}
  )

  function getNodeOutputs(
    node: LGraphNode
  ): ExecutedWsMessage['output'] | undefined {
    return app.nodeOutputs[nodeToNodeLocatorId(node)]
  }

  function getNodePreviews(node: LGraphNode): string[] | undefined {
    return app.nodePreviewImages[nodeToNodeLocatorId(node)]
  }

  /**
   * Check if a node's outputs includes images that should/can be loaded normally
   * by PIL.
   */
  const isImageOutputs = (
    node: LGraphNode,
    outputs: ExecutedWsMessage['output']
  ): boolean => {
    // If animated webp/png or video outputs, return false
    if (isAnimatedOutput(outputs) || isVideoNode(node)) return false

    // If no images, return false
    if (!outputs?.images?.length) return false

    // If svg images, return false
    if (outputs.images.some((image) => image.filename?.endsWith('svg')))
      return false

    return true
  }

  /**
   * Get the preview param for the node's outputs.
   *
   * If the output is an image, use the user's preferred format (from settings).
   * For non-image outputs, return an empty string, as including the preview param
   * will force the server to load the output file as an image.
   */
  function getPreviewParam(
    node: LGraphNode,
    outputs: ExecutedWsMessage['output']
  ): string {
    return isImageOutputs(node, outputs) ? app.getPreviewFormatParam() : ''
  }

  function getNodeImageUrls(node: LGraphNode): string[] | undefined {
    const previews = getNodePreviews(node)
    if (previews?.length) return previews

    const outputs = getNodeOutputs(node)
    if (!outputs?.images?.length) return

    const rand = app.getRandParam()
    const previewParam = getPreviewParam(node, outputs)

    return outputs.images.map((image) => {
      const imgUrlPart = new URLSearchParams(image)
      return api.apiURL(`/view?${imgUrlPart}${previewParam}${rand}`)
    })
  }

  /**
   * Internal function to set outputs by NodeLocatorId.
   * Handles the merge logic when needed.
   */
  function setOutputsByLocatorId(
    nodeLocatorId: NodeLocatorId,
    outputs: ExecutedWsMessage['output'] | ResultItem,
    options: SetOutputOptions = {}
  ) {
    // Skip if outputs is null/undefined - preserve existing output
    // This can happen when backend returns null for cached/deduplicated nodes
    // (e.g., two LoadImage nodes selecting the same image)
    if (outputs == null) return

    if (options.merge) {
      const existingOutput = app.nodeOutputs[nodeLocatorId]
      if (existingOutput && outputs) {
        for (const k in outputs) {
          const existingValue = existingOutput[k]
          const newValue = (outputs as Record<NodeLocatorId, unknown>)[k]

          if (Array.isArray(existingValue) && Array.isArray(newValue)) {
            existingOutput[k] = existingValue.concat(newValue)
          } else {
            existingOutput[k] = newValue
          }
        }
        nodeOutputs.value[nodeLocatorId] = { ...existingOutput }
        return
      }
    }

    app.nodeOutputs[nodeLocatorId] = outputs
    nodeOutputs.value[nodeLocatorId] = outputs
  }

  function setNodeOutputs(
    node: LGraphNode,
    filenames: string | string[] | ResultItem,
    {
      folder = 'input',
      isAnimated = false
    }: { folder?: ResultItemType; isAnimated?: boolean } = {}
  ) {
    if (!filenames || !node) return

    const locatorId = nodeToNodeLocatorId(node)
    if (!locatorId) return
    if (typeof filenames === 'string') {
      setOutputsByLocatorId(
        locatorId,
        createOutputs([filenames], folder, isAnimated)
      )
    } else if (!Array.isArray(filenames)) {
      setOutputsByLocatorId(locatorId, filenames)
    } else {
      const resultItems = createOutputs(filenames, folder, isAnimated)
      if (!resultItems?.images?.length) return
      setOutputsByLocatorId(locatorId, resultItems)
    }
  }

  /**
   * Set node outputs by execution ID (hierarchical ID from backend).
   * Converts the execution ID to a NodeLocatorId before storing.
   *
   * @param executionId - The execution ID (e.g., "123:456:789" or "789")
   * @param outputs - The outputs to store
   * @param options - Options for setting outputs
   * @param options.merge - If true, merge with existing outputs (arrays are concatenated)
   */
  function setNodeOutputsByExecutionId(
    executionId: string,
    outputs: ExecutedWsMessage['output'] | ResultItem,
    options: SetOutputOptions = {}
  ) {
    const nodeLocatorId = executionIdToNodeLocatorId(app.rootGraph, executionId)
    if (!nodeLocatorId) return

    setOutputsByLocatorId(nodeLocatorId, outputs, options)
  }

  /**
   * Set node preview images by execution ID (hierarchical ID from backend).
   * Converts the execution ID to a NodeLocatorId before storing.
   *
   * @param executionId - The execution ID (e.g., "123:456:789" or "789")
   * @param previewImages - Array of preview image URLs to store
   */
  function setNodePreviewsByExecutionId(
    executionId: string,
    previewImages: string[]
  ) {
    const nodeLocatorId = executionIdToNodeLocatorId(app.rootGraph, executionId)
    if (!nodeLocatorId) return
    const existingPreviews = app.nodePreviewImages[nodeLocatorId]
    if (scheduledRevoke[nodeLocatorId]) {
      scheduledRevoke[nodeLocatorId].stop()
      delete scheduledRevoke[nodeLocatorId]
    }
    if (existingPreviews?.[Symbol.iterator]) {
      for (const url of existingPreviews) {
        releaseSharedObjectUrl(url)
      }
    }
    for (const url of previewImages) {
      retainSharedObjectUrl(url)
    }
    latestPreview.value = previewImages
    app.nodePreviewImages[nodeLocatorId] = previewImages
    nodePreviewImages.value[nodeLocatorId] = previewImages
  }

  /**
   * Set node preview images by node ID.
   * Uses the current graph context to create the appropriate NodeLocatorId.
   *
   * @param nodeId - The node ID
   * @param previewImages - Array of preview image URLs to store
   */
  function setNodePreviewsByNodeId(
    nodeId: string | number,
    previewImages: string[]
  ) {
    const nodeLocatorId = nodeIdToNodeLocatorId(nodeId)
    const existingPreviews = app.nodePreviewImages[nodeLocatorId]
    if (scheduledRevoke[nodeLocatorId]) {
      scheduledRevoke[nodeLocatorId].stop()
      delete scheduledRevoke[nodeLocatorId]
    }
    if (existingPreviews?.[Symbol.iterator]) {
      for (const url of existingPreviews) {
        releaseSharedObjectUrl(url)
      }
    }
    for (const url of previewImages) {
      retainSharedObjectUrl(url)
    }
    app.nodePreviewImages[nodeLocatorId] = previewImages
    nodePreviewImages.value[nodeLocatorId] = previewImages
  }

  /**
   * Revoke preview images by execution ID.
   * Frees memory allocated to image preview blobs by revoking the URLs.
   *
   * @param executionId - The execution ID
   */
  function revokePreviewsByExecutionId(executionId: string) {
    const nodeLocatorId = executionIdToNodeLocatorId(app.rootGraph, executionId)
    if (!nodeLocatorId) return
    scheduleRevoke(nodeLocatorId, () =>
      revokePreviewsByLocatorId(nodeLocatorId)
    )
  }

  /**
   * Revoke preview images by node locator ID.
   * Frees memory allocated to image preview blobs by revoking the URLs.
   *
   * @param nodeLocatorId - The node locator ID
   */
  function revokePreviewsByLocatorId(nodeLocatorId: NodeLocatorId) {
    const previews = app.nodePreviewImages[nodeLocatorId]
    if (!previews?.[Symbol.iterator]) return

    for (const url of previews) {
      releaseSharedObjectUrl(url)
    }

    delete app.nodePreviewImages[nodeLocatorId]
    delete nodePreviewImages.value[nodeLocatorId]
  }

  /**
   * Revoke all preview images.
   * Frees memory allocated to all image preview blobs.
   */
  function revokeAllPreviews() {
    for (const nodeLocatorId of Object.keys(app.nodePreviewImages)) {
      const previews = app.nodePreviewImages[nodeLocatorId]
      if (!previews?.[Symbol.iterator]) continue

      for (const url of previews) {
        releaseSharedObjectUrl(url)
      }
    }
    app.nodePreviewImages = {}
    nodePreviewImages.value = {}
  }

  /**
   * Revoke all preview of a subgraph node and the graph it contains.
   * Does not recurse to contents of nested subgraphs.
   */
  function revokeSubgraphPreviews(subgraphNode: SubgraphNode) {
    const { graph } = subgraphNode
    if (!graph) return

    const graphId = graph.isRootGraph ? '' : graph.id + ':'
    revokePreviewsByLocatorId(graphId + subgraphNode.id)
    for (const node of subgraphNode.subgraph.nodes) {
      revokePreviewsByLocatorId(subgraphNode.subgraph.id + node.id)
    }
  }

  /**
   * Remove node outputs for a specific node
   * Clears both outputs and preview images
   */
  function removeNodeOutputs(nodeId: number | string) {
    const nodeLocatorId = nodeIdToNodeLocatorId(Number(nodeId))
    if (!nodeLocatorId) return false

    // Clear from app.nodeOutputs
    const hadOutputs = !!app.nodeOutputs[nodeLocatorId]
    delete app.nodeOutputs[nodeLocatorId]

    // Clear from reactive state
    delete nodeOutputs.value[nodeLocatorId]

    // Clear preview images
    if (app.nodePreviewImages[nodeLocatorId]) {
      const previews = app.nodePreviewImages[nodeLocatorId]
      if (previews?.[Symbol.iterator]) {
        for (const url of previews) {
          releaseSharedObjectUrl(url)
        }
      }
      delete app.nodePreviewImages[nodeLocatorId]
      delete nodePreviewImages.value[nodeLocatorId]
    }

    return hadOutputs
  }

  function restoreOutputs(
    outputs: Record<string, ExecutedWsMessage['output']>
  ) {
    app.nodeOutputs = outputs
    nodeOutputs.value = { ...outputs }
  }

  function updateNodeImages(node: LGraphNode) {
    if (!node.images?.length) return

    const nodeLocatorId = nodeIdToNodeLocatorId(node.id)

    if (nodeLocatorId) {
      const existingOutputs = app.nodeOutputs[nodeLocatorId]

      if (existingOutputs) {
        const updatedOutputs = {
          ...existingOutputs,
          images: node.images
        }

        app.nodeOutputs[nodeLocatorId] = updatedOutputs
        nodeOutputs.value[nodeLocatorId] = updatedOutputs
      }
    }
  }

  function refreshNodeOutputs(node: LGraphNode) {
    const locatorId = nodeToNodeLocatorId(node)
    if (!locatorId) return

    const outputs = app.nodeOutputs[locatorId]
    if (!outputs) return

    nodeOutputs.value[locatorId] = { ...outputs }
  }

  function resetAllOutputsAndPreviews() {
    app.nodeOutputs = {}
    nodeOutputs.value = {}
    revokeAllPreviews()
  }

  /**
   * Sync legacy node.imgs property for backwards compatibility.
   *
   * In Vue Nodes mode, legacy systems (Copy Image, Open Image, Save Image,
   * Open in Mask Editor) rely on `node.imgs` containing HTMLImageElement
   * references. Since Vue handles image rendering, we need to sync the
   * already-loaded element from the Vue component to the node.
   *
   * @param nodeId - The node ID
   * @param element - The loaded HTMLImageElement from the Vue component
   * @param activeIndex - The current image index (for multi-image outputs)
   */
  function syncLegacyNodeImgs(
    nodeId: string | number,
    element: HTMLImageElement,
    activeIndex: number = 0
  ) {
    if (!LiteGraph.vueNodesMode) return

    const node = app.rootGraph?.getNodeById(Number(nodeId))
    if (!node) return

    node.imgs = [element]
    node.imageIndex = activeIndex
  }

  return {
    // Getters
    getNodeOutputs,
    getNodeImageUrls,
    getNodePreviews,
    getPreviewParam,

    // Setters
    setNodeOutputs,
    setNodeOutputsByExecutionId,
    setNodePreviewsByExecutionId,
    setNodePreviewsByNodeId,
    updateNodeImages,
    refreshNodeOutputs,
    syncLegacyNodeImgs,

    // Cleanup
    revokePreviewsByExecutionId,
    revokeAllPreviews,
    revokeSubgraphPreviews,
    removeNodeOutputs,
    restoreOutputs,
    resetAllOutputsAndPreviews,

    // State
    nodeOutputs,
    nodePreviewImages,
    latestPreview
  }
})
