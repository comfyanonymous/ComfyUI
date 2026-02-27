import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { LGraphNode, Point } from '@/lib/litegraph/src/litegraph'
import { assetItemSchema } from '@/platform/assets/schemas/assetSchema'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import {
  MISSING_TAG,
  MODELS_TAG
} from '@/platform/assets/services/assetService'
import { getAssetFilename } from '@/platform/assets/utils/assetMetadataUtils'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useLitegraphService } from '@/services/litegraphService'
import { useModelToNodeStore } from '@/stores/modelToNodeStore'

interface CreateNodeOptions {
  position?: Point
}

type NodeCreationErrorCode =
  | 'INVALID_ASSET'
  | 'NO_PROVIDER'
  | 'NODE_CREATION_FAILED'
  | 'MISSING_WIDGET'
  | 'NO_GRAPH'

interface NodeCreationError {
  code: NodeCreationErrorCode
  message: string
  assetId: string
  details?: Record<string, unknown>
}

type Result<T, E> = { success: true; value: T } | { success: false; error: E }

/**
 * Creates a LiteGraph node from an asset item.
 *
 * **Boundary Function**: Bridges Vue reactive domain with LiteGraph canvas domain.
 *
 * @param asset - Asset item to create node from (Vue domain)
 * @param options - Optional position and configuration
 * @returns Result with LiteGraph node (Canvas domain) or error details
 *
 * @remarks
 * This function performs side effects on the canvas graph. Validation failures
 * return error results rather than throwing to allow graceful degradation in UI contexts.
 * Widget validation occurs before graph mutation to prevent orphaned nodes.
 */
export function createModelNodeFromAsset(
  asset: AssetItem,
  options?: CreateNodeOptions
): Result<LGraphNode, NodeCreationError> {
  const validatedAsset = assetItemSchema.safeParse(asset)

  if (!validatedAsset.success) {
    const errorMessage = validatedAsset.error.errors
      .map((e) => `${e.path.join('.')}: ${e.message}`)
      .join(', ')
    console.error('Invalid asset item:', errorMessage)
    return {
      success: false,
      error: {
        code: 'INVALID_ASSET',
        message: 'Asset schema validation failed',
        assetId: asset.id,
        details: { validationErrors: errorMessage }
      }
    }
  }

  const validAsset = validatedAsset.data

  const filename = getAssetFilename(validAsset)
  if (filename.length === 0) {
    console.error(
      `Asset ${validAsset.id} has invalid user_metadata.filename (expected non-empty string, got ${typeof filename})`
    )
    return {
      success: false,
      error: {
        code: 'INVALID_ASSET',
        message: `Invalid filename (expected non-empty string, got ${typeof filename})`,
        assetId: validAsset.id
      }
    }
  }

  if (validAsset.tags.length === 0) {
    console.error(
      `Asset ${validAsset.id} has no tags defined (expected at least one category tag)`
    )
    return {
      success: false,
      error: {
        code: 'INVALID_ASSET',
        message: 'Asset has no tags defined',
        assetId: validAsset.id
      }
    }
  }

  const category = validAsset.tags.find(
    (tag) => tag !== MODELS_TAG && tag !== MISSING_TAG
  )
  if (!category) {
    console.error(
      `Asset ${validAsset.id} has no valid category tag. Available tags: ${validAsset.tags.join(', ')} (expected tag other than '${MODELS_TAG}' or '${MISSING_TAG}')`
    )
    return {
      success: false,
      error: {
        code: 'INVALID_ASSET',
        message: 'Asset has no valid category tag',
        assetId: validAsset.id,
        details: { availableTags: validAsset.tags }
      }
    }
  }

  const modelToNodeStore = useModelToNodeStore()
  const provider = modelToNodeStore.getNodeProvider(category)
  if (!provider) {
    console.error(`No node provider registered for category: ${category}`)
    return {
      success: false,
      error: {
        code: 'NO_PROVIDER',
        message: `No node provider registered for category: ${category}`,
        assetId: validAsset.id,
        details: { category }
      }
    }
  }

  const litegraphService = useLitegraphService()
  const pos = options?.position ?? litegraphService.getCanvasCenter()

  const node = LiteGraph.createNode(
    provider.nodeDef.name,
    provider.nodeDef.display_name,
    { pos }
  )

  if (!node) {
    console.error(`Failed to create node for type: ${provider.nodeDef.name}`)
    return {
      success: false,
      error: {
        code: 'NODE_CREATION_FAILED',
        message: `Failed to create node for type: ${provider.nodeDef.name}`,
        assetId: validAsset.id,
        details: { nodeType: provider.nodeDef.name }
      }
    }
  }

  const workflowStore = useWorkflowStore()
  const targetGraph = workflowStore.isSubgraphActive
    ? workflowStore.activeSubgraph
    : app.canvas.graph

  if (!targetGraph) {
    console.error('No active graph available')
    return {
      success: false,
      error: {
        code: 'NO_GRAPH',
        message: 'No active graph available',
        assetId: validAsset.id
      }
    }
  }

  // Set widget value if provider specifies a key (some nodes auto-load models without a widget)
  if (provider.key) {
    const widget = node.widgets?.find((w) => w.name === provider.key)
    if (!widget) {
      console.error(
        `Widget ${provider.key} not found on node ${provider.nodeDef.name}`
      )
      return {
        success: false,
        error: {
          code: 'MISSING_WIDGET',
          message: `Widget ${provider.key} not found on node ${provider.nodeDef.name}`,
          assetId: validAsset.id,
          details: { widgetName: provider.key, nodeType: provider.nodeDef.name }
        }
      }
    }
    widget.value = filename
  }

  // Add the node to the graph
  targetGraph.add(node)

  return { success: true, value: node }
}
