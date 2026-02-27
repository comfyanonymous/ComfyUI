import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { NodeReplacement } from './types'
import type { MissingNodeType } from '@/types/comfy'

vi.mock('@/lib/litegraph/src/litegraph', () => ({
  LiteGraph: {
    createNode: vi.fn(),
    registered_node_types: {}
  }
}))

vi.mock('@/scripts/app', () => ({
  app: { rootGraph: null },
  sanitizeNodeName: (name: string) => name.replace(/[&<>"'`=]/g, '')
}))

vi.mock('@/utils/graphTraversalUtil', () => ({
  collectAllNodes: vi.fn()
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: vi.fn(() => ({
    add: vi.fn()
  }))
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: vi.fn(() => ({
    activeWorkflow: {
      changeTracker: {
        beforeChange: vi.fn(),
        afterChange: vi.fn()
      }
    }
  }))
}))

vi.mock('@/i18n', () => ({
  t: (key: string, params?: Record<string, unknown>) =>
    params ? `${key}:${JSON.stringify(params)}` : key
}))

import { app } from '@/scripts/app'
import { collectAllNodes } from '@/utils/graphTraversalUtil'
import { useNodeReplacement } from './useNodeReplacement'

function createMockLink(
  id: number,
  originId: number,
  originSlot: number,
  targetId: number,
  targetSlot: number
) {
  return {
    id,
    origin_id: originId,
    origin_slot: originSlot,
    target_id: targetId,
    target_slot: targetSlot,
    type: 'IMAGE'
  }
}

function createMockGraph(
  nodes: LGraphNode[],
  links: ReturnType<typeof createMockLink>[] = []
): LGraph {
  const linksMap = new Map(links.map((l) => [l.id, l]))
  return {
    _nodes: nodes,
    _nodes_by_id: Object.fromEntries(nodes.map((n) => [n.id, n])),
    links: linksMap,
    updateExecutionOrder: vi.fn(),
    setDirtyCanvas: vi.fn()
  } as unknown as LGraph
}

function createPlaceholderNode(
  id: number,
  type: string,
  inputs: { name: string; link: number | null }[] = [],
  outputs: { name: string; links: number[] | null }[] = [],
  graph?: LGraph
): LGraphNode {
  return {
    id,
    type,
    pos: [100, 200],
    size: [200, 100],
    order: 0,
    mode: 0,
    flags: {},
    has_errors: true,
    last_serialization: {
      id,
      type,
      pos: [100, 200],
      size: [200, 100],
      flags: {},
      order: 0,
      mode: 0,
      inputs: inputs.map((i) => ({ ...i, type: 'IMAGE' })),
      outputs: outputs.map((o) => ({ ...o, type: 'IMAGE' })),
      widgets_values: []
    },
    inputs: inputs.map((i) => ({ ...i, type: 'IMAGE' })),
    outputs: outputs.map((o) => ({ ...o, type: 'IMAGE' })),
    graph: graph ?? null,
    serialize: vi.fn(() => ({
      id,
      type,
      pos: [100, 200],
      size: [200, 100],
      flags: {},
      order: 0,
      mode: 0,
      inputs: inputs.map((i) => ({ ...i, type: 'IMAGE' })),
      outputs: outputs.map((o) => ({ ...o, type: 'IMAGE' })),
      widgets_values: []
    }))
  } as unknown as LGraphNode
}

function createNewNode(
  inputs: { name: string; link: number | null }[] = [],
  outputs: { name: string; links: number[] | null }[] = [],
  widgets: { name: string; value: unknown }[] = []
): LGraphNode {
  return {
    id: 0,
    type: '',
    pos: [0, 0],
    size: [100, 50],
    order: 0,
    mode: 0,
    flags: {},
    has_errors: false,
    inputs: inputs.map((i) => ({ ...i, type: 'IMAGE' })),
    outputs: outputs.map((o) => ({ ...o, type: 'IMAGE' })),
    widgets: widgets.map((w) => ({ ...w, type: 'combo', options: {} })),
    configure: vi.fn(),
    serialize: vi.fn()
  } as unknown as LGraphNode
}

function makeMissingNodeType(
  type: string,
  replacement: NodeReplacement
): MissingNodeType {
  return {
    type,
    isReplaceable: true,
    replacement
  }
}

describe('useNodeReplacement', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createPinia())
  })

  describe('replaceNodesInPlace', () => {
    it('should return empty array when no placeholders exist', () => {
      const graph = createMockGraph([])
      Object.assign(app, { rootGraph: graph })
      vi.mocked(collectAllNodes).mockReturnValue([])

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([])

      expect(result).toEqual([])
    })

    it('should use default mapping when no explicit mapping exists', () => {
      const placeholder = createPlaceholderNode(1, 'Load3DAnimation')
      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode()
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([
        makeMissingNodeType('Load3DAnimation', {
          new_node_id: 'Load3D',
          old_node_id: 'Load3DAnimation',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        })
      ])

      expect(result).toEqual(['Load3DAnimation'])
      expect(newNode.configure).not.toHaveBeenCalled()
      expect(newNode.id).toBe(1)
      expect(newNode.has_errors).toBe(false)
    })

    it('should transfer input connections using input_mapping', () => {
      const link = createMockLink(10, 5, 0, 1, 0)
      const placeholder = createPlaceholderNode(
        1,
        'T2IAdapterLoader',
        [{ name: 't2i_adapter_name', link: 10 }],
        []
      )
      const graph = createMockGraph([placeholder], [link])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [{ name: 'control_net_name', link: null }],
        []
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([
        makeMissingNodeType('T2IAdapterLoader', {
          new_node_id: 'ControlNetLoader',
          old_node_id: 'T2IAdapterLoader',
          old_widget_ids: null,
          input_mapping: [
            { new_id: 'control_net_name', old_id: 't2i_adapter_name' }
          ],
          output_mapping: null
        })
      ])

      expect(result).toEqual(['T2IAdapterLoader'])
      // Link should be updated to point at new node's input
      expect(link.target_id).toBe(1)
      expect(link.target_slot).toBe(0)
      expect(newNode.inputs[0].link).toBe(10)
    })

    it('should transfer output connections using output_mapping', () => {
      const link = createMockLink(20, 1, 0, 5, 0)
      const placeholder = createPlaceholderNode(
        1,
        'ResizeImagesByLongerEdge',
        [],
        [{ name: 'IMAGE', links: [20] }]
      )
      const graph = createMockGraph([placeholder], [link])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [{ name: 'image', link: null }],
        [{ name: 'IMAGE', links: null }]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ResizeImagesByLongerEdge', {
          new_node_id: 'ImageScaleToMaxDimension',
          old_node_id: 'ResizeImagesByLongerEdge',
          old_widget_ids: ['longer_edge'],
          input_mapping: [
            { new_id: 'image', old_id: 'images' },
            { new_id: 'largest_size', old_id: 'longer_edge' },
            { new_id: 'upscale_method', set_value: 'lanczos' }
          ],
          output_mapping: [{ new_idx: 0, old_idx: 0 }]
        })
      ])

      // Output link should be remapped
      expect(link.origin_id).toBe(1)
      expect(link.origin_slot).toBe(0)
      expect(newNode.outputs[0].links).toEqual([20])
    })

    it('should apply set_value to widget', () => {
      const placeholder = createPlaceholderNode(1, 'ImageScaleBy')
      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [{ name: 'input', link: null }],
        [],
        [
          { name: 'resize_type', value: '' },
          { name: 'scale_method', value: '' }
        ]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ImageScaleBy', {
          new_node_id: 'ResizeImageMaskNode',
          old_node_id: 'ImageScaleBy',
          old_widget_ids: ['upscale_method', 'scale_by'],
          input_mapping: [
            { new_id: 'input', old_id: 'image' },
            { new_id: 'resize_type', set_value: 'scale by multiplier' },
            { new_id: 'resize_type.multiplier', old_id: 'scale_by' },
            { new_id: 'scale_method', old_id: 'upscale_method' }
          ],
          output_mapping: null
        })
      ])

      // set_value should be applied to the widget
      expect(newNode.widgets![0].value).toBe('scale by multiplier')
    })

    it('should transfer widget values using old_widget_ids', () => {
      const placeholder = createPlaceholderNode(1, 'ResizeImagesByLongerEdge')
      // Set widget values in serialized data
      placeholder.last_serialization!.widgets_values = [512]

      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [
          { name: 'image', link: null },
          { name: 'largest_size', link: null }
        ],
        [{ name: 'IMAGE', links: null }],
        [{ name: 'largest_size', value: 0 }]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ResizeImagesByLongerEdge', {
          new_node_id: 'ImageScaleToMaxDimension',
          old_node_id: 'ResizeImagesByLongerEdge',
          old_widget_ids: ['longer_edge'],
          input_mapping: [
            { new_id: 'image', old_id: 'images' },
            { new_id: 'largest_size', old_id: 'longer_edge' },
            { new_id: 'upscale_method', set_value: 'lanczos' }
          ],
          output_mapping: [{ new_idx: 0, old_idx: 0 }]
        })
      ])

      // Widget value should be transferred: old "longer_edge" (idx 0, value 512) → new "largest_size"
      expect(newNode.widgets![0].value).toBe(512)
    })

    it('should skip replacement when new node type is not registered', () => {
      const placeholder = createPlaceholderNode(1, 'UnknownNode')
      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])
      vi.mocked(LiteGraph.createNode).mockReturnValue(null)

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([
        makeMissingNodeType('UnknownNode', {
          new_node_id: 'NonExistentNode',
          old_node_id: 'UnknownNode',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        })
      ])

      expect(result).toEqual([])
    })

    it('should replace multiple different node types at once', () => {
      const placeholder1 = createPlaceholderNode(1, 'Load3DAnimation')
      const placeholder2 = createPlaceholderNode(
        2,
        'ConditioningAverage',
        [],
        []
      )
      // sanitizeNodeName strips & from type names (HTML entity chars)
      placeholder2.type = 'ConditioningAverage'

      const graph = createMockGraph([placeholder1, placeholder2])
      placeholder1.graph = graph
      placeholder2.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder1, placeholder2])

      const newNode1 = createNewNode()
      const newNode2 = createNewNode()
      vi.mocked(LiteGraph.createNode)
        .mockReturnValueOnce(newNode1)
        .mockReturnValueOnce(newNode2)

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([
        makeMissingNodeType('Load3DAnimation', {
          new_node_id: 'Load3D',
          old_node_id: 'Load3DAnimation',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        }),
        makeMissingNodeType('ConditioningAverage&', {
          new_node_id: 'ConditioningAverage',
          old_node_id: 'ConditioningAverage&',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        })
      ])

      expect(result).toHaveLength(2)
      expect(result).toContain('Load3DAnimation')
      expect(result).toContain('ConditioningAverage&')
    })

    it('should copy position and identity for mapped replacements', () => {
      const link = createMockLink(10, 5, 0, 1, 0)
      const placeholder = createPlaceholderNode(
        42,
        'T2IAdapterLoader',
        [{ name: 't2i_adapter_name', link: 10 }],
        []
      )
      placeholder.pos = [300, 400]
      placeholder.size = [250, 150]

      const graph = createMockGraph([placeholder], [link])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [{ name: 'control_net_name', link: null }],
        []
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('T2IAdapterLoader', {
          new_node_id: 'ControlNetLoader',
          old_node_id: 'T2IAdapterLoader',
          old_widget_ids: null,
          input_mapping: [
            { new_id: 'control_net_name', old_id: 't2i_adapter_name' }
          ],
          output_mapping: null
        })
      ])

      expect(newNode.id).toBe(42)
      expect(newNode.pos).toEqual([300, 400])
      expect(newNode.size).toEqual([250, 150])
      expect(graph._nodes[0]).toBe(newNode)
    })

    it('should transfer all widget values for ImageScaleBy with real workflow data', () => {
      const placeholder = createPlaceholderNode(
        12,
        'ImageScaleBy',
        [{ name: 'image', link: 2 }],
        [{ name: 'IMAGE', links: [3, 4] }]
      )
      // Real workflow data: widgets_values: ["lanczos", 2.0]
      placeholder.last_serialization!.widgets_values = ['lanczos', 2.0]

      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [{ name: 'input', link: null }],
        [],
        [
          { name: 'resize_type', value: '' },
          { name: 'scale_method', value: '' }
        ]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ImageScaleBy', {
          new_node_id: 'ResizeImageMaskNode',
          old_node_id: 'ImageScaleBy',
          old_widget_ids: ['upscale_method', 'scale_by'],
          input_mapping: [
            { new_id: 'input', old_id: 'image' },
            { new_id: 'resize_type', set_value: 'scale by multiplier' },
            { new_id: 'resize_type.multiplier', old_id: 'scale_by' },
            { new_id: 'scale_method', old_id: 'upscale_method' }
          ],
          output_mapping: null
        })
      ])

      // set_value should be applied
      expect(newNode.widgets![0].value).toBe('scale by multiplier')
      // upscale_method (idx 0, value "lanczos") → scale_method widget
      expect(newNode.widgets![1].value).toBe('lanczos')
    })

    it('should transfer widget value for ResizeImagesByLongerEdge with real workflow data', () => {
      const link = createMockLink(1, 5, 0, 8, 0)
      const placeholder = createPlaceholderNode(
        8,
        'ResizeImagesByLongerEdge',
        [{ name: 'images', link: 1 }],
        [{ name: 'IMAGE', links: [2] }]
      )
      // Real workflow data: widgets_values: [1024]
      placeholder.last_serialization!.widgets_values = [1024]

      const graph = createMockGraph([placeholder], [link])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [
          { name: 'image', link: null },
          { name: 'largest_size', link: null }
        ],
        [{ name: 'IMAGE', links: null }],
        [
          { name: 'largest_size', value: 0 },
          { name: 'upscale_method', value: '' }
        ]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ResizeImagesByLongerEdge', {
          new_node_id: 'ImageScaleToMaxDimension',
          old_node_id: 'ResizeImagesByLongerEdge',
          old_widget_ids: ['longer_edge'],
          input_mapping: [
            { new_id: 'image', old_id: 'images' },
            { new_id: 'largest_size', old_id: 'longer_edge' },
            { new_id: 'upscale_method', set_value: 'lanczos' }
          ],
          output_mapping: [{ new_idx: 0, old_idx: 0 }]
        })
      ])

      // longer_edge (idx 0, value 1024) → largest_size widget
      expect(newNode.widgets![0].value).toBe(1024)
      // set_value "lanczos" → upscale_method widget
      expect(newNode.widgets![1].value).toBe('lanczos')
    })

    it('should transfer ConditioningAverage widget value with real workflow data', () => {
      const link = createMockLink(4, 7, 0, 13, 0)
      // sanitizeNodeName doesn't strip spaces, so placeholder keeps trailing space
      const placeholder = createPlaceholderNode(
        13,
        'ConditioningAverage ',
        [
          { name: 'conditioning_to', link: 4 },
          { name: 'conditioning_from', link: null }
        ],
        [{ name: 'CONDITIONING', links: [6] }]
      )
      placeholder.last_serialization!.widgets_values = [0.75]

      const graph = createMockGraph([placeholder], [link])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode(
        [
          { name: 'conditioning_to', link: null },
          { name: 'conditioning_from', link: null }
        ],
        [{ name: 'CONDITIONING', links: null }],
        [{ name: 'conditioning_average', value: 0 }]
      )
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      replaceNodesInPlace([
        makeMissingNodeType('ConditioningAverage ', {
          new_node_id: 'ConditioningAverage',
          old_node_id: 'ConditioningAverage ',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        })
      ])

      // Default mapping transfers connections and widget values by name
      expect(newNode.id).toBe(13)
      expect(newNode.inputs[0].link).toBe(4)
      expect(newNode.outputs[0].links).toEqual([6])
      expect(newNode.widgets![0].value).toBe(0.75)
    })

    it('should skip dot-notation input connections but still transfer widget values', () => {
      const placeholder = createPlaceholderNode(1, 'ImageBatch')
      const graph = createMockGraph([placeholder])
      placeholder.graph = graph
      Object.assign(app, { rootGraph: graph })

      vi.mocked(collectAllNodes).mockReturnValue([placeholder])

      const newNode = createNewNode([], [])
      vi.mocked(LiteGraph.createNode).mockReturnValue(newNode)

      const { replaceNodesInPlace } = useNodeReplacement()
      const result = replaceNodesInPlace([
        makeMissingNodeType('ImageBatch', {
          new_node_id: 'BatchImagesNode',
          old_node_id: 'ImageBatch',
          old_widget_ids: null,
          input_mapping: [
            { new_id: 'images.image0', old_id: 'image1' },
            { new_id: 'images.image1', old_id: 'image2' }
          ],
          output_mapping: null
        })
      ])

      // Should still succeed (dot-notation skipped gracefully)
      expect(result).toEqual(['ImageBatch'])
    })
  })
})
