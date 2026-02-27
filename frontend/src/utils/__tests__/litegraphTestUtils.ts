import type {
  INodeInputSlot,
  INodeOutputSlot,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import type {
  CanvasPointerEvent,
  LGraph,
  LGraphCanvas,
  LGraphGroup,
  LinkNetwork,
  LLink
} from '@/lib/litegraph/src/litegraph'
import { LGraphEventMode, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { vi } from 'vitest'
import type { ChangeTracker } from '@/scripts/changeTracker'

/**
 * Creates a mock LGraphNode with minimal required properties
 */
export function createMockLGraphNode(
  overrides: Partial<LGraphNode> | Record<string, unknown> = {}
): LGraphNode {
  const partial: Partial<LGraphNode> = {
    id: 1,
    pos: [0, 0],
    size: [100, 100],
    title: 'Test Node',
    mode: LGraphEventMode.ALWAYS,
    ...(overrides as Partial<LGraphNode>)
  }
  return partial as Partial<LGraphNode> as LGraphNode
}

/**
 * Creates a mock Positionable object
 */
export function createMockPositionable(
  overrides: Partial<Positionable> = {}
): Positionable {
  const partial: Partial<Positionable> = {
    id: 1,
    pos: [0, 0],
    ...overrides
  }
  return partial as Partial<Positionable> as Positionable
}

/**
 * Creates a mock LGraphGroup with minimal required properties
 */
export function createMockLGraphGroup(
  overrides: Partial<LGraphGroup> = {}
): LGraphGroup {
  const partial: Partial<LGraphGroup> = {
    id: 1,
    pos: [0, 0],
    boundingRect: new Rectangle(0, 0, 100, 100),
    ...overrides
  }
  return partial as Partial<LGraphGroup> as LGraphGroup
}

/**
 * Creates a mock SubgraphNode with sub-nodes
 */
export function createMockSubgraphNode(
  subNodes: LGraphNode[],
  overrides: Partial<LGraphNode> | Record<string, unknown> = {}
): LGraphNode {
  const baseNode = createMockLGraphNode(overrides)
  return Object.assign(baseNode, {
    isSubgraphNode: () => true,
    subgraph: {
      nodes: subNodes
    }
  })
}

/**
 * Creates a mock LGraphCanvas with minimal required properties for testing
 */
export function createMockCanvas(
  overrides: Partial<LGraphCanvas> | Record<string, unknown> = {}
): LGraphCanvas {
  return {
    setDirty: vi.fn(),
    state: {
      selectionChanged: false
    },
    ...(overrides as Partial<LGraphCanvas>)
  } as LGraphCanvas
}

/**
 * Creates a mock LGraph with trigger function
 */
export function createMockLGraph(overrides: Partial<LGraph> = {}): LGraph {
  return {
    trigger: vi.fn(),
    ...overrides
  } as LGraph
}

/**
 * Creates a mock CanvasPointerEvent
 */
export function createMockCanvasPointerEvent(
  canvasX: number,
  canvasY: number,
  overrides: Partial<CanvasPointerEvent> = {}
): CanvasPointerEvent {
  return {
    canvasX,
    canvasY,
    ...overrides
  } as CanvasPointerEvent
}

/**
 * Creates a mock CanvasRenderingContext2D
 */
export function createMockCanvasRenderingContext2D(
  overrides: Partial<CanvasRenderingContext2D> = {}
): CanvasRenderingContext2D {
  const partial: Partial<CanvasRenderingContext2D> = {
    measureText: vi.fn(() => ({ width: 10 }) as TextMetrics),
    ...overrides
  }
  return partial as CanvasRenderingContext2D
}

/**
 * Creates a mock LinkNetwork
 */
export function createMockLinkNetwork(
  overrides: Partial<LinkNetwork> = {}
): LinkNetwork {
  return {
    ...overrides
  } as LinkNetwork
}

/**
 * Creates a mock INodeInputSlot
 */
export function createMockNodeInputSlot(
  overrides: Partial<INodeInputSlot> = {}
): INodeInputSlot {
  return {
    ...overrides
  } as INodeInputSlot
}

/**
 * Creates a mock INodeOutputSlot
 */
export function createMockNodeOutputSlot(
  overrides: Partial<INodeOutputSlot> = {}
): INodeOutputSlot {
  return {
    ...overrides
  } as INodeOutputSlot
}

/**
 * Creates a real LGraphNode instance (not a lightweight mock) with its boundingRect
 * property represented as a Float64Array for testing position methods.
 *
 * Use createMockLGraphNodeWithArrayBoundingRect when:
 * - Tests rely on Float64Array boundingRect behavior
 * - Tests call position-related methods like updateArea()
 * - Tests need actual LGraphNode implementation details
 *
 * Use createMockLGraphNode when:
 * - Tests only need simple/mock-only behavior
 * - Tests don't depend on boundingRect being a Float64Array
 * - A lightweight mock with minimal properties is sufficient
 *
 * @param name - The node name/type to pass to the LGraphNode constructor
 * @returns A fully constructed LGraphNode instance with Float64Array boundingRect
 */
export function createMockLGraphNodeWithArrayBoundingRect(
  name: string
): LGraphNode {
  const node = new LGraphNode(name)
  // The actual node has a Float64Array boundingRect, we just need to type it correctly
  return node
}

/**
 * Creates a mock FileList from an array of files
 */
export function createMockFileList(files: File[]): FileList {
  const fileList = Object.assign(
    {
      length: files.length,
      item: (index: number) => files[index] ?? null,
      [Symbol.iterator]: function* () {
        yield* files
      }
    },
    files
  )
  return fileList as FileList
}

/**
 * Creates a mock ChangeTracker for workflow testing
 * The ChangeTracker requires a proper ComfyWorkflowJSON structure
 */
export function createMockChangeTracker(
  overrides: Partial<ChangeTracker> = {}
): ChangeTracker {
  const partial = {
    activeState: {
      last_node_id: 0,
      last_link_id: 0,
      nodes: [],
      links: [],
      groups: [],
      config: {},
      version: 0.4
    },
    undoQueue: [],
    redoQueue: [],
    changeCount: 0,
    checkState: vi.fn(),
    reset: vi.fn(),
    restore: vi.fn(),
    store: vi.fn(),
    ...overrides
  }
  return partial as Partial<ChangeTracker> as ChangeTracker
}

/**
 * Creates a mock MinimapCanvas for minimap testing
 */
export function createMockMinimapCanvas(
  overrides: Partial<HTMLCanvasElement> = {}
): HTMLCanvasElement {
  const mockGetContext = vi.fn()
  mockGetContext.mockImplementation((contextId: string) =>
    contextId === '2d' ? createMockCanvas2DContext() : null
  )

  const partial: Partial<HTMLCanvasElement> = {
    width: 200,
    height: 200,
    clientWidth: 200,
    clientHeight: 200,
    getContext: mockGetContext as HTMLCanvasElement['getContext'],
    ...overrides
  }
  return partial as HTMLCanvasElement
}

/**
 * Creates a mock CanvasRenderingContext2D for canvas testing
 */
export function createMockCanvas2DContext(
  overrides: Partial<CanvasRenderingContext2D> = {}
): CanvasRenderingContext2D {
  const partial: Partial<CanvasRenderingContext2D> = {
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    ...overrides
  }
  return partial as CanvasRenderingContext2D
}

export function createMockLLink(overrides: Partial<LLink> = {}): LLink {
  const partial: Partial<LLink> = {
    id: 1,
    type: '*',
    origin_id: 1,
    origin_slot: 0,
    target_id: 2,
    target_slot: 0,
    _pos: [0, 0],
    ...overrides
  }
  return partial as LLink
}

export function createMockLinks(links: LLink[]): LGraph['links'] {
  const map = new Map<number, LLink>()
  const record: Record<number, LLink> = {}
  for (const link of links) {
    map.set(link.id, link)
    record[link.id] = link
  }
  return Object.assign(map, record) as LGraph['links']
}
