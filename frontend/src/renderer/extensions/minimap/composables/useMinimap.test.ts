import type { Mock } from 'vitest'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, shallowRef } from 'vue'

import {
  createMockCanvas2DContext,
  createMockMinimapCanvas
} from '@/utils/__tests__/litegraphTestUtils'

interface MockNode {
  id: string
  pos: number[]
  size: number[]
  color?: string
  constructor?: { color: string }
  outputs?: { links: string[] }[] | null
}

interface MockGraph {
  _nodes: MockNode[]
  links: Record<string, { id: string; target_id: string }>
  getNodeById: Mock
  setDirtyCanvas: Mock
  onNodeAdded: ((node: MockNode) => void) | null
  onNodeRemoved: ((node: MockNode) => void) | null
  onConnectionChange: ((node: MockNode) => void) | null
}

interface MockCanvas {
  graph: MockGraph
  canvas: {
    width: number
    height: number
    clientWidth: number
    clientHeight: number
  }
  ds: {
    scale: number
    offset: [number, number]
  }
  setDirty: Mock
}

interface MockContainerElement {
  getBoundingClientRect: Mock
}

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0))

const triggerRAF = async () => {
  // Trigger all RAF callbacks
  Object.values(rafCallbacks).forEach((cb) => cb?.())
  await flushPromises()
}

const mockPause = vi.fn()
const mockResume = vi.fn()

const rafCallbacks: Record<string, () => void> = {}
let rafCallbackId = 0

vi.mock('@vueuse/core', () => {
  return {
    useRafFn: vi.fn((callback, options) => {
      const id = rafCallbackId++
      rafCallbacks[id] = callback

      if (options?.immediate !== false) {
        void Promise.resolve().then(() => callback())
      }

      const resumeFn = vi.fn(() => {
        mockResume()
        // Execute the RAF callback immediately when resumed
        if (rafCallbacks[id]) {
          rafCallbacks[id]()
        }
      })

      return {
        pause: mockPause,
        resume: resumeFn
      }
    }),
    useThrottleFn: vi.fn((callback) => {
      return (...args: unknown[]) => {
        return callback(...args)
      }
    })
  }
})

let moduleMockCanvas: MockCanvas = null!
let moduleMockGraph: MockGraph = null!

const setupMocks = () => {
  const mockNodes = [
    {
      id: 'node1',
      pos: [0, 0],
      size: [100, 50],
      color: '#ff0000',
      constructor: { color: '#666' },
      outputs: [
        {
          links: ['link1']
        }
      ]
    },
    {
      id: 'node2',
      pos: [200, 100],
      size: [150, 75],
      constructor: { color: '#666' },
      outputs: []
    }
  ]

  moduleMockGraph = {
    _nodes: mockNodes,
    links: {
      link1: {
        id: 'link1',
        target_id: 'node2'
      }
    },
    getNodeById: vi.fn((id) => mockNodes.find((n) => n.id === id)),
    setDirtyCanvas: vi.fn(),
    onNodeAdded: null,
    onNodeRemoved: null,
    onConnectionChange: null
  }

  moduleMockCanvas = {
    graph: moduleMockGraph,
    canvas: {
      width: 1000,
      height: 800,
      clientWidth: 1000,
      clientHeight: 800
    },
    ds: {
      scale: 1,
      offset: [0, 0]
    },
    setDirty: vi.fn()
  }
}

setupMocks()

const defaultCanvasStore: {
  canvas: MockCanvas | null
  getCanvas: () => MockCanvas | null
} = {
  canvas: moduleMockCanvas,
  getCanvas: () => defaultCanvasStore.canvas
}

const defaultSettingStore = {
  get: vi.fn().mockReturnValue(true),
  set: vi.fn().mockResolvedValue(undefined)
}

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: vi.fn(() => defaultCanvasStore)
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => defaultSettingStore)
}))

vi.mock('@/stores/workspace/colorPaletteStore', () => ({
  useColorPaletteStore: vi.fn(() => ({
    completedActivePalette: {
      light_theme: false
    }
  }))
}))

vi.mock('@/scripts/api', () => ({
  api: {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    apiURL: vi.fn().mockReturnValue('http://localhost:8188')
  }
}))

vi.mock('@/scripts/app', () => ({
  app: {
    canvas: {
      graph: moduleMockGraph
    }
  }
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: vi.fn(() => ({
    activeSubgraph: null
  }))
}))

import { useMinimap } from '@/renderer/extensions/minimap/composables/useMinimap'
import { api } from '@/scripts/api'

describe('useMinimap', () => {
  let moduleMockCanvasElement: HTMLCanvasElement
  let mockContainerElement: MockContainerElement
  let mockContext2D: CanvasRenderingContext2D

  async function createAndInitializeMinimap() {
    const minimap = useMinimap({
      containerRefMaybe: shallowRef(
        mockContainerElement as Partial<HTMLDivElement> as HTMLDivElement
      ),
      canvasRefMaybe: shallowRef(moduleMockCanvasElement)
    })
    await minimap.init()
    await nextTick()
    await flushPromises()
    return minimap
  }

  beforeEach(() => {
    vi.clearAllMocks()

    mockPause.mockClear()
    mockResume.mockClear()

    mockContext2D = createMockCanvas2DContext()

    moduleMockCanvasElement = createMockMinimapCanvas({
      getContext: vi
        .fn()
        .mockImplementation((contextId) =>
          contextId === '2d' ? mockContext2D : null
        ) as HTMLCanvasElement['getContext']
    })

    const mockRect = {
      left: 100,
      top: 100,
      width: 250,
      height: 200,
      right: 350,
      bottom: 300,
      x: 100,
      y: 100
    }

    mockContainerElement = {
      getBoundingClientRect: vi.fn(() => ({ ...mockRect }))
    }

    const mockNodes = [
      {
        id: 'node1',
        pos: [0, 0],
        size: [100, 50],
        color: '#ff0000',
        constructor: { color: '#666' },
        outputs: [
          {
            links: ['link1']
          }
        ]
      },
      {
        id: 'node2',
        pos: [200, 100],
        size: [150, 75],
        constructor: { color: '#666' },
        outputs: []
      }
    ]

    moduleMockGraph = {
      _nodes: mockNodes,
      links: {
        link1: {
          id: 'link1',
          target_id: 'node2'
        }
      },
      getNodeById: vi.fn((id) => mockNodes.find((n) => n.id === id)),
      setDirtyCanvas: vi.fn(),
      onNodeAdded: null,
      onNodeRemoved: null,
      onConnectionChange: null
    }

    moduleMockCanvas = {
      graph: moduleMockGraph,
      canvas: {
        width: 1000,
        height: 800,
        clientWidth: 1000,
        clientHeight: 800
      },
      ds: {
        scale: 1,
        offset: [0, 0]
      },
      setDirty: vi.fn()
    }

    defaultCanvasStore.canvas = moduleMockCanvas

    defaultSettingStore.get = vi.fn().mockReturnValue(true)
    defaultSettingStore.set = vi.fn().mockResolvedValue(undefined)

    Object.defineProperty(window, 'devicePixelRatio', {
      writable: true,
      configurable: true,
      value: 1
    })

    window.addEventListener = vi.fn()
    window.removeEventListener = vi.fn()
  })

  describe('initialization', () => {
    it('should initialize with default values', () => {
      const originalCanvas = defaultCanvasStore.canvas
      defaultCanvasStore.canvas = null

      const minimap = useMinimap()

      expect(minimap.width).toBe(250)
      expect(minimap.height).toBe(200)
      expect(minimap.visible.value).toBe(true)
      expect(minimap.initialized.value).toBe(false)

      defaultCanvasStore.canvas = originalCanvas
    })

    it('should initialize minimap when canvas is available', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      expect(minimap.initialized.value).toBe(true)
      expect(defaultSettingStore.get).toHaveBeenCalledWith(
        'Comfy.Minimap.Visible'
      )
      expect(api.addEventListener).toHaveBeenCalledWith(
        'graphChanged',
        expect.any(Function)
      )

      if (minimap.visible.value) {
        expect(mockResume).toHaveBeenCalled()
      }
    })

    it('should not initialize without canvas and graph', async () => {
      const originalCanvas = defaultCanvasStore.canvas
      defaultCanvasStore.canvas = null

      const minimap = useMinimap()
      await minimap.init()

      expect(minimap.initialized.value).toBe(false)
      expect(api.addEventListener).not.toHaveBeenCalled()

      defaultCanvasStore.canvas = originalCanvas
    })

    it('should setup event listeners on graph', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      expect(moduleMockGraph.onNodeAdded).toBeDefined()
      expect(moduleMockGraph.onNodeRemoved).toBeDefined()
      expect(moduleMockGraph.onConnectionChange).toBeDefined()
    })

    it('should handle visibility from settings', async () => {
      defaultSettingStore.get.mockReturnValue(false)
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      expect(minimap.visible.value).toBe(false)
      expect(mockResume).not.toHaveBeenCalled()
    })
  })

  describe('destroy', () => {
    it('should cleanup all resources', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()
      minimap.destroy()

      expect(mockPause).toHaveBeenCalled()
      expect(api.removeEventListener).toHaveBeenCalledWith(
        'graphChanged',
        expect.any(Function)
      )
      expect(window.removeEventListener).toHaveBeenCalled()
      expect(minimap.initialized.value).toBe(false)
    })

    it('should restore original graph callbacks', async () => {
      const originalCallbacks = {
        onNodeAdded: vi.fn(),
        onNodeRemoved: vi.fn(),
        onConnectionChange: vi.fn()
      }

      moduleMockGraph.onNodeAdded = originalCallbacks.onNodeAdded
      moduleMockGraph.onNodeRemoved = originalCallbacks.onNodeRemoved
      moduleMockGraph.onConnectionChange = originalCallbacks.onConnectionChange

      const minimap = await createAndInitializeMinimap()

      await minimap.init()
      minimap.destroy()

      expect(moduleMockGraph.onNodeAdded).toBe(originalCallbacks.onNodeAdded)
      expect(moduleMockGraph.onNodeRemoved).toBe(
        originalCallbacks.onNodeRemoved
      )
      expect(moduleMockGraph.onConnectionChange).toBe(
        originalCallbacks.onConnectionChange
      )
    })
  })

  describe('toggle', () => {
    it('should toggle visibility and save to settings', async () => {
      const minimap = useMinimap()
      const initialVisibility = minimap.visible.value

      await minimap.toggle()

      expect(minimap.visible.value).toBe(!initialVisibility)
      expect(defaultSettingStore.set).toHaveBeenCalledWith(
        'Comfy.Minimap.Visible',
        !initialVisibility
      )

      await minimap.toggle()

      expect(minimap.visible.value).toBe(initialVisibility)
      expect(defaultSettingStore.set).toHaveBeenCalledWith(
        'Comfy.Minimap.Visible',
        initialVisibility
      )
    })
  })

  describe('rendering', () => {
    it('should verify context is obtained during render', async () => {
      const minimap = await createAndInitializeMinimap()

      const getContextSpy = vi.spyOn(moduleMockCanvasElement, 'getContext')

      await minimap.init()

      // Force initial render
      minimap.renderMinimap()

      // Force a render by triggering a graph change
      moduleMockGraph._nodes.push({
        id: 'new-node',
        pos: [150, 150],
        size: [100, 50],
        constructor: { color: '#666' },
        outputs: []
      })

      // Trigger RAF to process changes
      await triggerRAF()
      await nextTick()

      expect(getContextSpy).toHaveBeenCalled()
      expect(getContextSpy).toHaveBeenCalledWith('2d')
    })

    it('should render at least once after initialization', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      // Force initial render
      minimap.renderMinimap()

      // Force a render by modifying a node position
      moduleMockGraph._nodes[0].pos = [50, 50]

      // Trigger RAF to process changes
      await triggerRAF()
      await nextTick()

      const renderingOccurred =
        vi.mocked(mockContext2D.clearRect).mock.calls.length > 0 ||
        vi.mocked(mockContext2D.fillRect).mock.calls.length > 0

      if (!renderingOccurred) {
        console.log('Minimap visible:', minimap.visible.value)
        console.log('Minimap initialized:', minimap.initialized.value)
        console.log('Canvas exists:', !!defaultCanvasStore.canvas)
        console.log('Graph exists:', !!defaultCanvasStore.canvas?.graph)
        console.log(
          'clearRect calls:',
          vi.mocked(mockContext2D.clearRect).mock.calls.length
        )
        console.log(
          'fillRect calls:',
          vi.mocked(mockContext2D.fillRect).mock.calls.length
        )
        console.log(
          'getContext calls:',
          vi.mocked(moduleMockCanvasElement.getContext).mock.calls.length
        )
      }

      expect(renderingOccurred).toBe(true)
    })

    it('should not render when context is null', async () => {
      moduleMockCanvasElement = createMockMinimapCanvas({
        getContext: vi
          .fn()
          .mockImplementation(() => null) as HTMLCanvasElement['getContext']
      })

      const minimap = await createAndInitializeMinimap()

      await minimap.init()
      await new Promise((resolve) => setTimeout(resolve, 100))

      expect(mockContext2D.clearRect).not.toHaveBeenCalled()

      moduleMockCanvasElement = createMockMinimapCanvas({
        getContext: vi
          .fn()
          .mockImplementation((contextId) =>
            contextId === '2d' ? mockContext2D : null
          ) as HTMLCanvasElement['getContext']
      })
    })

    it('should handle empty graph', async () => {
      const originalNodes = [...moduleMockGraph._nodes]
      moduleMockGraph._nodes = []

      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      // The renderer has a fast path for empty graphs, force it to execute
      minimap.renderMinimap()

      await new Promise((resolve) => setTimeout(resolve, 100))

      expect(minimap.initialized.value).toBe(true)

      // With the new reactive system, the minimap may still render some elements
      // The key test is that it doesn't crash and properly initializes
      expect(mockContext2D.clearRect).toHaveBeenCalled()

      moduleMockGraph._nodes = originalNodes
    })
  })

  describe('pointer interactions', () => {
    it('should handle pointer down and start dragging (mouse)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'mouse'
      })

      minimap.handlePointerDown(pointerEvent)

      expect(mockContainerElement.getBoundingClientRect).toHaveBeenCalled()
      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
    })

    it('should handle pointer move while dragging (mouse)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerDownEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'mouse'
      })
      minimap.handlePointerDown(pointerDownEvent)

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'mouse'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
      expect(moduleMockCanvas.ds.offset).toBeDefined()
    })

    it('should handle pointer up to stop dragging (mouse)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerDownEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'mouse'
      })
      minimap.handlePointerDown(pointerDownEvent)

      minimap.handlePointerUp()

      moduleMockCanvas.setDirty.mockClear()

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'mouse'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).not.toHaveBeenCalled()
    })

    it('should handle pointer down and start dragging (touch)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'touch'
      })

      minimap.handlePointerDown(pointerEvent)

      expect(mockContainerElement.getBoundingClientRect).toHaveBeenCalled()
      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
    })

    it('should handle pointer move while dragging (touch)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerDownEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'touch'
      })
      minimap.handlePointerDown(pointerDownEvent)

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'touch'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
      expect(moduleMockCanvas.ds.offset).toBeDefined()
    })

    it('should handle pointer move while dragging (pen)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerDownEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'pen'
      })
      minimap.handlePointerDown(pointerDownEvent)

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'pen'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
      expect(moduleMockCanvas.ds.offset).toBeDefined()
    })

    it('should not move when not dragging with pointer', async () => {
      const minimap = await createAndInitializeMinimap()

      moduleMockCanvas.setDirty.mockClear()

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'touch'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).not.toHaveBeenCalled()
    })

    it('should handle pointer up to stop dragging (touch)', async () => {
      const minimap = await createAndInitializeMinimap()

      const pointerDownEvent = new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 150,
        pointerType: 'touch'
      })
      minimap.handlePointerDown(pointerDownEvent)

      minimap.handlePointerUp()

      moduleMockCanvas.setDirty.mockClear()

      const pointerMoveEvent = new PointerEvent('pointermove', {
        clientX: 200,
        clientY: 200,
        pointerType: 'touch'
      })
      minimap.handlePointerMove(pointerMoveEvent)

      expect(moduleMockCanvas.setDirty).not.toHaveBeenCalled()
    })
  })

  describe('wheel interactions', () => {
    it('should handle wheel zoom in', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      const wheelEvent = new WheelEvent('wheel', {
        deltaY: -100,
        clientX: 150,
        clientY: 150
      })

      const preventDefault = vi.fn()
      Object.defineProperty(wheelEvent, 'preventDefault', {
        value: preventDefault
      })

      minimap.handleWheel(wheelEvent)

      expect(preventDefault).toHaveBeenCalled()
      expect(moduleMockCanvas.ds.scale).toBeCloseTo(1.1)
      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
    })

    it('should handle wheel zoom out', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      const wheelEvent = new WheelEvent('wheel', {
        deltaY: 100,
        clientX: 150,
        clientY: 150
      })

      const preventDefault = vi.fn()
      Object.defineProperty(wheelEvent, 'preventDefault', {
        value: preventDefault
      })

      minimap.handleWheel(wheelEvent)

      expect(preventDefault).toHaveBeenCalled()
      expect(moduleMockCanvas.ds.scale).toBeCloseTo(0.9)
      expect(moduleMockCanvas.setDirty).toHaveBeenCalledWith(true, true)
    })

    it('should respect zoom limits', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      moduleMockCanvas.ds.scale = 0.1

      const wheelEvent = new WheelEvent('wheel', {
        deltaY: 100,
        clientX: 150,
        clientY: 150
      })

      const preventDefault = vi.fn()
      Object.defineProperty(wheelEvent, 'preventDefault', {
        value: preventDefault
      })

      minimap.handleWheel(wheelEvent)

      expect(moduleMockCanvas.ds.scale).toBe(0.1)
    })

    it('should update container rect if needed', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      const wheelEvent = new WheelEvent('wheel', {
        deltaY: -100,
        clientX: 150,
        clientY: 150
      })

      const preventDefault = vi.fn()
      Object.defineProperty(wheelEvent, 'preventDefault', {
        value: preventDefault
      })

      minimap.handleWheel(wheelEvent)

      expect(mockContainerElement.getBoundingClientRect).toHaveBeenCalled()
    })
  })

  describe('viewport updates', () => {
    it('should update viewport transform correctly', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()
      await nextTick()

      const viewportStyles = minimap.viewportStyles.value

      expect(viewportStyles).toBeDefined()
      expect(viewportStyles.transform).toMatch(
        /translate\(-?\d+(\.\d+)?px, -?\d+(\.\d+)?px\)/
      )
      expect(viewportStyles.width).toMatch(/\d+(\.\d+)?px/)
      expect(viewportStyles.height).toMatch(/\d+(\.\d+)?px/)
      expect(viewportStyles.border).toBe('2px solid #FFF')
    })

    it('should handle canvas dimension updates', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      moduleMockCanvas.canvas.clientWidth = 1200
      moduleMockCanvas.canvas.clientHeight = 900

      const resizeHandler = vi
        .mocked(window.addEventListener)
        .mock.calls.find((call) => call[0] === 'resize')?.[1] as
        | EventListener
        | undefined

      if (resizeHandler) {
        resizeHandler(new Event('resize'))
      }

      await nextTick()

      expect(minimap.viewportStyles.value).toBeDefined()
    })
  })

  describe('graph change handling', () => {
    it('should handle node addition', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      const newNode = {
        id: 'node3',
        pos: [300, 200],
        size: [100, 100],
        constructor: { color: '#666' },
        outputs: []
      }

      moduleMockGraph._nodes.push(newNode)
      if (moduleMockGraph.onNodeAdded) {
        moduleMockGraph.onNodeAdded(newNode)
      }

      await new Promise((resolve) => setTimeout(resolve, 600))
    })

    it('should handle node removal', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      const removedNode = moduleMockGraph._nodes[0]
      moduleMockGraph._nodes.splice(0, 1)

      if (moduleMockGraph.onNodeRemoved) {
        moduleMockGraph.onNodeRemoved(removedNode)
      }

      await new Promise((resolve) => setTimeout(resolve, 600))
    })

    it('should handle connection changes', async () => {
      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      if (moduleMockGraph.onConnectionChange) {
        moduleMockGraph.onConnectionChange(moduleMockGraph._nodes[0])
      }

      await new Promise((resolve) => setTimeout(resolve, 600))
    })
  })

  describe('container styles', () => {
    it('should provide correct container styles for dark theme', () => {
      const minimap = useMinimap()
      const styles = minimap.containerStyles.value

      expect(styles.width).toBe('253px')
      expect(styles.height).toBe('200px')
      expect(styles.border).toBe('1px solid var(--interface-stroke)')
      expect(styles.borderRadius).toBe('8px')
    })
  })

  describe('edge cases', () => {
    it('should handle missing node outputs', async () => {
      moduleMockGraph._nodes[0].outputs = null
      const minimap = await createAndInitializeMinimap()

      await expect(minimap.init()).resolves.not.toThrow()
      expect(minimap.initialized.value).toBe(true)
    })

    it('should handle invalid link references', async () => {
      moduleMockGraph.links.link1.target_id = 'invalid-node'
      moduleMockGraph.getNodeById.mockReturnValue(null)

      const minimap = await createAndInitializeMinimap()

      await expect(minimap.init()).resolves.not.toThrow()
      expect(minimap.initialized.value).toBe(true)
    })

    it('should handle high DPI displays', async () => {
      window.devicePixelRatio = 2

      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      expect(minimap.initialized.value).toBe(true)
    })

    it('should handle nodes without color', async () => {
      moduleMockGraph._nodes[0].color = undefined

      const minimap = await createAndInitializeMinimap()

      await minimap.init()

      minimap.renderMinimap()

      expect(mockContext2D.fillRect).toHaveBeenCalled()
      expect(mockContext2D.fillStyle).toBeDefined()
    })
  })

  describe('setMinimapRef', () => {
    it('should set minimap reference', () => {
      const minimap = useMinimap()
      const ref = document.createElement('div')

      minimap.setMinimapRef(ref)

      expect(() => minimap.setMinimapRef(ref)).not.toThrow()
    })
  })
})
