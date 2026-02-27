import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, nextTick } from 'vue'

import { useTransformState } from '@/renderer/core/layout/transform/useTransformState'
import { createMockCanvas } from '@/utils/__tests__/litegraphTestUtils'

import TransformPane from '../transform/TransformPane.vue'

const mockData = vi.hoisted(() => ({
  mockTransformStyle: {
    transform: 'scale(1) translate(0px, 0px)',
    transformOrigin: '0 0'
  },
  mockCamera: { x: 0, y: 0, z: 1 }
}))

vi.mock('@/renderer/core/layout/transform/useTransformState', () => {
  const syncWithCanvas = vi.fn()
  return {
    useTransformState: () => ({
      camera: computed(() => mockData.mockCamera),
      transformStyle: computed(() => mockData.mockTransformStyle),
      canvasToScreen: vi.fn(),
      screenToCanvas: vi.fn(),
      isNodeInViewport: vi.fn(),
      syncWithCanvas
    })
  }
})

function createMockLGraphCanvas() {
  return createMockCanvas({
    canvas: {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    },
    ds: {
      offset: [0, 0],
      scale: 1
    }
  })
}

describe('TransformPane', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.resetAllMocks()

    // Create mock canvas with LiteGraph interface
  })

  describe('component mounting', () => {
    it('should mount successfully with minimal props', () => {
      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      expect(wrapper.exists()).toBe(true)
      expect(wrapper.find('[data-testid="transform-pane"]').exists()).toBe(true)
    })

    it('should apply transform style from composable', async () => {
      mockData.mockTransformStyle = {
        transform: 'scale(2) translate(100px, 50px)',
        transformOrigin: '0 0'
      }

      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })
      await nextTick()

      const transformPane = wrapper.find('[data-testid="transform-pane"]')
      const style = transformPane.attributes('style')
      expect(style).toContain('transform: scale(2) translate(100px, 50px)')
    })

    it('should render slot content', () => {
      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        },
        slots: {
          default: '<div class="test-content">Test Node</div>'
        }
      })

      expect(wrapper.find('.test-content').exists()).toBe(true)
      expect(wrapper.find('.test-content').text()).toBe('Test Node')
    })
  })

  describe('RAF synchronization', () => {
    it('should call syncWithCanvas during RAF updates', async () => {
      const mockCanvas = createMockLGraphCanvas()
      mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      await nextTick()

      // Allow RAF to execute
      vi.advanceTimersToNextFrame()

      const transformState = useTransformState()
      expect(transformState.syncWithCanvas).toHaveBeenCalledWith(mockCanvas)
    })
  })

  describe('canvas event listeners', () => {
    it('should add event listeners to canvas on mount', async () => {
      const mockCanvas = createMockLGraphCanvas()
      mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      await nextTick()

      expect(mockCanvas.canvas.addEventListener).toHaveBeenCalledWith(
        'wheel',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.addEventListener).not.toHaveBeenCalledWith(
        'pointerdown',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.addEventListener).not.toHaveBeenCalledWith(
        'pointerup',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.addEventListener).not.toHaveBeenCalledWith(
        'pointercancel',
        expect.any(Function),
        expect.any(Object)
      )
    })

    it('should remove event listeners on unmount', async () => {
      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      await nextTick()
      wrapper.unmount()

      expect(mockCanvas.canvas.removeEventListener).toHaveBeenCalledWith(
        'wheel',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.removeEventListener).not.toHaveBeenCalledWith(
        'pointerdown',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.removeEventListener).not.toHaveBeenCalledWith(
        'pointerup',
        expect.any(Function),
        expect.any(Object)
      )
      expect(mockCanvas.canvas.removeEventListener).not.toHaveBeenCalledWith(
        'pointercancel',
        expect.any(Function),
        expect.any(Object)
      )
    })
  })

  describe('interaction state management', () => {
    it('should apply interacting class during interactions', async () => {
      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      // Simulate interaction start by checking internal state
      // Note: This tests the CSS class application logic
      const transformPane = wrapper.find('[data-testid="transform-pane"]')

      // Initially should not have interacting class
      expect(transformPane.classes()).not.toContain(
        'transform-pane--interacting'
      )
    })

    it('should handle pointer events for node delegation', async () => {
      const mockCanvas = createMockLGraphCanvas()
      const wrapper = mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      const transformPane = wrapper.find('[data-testid="transform-pane"]')

      // Simulate pointer down - we can't test the exact delegation logic
      // in unit tests due to vue-test-utils limitations, but we can verify
      // the event handler is set up correctly
      await transformPane.trigger('pointerdown')

      // The test passes if no errors are thrown during event handling
      expect(transformPane.exists()).toBe(true)
    })
  })

  describe('transform state integration', () => {
    it('should provide transform utilities to child components', () => {
      const mockCanvas = createMockLGraphCanvas()
      mount(TransformPane, {
        props: {
          canvas: mockCanvas
        }
      })

      const transformState = useTransformState()
      // The component should provide transform state via Vue's provide/inject
      // This is tested indirectly through the composable integration
      expect(transformState.syncWithCanvas).toBeDefined()
      expect(transformState.canvasToScreen).toBeDefined()
      expect(transformState.screenToCanvas).toBeDefined()
    })
  })

  describe('error handling', () => {
    it('should handle null canvas gracefully', () => {
      const wrapper = mount(TransformPane, {
        props: {
          canvas: undefined
        }
      })

      expect(wrapper.exists()).toBe(true)
      expect(wrapper.find('[data-testid="transform-pane"]').exists()).toBe(true)
    })
  })
})
