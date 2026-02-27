import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import type { LGraphNode as LGraphNodeType } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type { IColorWidget } from '@/lib/litegraph/src/types/widgets'
import type { ColorWidget as ColorWidgetType } from '@/lib/litegraph/src/widgets/ColorWidget'

type LGraphCanvasType = InstanceType<typeof LGraphCanvas>

function createMockWidgetConfig(
  overrides: Partial<IColorWidget> = {}
): IColorWidget {
  return {
    type: 'color',
    name: 'test_color',
    value: '#ff0000',
    options: {},
    y: 0,
    ...overrides
  }
}

function createMockCanvas(): LGraphCanvasType {
  return {
    setDirty: vi.fn()
  } as Partial<LGraphCanvasType> as LGraphCanvasType
}

function createMockEvent(clientX = 100, clientY = 200): CanvasPointerEvent {
  return { clientX, clientY } as CanvasPointerEvent
}

describe('ColorWidget', () => {
  let node: LGraphNodeType
  let widget: ColorWidgetType
  let mockCanvas: LGraphCanvasType
  let mockEvent: CanvasPointerEvent
  let ColorWidget: typeof ColorWidgetType
  let LGraphNode: typeof LGraphNodeType

  beforeEach(async () => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    // Reset modules to get fresh globalColorInput state
    vi.resetModules()

    const litegraph = await import('@/lib/litegraph/src/litegraph')
    LGraphNode = litegraph.LGraphNode

    const colorWidgetModule =
      await import('@/lib/litegraph/src/widgets/ColorWidget')
    ColorWidget = colorWidgetModule.ColorWidget

    node = new LGraphNode('TestNode')
    mockCanvas = createMockCanvas()
    mockEvent = createMockEvent()
  })

  afterEach(() => {
    vi.useRealTimers()
    document
      .querySelectorAll('input[type="color"]')
      .forEach((el) => el.remove())
  })

  describe('onClick', () => {
    it('should create a color input and append it to document body', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      expect(input).toBeTruthy()
      expect(input.parentElement).toBe(document.body)
    })

    it('should set input value from widget value', () => {
      widget = new ColorWidget(
        createMockWidgetConfig({ value: '#00ff00' }),
        node
      )

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      expect(input.value).toBe('#00ff00')
    })

    it('should default to #000000 when widget value is empty', () => {
      widget = new ColorWidget(createMockWidgetConfig({ value: '' }), node)

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      expect(input.value).toBe('#000000')
    })

    it('should position input at click coordinates', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)
      const event = createMockEvent(150, 250)

      widget.onClick({ e: event, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      expect(input.style.left).toBe('150px')
      expect(input.style.top).toBe('250px')
    })

    it('should click the input on next animation frame', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      const clickSpy = vi.spyOn(input, 'click')

      expect(clickSpy).not.toHaveBeenCalled()
      vi.runAllTimers()
      expect(clickSpy).toHaveBeenCalled()
    })

    it('should reuse the same input element on subsequent clicks', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })
      const firstInput = document.querySelector('input[type="color"]')

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })
      const secondInput = document.querySelector('input[type="color"]')

      expect(firstInput).toBe(secondInput)
      expect(document.querySelectorAll('input[type="color"]').length).toBe(1)
    })

    it('should update input value when clicking with different widget values', () => {
      const widget1 = new ColorWidget(
        createMockWidgetConfig({ value: '#ff0000' }),
        node
      )
      const widget2 = new ColorWidget(
        createMockWidgetConfig({ value: '#0000ff' }),
        node
      )

      widget1.onClick({ e: mockEvent, node, canvas: mockCanvas })
      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      expect(input.value).toBe('#ff0000')

      widget2.onClick({ e: mockEvent, node, canvas: mockCanvas })
      expect(input.value).toBe('#0000ff')
    })
  })

  describe('onChange', () => {
    it('should call setValue when color input changes', () => {
      widget = new ColorWidget(
        createMockWidgetConfig({ value: '#ff0000' }),
        node
      )
      const setValueSpy = vi.spyOn(widget, 'setValue')

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      input.value = '#00ff00'
      input.dispatchEvent(new Event('change'))

      expect(setValueSpy).toHaveBeenCalledWith('#00ff00', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should call canvas.setDirty after value change', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      input.value = '#00ff00'
      input.dispatchEvent(new Event('change'))

      expect(mockCanvas.setDirty).toHaveBeenCalledWith(true)
    })

    it('should remove change listener after firing once', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)
      const setValueSpy = vi.spyOn(widget, 'setValue')

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement

      input.value = '#00ff00'
      input.dispatchEvent(new Event('change'))
      input.value = '#0000ff'
      input.dispatchEvent(new Event('change'))

      // Should only be called once despite two change events
      expect(setValueSpy).toHaveBeenCalledTimes(1)
      expect(setValueSpy).toHaveBeenCalledWith('#00ff00', expect.any(Object))
    })

    it('should register new change listener on subsequent onClick', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)
      const setValueSpy = vi.spyOn(widget, 'setValue')

      // First click and change
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })
      const input = document.querySelector(
        'input[type="color"]'
      ) as HTMLInputElement
      input.value = '#00ff00'
      input.dispatchEvent(new Event('change'))

      // Second click and change
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })
      input.value = '#0000ff'
      input.dispatchEvent(new Event('change'))

      expect(setValueSpy).toHaveBeenCalledTimes(2)
      expect(setValueSpy).toHaveBeenNthCalledWith(
        1,
        '#00ff00',
        expect.any(Object)
      )
      expect(setValueSpy).toHaveBeenNthCalledWith(
        2,
        '#0000ff',
        expect.any(Object)
      )
    })
  })

  describe('type', () => {
    it('should have type "color"', () => {
      widget = new ColorWidget(createMockWidgetConfig(), node)
      expect(widget.type).toBe('color')
    })
  })
})
