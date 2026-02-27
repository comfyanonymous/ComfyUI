import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as LGraphCanvasModule from '@/lib/litegraph/src/LGraphCanvas'
import { LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type { IComboWidget } from '@/lib/litegraph/src/types/widgets'
import { ComboWidget } from '@/lib/litegraph/src/widgets/ComboWidget'

const { LGraphCanvas } = await vi.importActual<typeof LGraphCanvasModule>(
  '@/lib/litegraph/src/LGraphCanvas'
)
type LGraphCanvasType = InstanceType<typeof LGraphCanvas>

interface MockWidgetConfig extends Omit<IComboWidget, 'options'> {
  options: IComboWidget['options']
}

function createMockWidgetConfig(
  overrides: Partial<MockWidgetConfig> = {}
): MockWidgetConfig {
  return {
    type: 'combo',
    name: 'test',
    value: '',
    options: { values: [] },
    y: 0,
    ...overrides
  }
}

function setupIncrementDecrementTest() {
  const mockCanvas = {
    ds: { scale: 1 },
    last_mouseclick: 1
  } as LGraphCanvasType
  const mockEvent = {} as CanvasPointerEvent
  return { mockCanvas, mockEvent }
}

describe('ComboWidget', () => {
  let node: LGraphNode
  let widget: ComboWidget

  beforeEach(() => {
    vi.clearAllMocks()
    node = new LGraphNode('TestNode')
  })

  describe('_displayValue', () => {
    it('should return value as-is for array values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'slow', 'medium'] }
        }),
        node
      )

      expect(widget._displayValue).toBe('fast')
    })

    it('should return mapped value for object values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'quality',
          value: 'hq',
          options: {
            values: {
              hq: 'High Quality',
              mq: 'Medium Quality',
              lq: 'Low Quality'
            }
          }
        }),
        node
      )

      expect(widget._displayValue).toBe('High Quality')
    })

    it('should return empty string when disabled', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'slow'] },
          computedDisabled: true
        }),
        node
      )

      expect(widget._displayValue).toBe('')
    })

    it('should convert number values to string before display', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'index',
          value: 42,
          options: { values: ['0', '1', '42'] }
        }),
        node
      )

      expect(widget._displayValue).toBe('42')
    })
  })

  describe('canIncrement / canDecrement', () => {
    it('should return true when not at end/start of list', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'medium',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      expect(widget.canIncrement()).toBe(true)
      expect(widget.canDecrement()).toBe(true)
    })

    it('should return false from canDecrement when at first value', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      expect(widget.canDecrement()).toBe(false)
      expect(widget.canIncrement()).toBe(true)
    })

    it('should return false from canIncrement when at last value', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'slow',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      expect(widget.canIncrement()).toBe(false)
      expect(widget.canDecrement()).toBe(true)
    })

    it('should return false when list has only one item', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'only',
          options: { values: ['only'] }
        }),
        node
      )

      expect(widget.canIncrement()).toBe(false)
      expect(widget.canDecrement()).toBe(false)
    })

    it('should allow increment/decrement when duplicate values exist at different indices', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'duplicate',
          options: { values: ['duplicate', 'other', 'duplicate'] }
        }),
        node
      )

      expect(widget.canIncrement()).toBe(true)
      expect(widget.canDecrement()).toBe(true)
    })

    it('should return false for function values (DEPRECATED - legacy duck-typed behavior)', () => {
      const valuesFn = vi.fn().mockReturnValue(['a', 'b', 'c'])
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'b',
          options: { values: valuesFn }
        }),
        node
      )

      // Function values are legacy - should be permissive (return false)
      expect(widget.canIncrement()).toBe(false)
      expect(widget.canDecrement()).toBe(false)
    })
  })

  describe('incrementValue / decrementValue', () => {
    it('should increment value to next in list', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.incrementValue({ e: mockEvent, node, canvas: mockCanvas })

      expect(setValueSpy).toHaveBeenCalledWith('medium', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
      expect(mockCanvas.last_mouseclick).toBe(0) // Avoid double click event
    })

    it('should decrement value to previous in list', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'medium',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.decrementValue({ e: mockEvent, node, canvas: mockCanvas })

      expect(setValueSpy).toHaveBeenCalledWith('fast', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
      expect(mockCanvas.last_mouseclick).toBe(0)
    })

    it('should clamp at last value when incrementing beyond', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'slow',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.incrementValue({ e: mockEvent, node, canvas: mockCanvas })

      // Should stay at 'slow' (last value)
      expect(setValueSpy).toHaveBeenCalledWith('slow', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should clamp at first value when decrementing beyond', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.decrementValue({ e: mockEvent, node, canvas: mockCanvas })

      // Should stay at 'fast' (first value)
      expect(setValueSpy).toHaveBeenCalledWith('fast', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should set value to index position when incrementing object-type values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'quality',
          value: 'hq',
          options: {
            values: {
              hq: 'High Quality',
              mq: 'Medium Quality'
            }
          }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.incrementValue({ e: mockEvent, node, canvas: mockCanvas })

      // For object values, setValue receives the index
      expect(setValueSpy).toHaveBeenCalledWith(1, {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })
  })

  describe('onClick', () => {
    it('should decrement value when left arrow clicked', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'medium',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const mockCanvas = {
        ds: { scale: 1 },
        last_mouseclick: 0
      } as LGraphCanvasType
      const mockEvent = { canvasX: 60 } as CanvasPointerEvent // 60 - 50 = 10 < 40 (left arrow)
      node.pos = [50, 50]

      const decrementSpy = vi.spyOn(widget, 'decrementValue')
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      expect(decrementSpy).toHaveBeenCalledWith({
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should increment value when right arrow clicked', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'medium',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const mockCanvas = {
        ds: { scale: 1 },
        last_mouseclick: 0
      } as LGraphCanvasType
      const mockEvent = { canvasX: 240 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      const incrementSpy = vi.spyOn(widget, 'incrementValue')
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      expect(incrementSpy).toHaveBeenCalledWith({
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should show dropdown menu when clicking center area with array values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'medium',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      const mockContextMenu = vi.fn()
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      expect(mockContextMenu).toHaveBeenCalledWith(
        ['fast', 'medium', 'slow'],
        expect.objectContaining({
          scale: 1,
          event: mockEvent,
          className: 'dark'
        })
      )
    })

    it('should show dropdown menu with object display values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'quality',
          value: 'mq',
          options: {
            values: {
              hq: 'High Quality',
              mq: 'Medium Quality',
              lq: 'Low Quality'
            }
          }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      const mockContextMenu = vi.fn()
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      // Should show the display values (values), not keys
      expect(mockContextMenu).toHaveBeenCalledWith(
        ['High Quality', 'Medium Quality', 'Low Quality'],
        expect.objectContaining({
          scale: 1,
          event: mockEvent,
          className: 'dark'
        })
      )
    })

    it('should set value when selecting from dropdown with array values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      let capturedCallback: ((value: string) => void) | undefined
      const mockContextMenu = vi
        .fn<typeof LiteGraph.ContextMenu>()
        .mockImplementation(function (_values, options) {
          capturedCallback = options.callback
        })
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      // Simulate selecting 'slow' from dropdown
      capturedCallback?.('slow')

      expect(setValueSpy).toHaveBeenCalledWith('slow', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should set value to selected index for object-type values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'quality',
          value: 'hq',
          options: {
            values: {
              hq: 'High Quality',
              mq: 'Medium Quality',
              lq: 'Low Quality'
            }
          }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      let capturedCallback: ((value: string) => void) | undefined
      const mockContextMenu = vi
        .fn<typeof LiteGraph.ContextMenu>()
        .mockImplementation(function (_values, options) {
          capturedCallback = options.callback
        })
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      // Simulate selecting 'Medium Quality' (index 1) from dropdown
      capturedCallback?.('Medium Quality')

      expect(setValueSpy).toHaveBeenCalledWith(1, {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })

    it('should prevent menu scaling below 100%', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'fast',
          options: { values: ['fast', 'slow'] }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 0.5 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      const mockContextMenu = vi.fn()
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      expect(mockContextMenu).toHaveBeenCalledWith(
        expect.any(Array),
        expect.objectContaining({
          scale: 1 // Math.max(1, 0.5) = 1
        })
      )
    })

    it('should warn when using deprecated function values', () => {
      const deprecationCallback = vi.fn()
      const originalCallbacks = LiteGraph.onDeprecationWarning
      LiteGraph.onDeprecationWarning = [deprecationCallback]

      const valuesFn = vi.fn().mockReturnValue(['a', 'b', 'c'])
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'a',
          options: { values: valuesFn }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      const mockContextMenu = vi.fn()
      LiteGraph.ContextMenu = mockContextMenu as Partial<
        typeof LiteGraph.ContextMenu
      > as typeof LiteGraph.ContextMenu

      widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

      expect(deprecationCallback).toHaveBeenCalledWith(
        'Using a function for values is deprecated. Use an array of unique values instead.',
        undefined
      )

      LiteGraph.onDeprecationWarning = originalCallbacks
    })
  })

  describe('with getOptionLabel', () => {
    const HASH_FILENAME =
      '72e786ff2a44d682c4294db0b7098e569832bc394efc6dad644e6ec85a78efb7.png'
    const HASH_FILENAME_2 =
      'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456.jpg'

    describe('_displayValue', () => {
      it('should return formatted value when getOptionLabel provided', () => {
        const mockGetOptionLabel = vi
          .fn()
          .mockReturnValue('Beautiful Sunset.png')

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        expect(widget._displayValue).toBe('Beautiful Sunset.png')
        expect(mockGetOptionLabel).toHaveBeenCalledWith(HASH_FILENAME)
      })

      it('should return original value when getOptionLabel not provided', () => {
        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: { values: [HASH_FILENAME] }
          }),
          node
        )

        expect(widget._displayValue).toBe(HASH_FILENAME)
      })

      it('should not call getOptionLabel when disabled', () => {
        const mockGetOptionLabel = vi.fn()

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME],
              getOptionLabel: mockGetOptionLabel
            },
            computedDisabled: true
          }),
          node
        )

        expect(widget._displayValue).toBe('')
        expect(mockGetOptionLabel).not.toHaveBeenCalled()
      })

      it('should handle getOptionLabel error gracefully', () => {
        const mockGetOptionLabel = vi.fn().mockImplementation(function () {
          throw new Error('Formatting failed')
        })
        const consoleErrorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {})

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        expect(widget._displayValue).toBe(HASH_FILENAME)
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          'Failed to map value:',
          expect.any(Error)
        )

        consoleErrorSpy.mockRestore()
      })

      it('should format non-hash filenames using getOptionLabel', () => {
        const mockGetOptionLabel = vi.fn((value) => `Formatted ${value}`)

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'file',
            value: 'regular-file.png',
            options: {
              values: ['regular-file.png'],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        expect(widget._displayValue).toBe('Formatted regular-file.png')
        expect(mockGetOptionLabel).toHaveBeenCalledWith('regular-file.png')
      })

      it('should use getOptionLabel over object value mapping when both present', () => {
        const mockGetOptionLabel = vi.fn((value) => `Label: ${value}`)

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'quality',
            value: 'hq',
            options: {
              values: {
                hq: 'High Quality',
                mq: 'Medium Quality'
              },
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        // getOptionLabel should take precedence over object value mapping
        expect(widget._displayValue).toBe('Label: hq')
        expect(mockGetOptionLabel).toHaveBeenCalledWith('hq')
      })

      it('should format number values using getOptionLabel when provided', () => {
        const mockGetOptionLabel = vi.fn((value) => `Number: ${value}`)

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'index',
            value: 42,
            options: {
              values: ['0', '1', '42'],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        expect(widget._displayValue).toBe('Number: 42')
        expect(mockGetOptionLabel).toHaveBeenCalledWith('42')
      })
    })

    describe('onClick', () => {
      it('should show dropdown with formatted labels', () => {
        const mockGetOptionLabel = vi
          .fn()
          .mockReturnValueOnce('Beautiful Sunset.png')
          .mockReturnValueOnce('Mountain Vista.jpg')

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME, HASH_FILENAME_2],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
        const mockEvent = { canvasX: 150 } as CanvasPointerEvent
        node.pos = [50, 50]
        node.size = [200, 30]

        const mockAddItem = vi.fn()
        const mockContextMenu = vi
          .fn<typeof LiteGraph.ContextMenu>()
          .mockImplementation(function () {
            this.addItem = mockAddItem
          })
        LiteGraph.ContextMenu = mockContextMenu as Partial<
          typeof LiteGraph.ContextMenu
        > as typeof LiteGraph.ContextMenu
        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

        // Should show formatted labels in dropdown
        expect(mockContextMenu).toHaveBeenCalledWith(
          [],
          expect.objectContaining({
            scale: 1,
            event: mockEvent,
            className: 'dark'
          })
        )

        expect(mockAddItem).toHaveBeenCalledWith(
          'Beautiful Sunset.png',
          HASH_FILENAME,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
        expect(mockAddItem).toHaveBeenCalledWith(
          'Mountain Vista.jpg',
          HASH_FILENAME_2,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
      })

      it('should set original value when selecting formatted label from dropdown', () => {
        const mockGetOptionLabel = vi
          .fn()
          .mockReturnValueOnce('Beautiful Sunset.png')
          .mockReturnValueOnce('Mountain Vista.jpg')

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME, HASH_FILENAME_2],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
        const mockEvent = { canvasX: 150 } as CanvasPointerEvent
        node.pos = [50, 50]
        node.size = [200, 30]

        const mockAddItem = vi.fn()
        let capturedCallback: ((value: string) => void) | undefined
        const mockContextMenu = vi
          .fn<typeof LiteGraph.ContextMenu>()
          .mockImplementation(function (_values, options) {
            capturedCallback = options.callback
            this.addItem = mockAddItem
          })
        LiteGraph.ContextMenu = mockContextMenu as Partial<
          typeof LiteGraph.ContextMenu
        > as typeof LiteGraph.ContextMenu

        const setValueSpy = vi.spyOn(widget, 'setValue')
        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

        // Simulate selecting second item (Mountain Vista.jpg -> HASH_FILENAME_2)
        capturedCallback?.(HASH_FILENAME_2)

        // Should set the actual hash value, not the formatted label
        expect(setValueSpy).toHaveBeenCalledWith(HASH_FILENAME_2, {
          e: mockEvent,
          node,
          canvas: mockCanvas
        })
      })

      it('should preserve value identity when multiple options have same display label', () => {
        const mockGetOptionLabel = vi
          .fn()
          .mockReturnValueOnce('sunset.png')
          .mockReturnValueOnce('sunset.png') // Same label, different values
          .mockReturnValueOnce('mountain.png')

        const hash1 = HASH_FILENAME
        const hash2 = HASH_FILENAME_2
        const hash3 = 'abc123def456.png'

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: hash1,
            options: {
              values: [hash1, hash2, hash3],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
        const mockEvent = { canvasX: 150 } as CanvasPointerEvent
        node.pos = [50, 50]
        node.size = [200, 30]

        const mockAddItem = vi.fn()
        let capturedCallback: ((value: string) => void) | undefined

        const mockContextMenu = vi
          .fn<typeof LiteGraph.ContextMenu>()
          .mockImplementation(function (_values, options) {
            capturedCallback = options.callback
            this.addItem = mockAddItem
          })
        LiteGraph.ContextMenu = mockContextMenu as Partial<
          typeof LiteGraph.ContextMenu
        > as typeof LiteGraph.ContextMenu

        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

        // Should use addItem API with separate name/value
        expect(mockAddItem).toHaveBeenCalledWith(
          'sunset.png',
          hash1,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
        expect(mockAddItem).toHaveBeenCalledWith(
          'sunset.png',
          hash2,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
        expect(mockAddItem).toHaveBeenCalledWith(
          'mountain.png',
          hash3,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )

        const setValueSpy = vi.spyOn(widget, 'setValue')

        // Simulate selecting the SECOND "sunset.png" (should pass hash2 directly)
        capturedCallback?.(hash2)

        // Should set hash2, not hash1 (fixes duplicate name bug)
        expect(setValueSpy).toHaveBeenCalledWith(hash2, {
          e: mockEvent,
          node,
          canvas: mockCanvas
        })
      })

      it('should handle getOptionLabel error in dropdown gracefully', () => {
        const mockGetOptionLabel = vi
          .fn()
          .mockReturnValueOnce('Beautiful Sunset.png')
          .mockImplementationOnce(function () {
            throw new Error('Formatting failed')
          })

        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME, HASH_FILENAME_2],
              getOptionLabel: mockGetOptionLabel
            }
          }),
          node
        )

        const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
        const mockEvent = { canvasX: 150 } as CanvasPointerEvent
        node.pos = [50, 50]
        node.size = [200, 30]

        const consoleErrorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {})

        const mockAddItem = vi.fn()
        const mockContextMenu = vi
          .fn<typeof LiteGraph.ContextMenu>()
          .mockImplementation(function () {
            this.addItem = mockAddItem
          })
        LiteGraph.ContextMenu = mockContextMenu as Partial<
          typeof LiteGraph.ContextMenu
        > as typeof LiteGraph.ContextMenu

        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

        // Should show formatted label for first, fallback to hash for second
        expect(mockAddItem).toHaveBeenCalledWith(
          'Beautiful Sunset.png',
          HASH_FILENAME,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
        expect(mockAddItem).toHaveBeenCalledWith(
          HASH_FILENAME_2,
          HASH_FILENAME_2,
          expect.objectContaining({
            callback: expect.any(Function),
            className: 'dark'
          })
        )
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          'Failed to map value:',
          expect.any(Error)
        )

        consoleErrorSpy.mockRestore()
      })

      it('should show hash values in dropdown when getOptionLabel not provided', () => {
        widget = new ComboWidget(
          createMockWidgetConfig({
            name: 'image',
            value: HASH_FILENAME,
            options: {
              values: [HASH_FILENAME, HASH_FILENAME_2]
            }
          }),
          node
        )

        const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
        const mockEvent = { canvasX: 150 } as CanvasPointerEvent
        node.pos = [50, 50]
        node.size = [200, 30]

        const mockContextMenu = vi.fn<typeof LiteGraph.ContextMenu>()
        LiteGraph.ContextMenu = mockContextMenu as Partial<
          typeof LiteGraph.ContextMenu
        > as typeof LiteGraph.ContextMenu

        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })

        // Should show hash filenames directly (no formatting)
        expect(mockContextMenu).toHaveBeenCalledWith(
          [HASH_FILENAME, HASH_FILENAME_2],
          expect.objectContaining({
            scale: 1,
            event: mockEvent,
            className: 'dark'
          })
        )
      })
    })
  })

  describe('edge cases', () => {
    it('should return empty display value and disallow increment/decrement for empty values', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: '',
          options: { values: [] }
        }),
        node
      )

      expect(widget._displayValue).toBe('')
      expect(widget.canIncrement()).toBe(false)
      expect(widget.canDecrement()).toBe(false)
    })

    it('should throw error when values is null in getValues', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'test',
          // @ts-expect-error - Testing with intentionally invalid null value
          options: { values: null }
        }),
        node
      )

      const mockCanvas = { ds: { scale: 1 } } as LGraphCanvasType
      const mockEvent = { canvasX: 150 } as CanvasPointerEvent
      node.pos = [50, 50]
      node.size = [200, 30]

      expect(() => {
        widget.onClick({ e: mockEvent, node, canvas: mockCanvas })
      }).toThrow('[ComboWidget]: values is required')
    })

    it('should default to first value when incrementing from invalid value', () => {
      widget = new ComboWidget(
        createMockWidgetConfig({
          name: 'mode',
          value: 'nonexistent',
          options: { values: ['fast', 'medium', 'slow'] }
        }),
        node
      )

      const { mockCanvas, mockEvent } = setupIncrementDecrementTest()

      const setValueSpy = vi.spyOn(widget, 'setValue')
      widget.incrementValue({ e: mockEvent, node, canvas: mockCanvas })

      // When value not found (indexOf returns -1), -1 + 1 = 0, clamped to 0
      expect(setValueSpy).toHaveBeenCalledWith('fast', {
        e: mockEvent,
        node,
        canvas: mockCanvas
      })
    })
  })
})
