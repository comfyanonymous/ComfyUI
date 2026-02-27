import { describe, expect, it } from 'vitest'

describe('PrimitiveFloat widget type bridging', () => {
  function createMockNodeAndWidget() {
    const properties: Record<string, unknown> = {}
    const options: Record<string, unknown> = {
      min: -Infinity,
      max: Infinity
    }
    const widget = { type: 'number', options, value: 0, callback: undefined }
    return { properties, widget }
  }

  function applyFloatPropertyBridges(
    properties: Record<string, unknown>,
    widget: { type: string; options: Record<string, unknown> }
  ) {
    const DISPLAY_WIDGET_TYPES = new Set(['gradientslider', 'slider', 'knob'])

    let baseType = widget.type
    Object.defineProperty(widget, 'type', {
      get: () => {
        const display = properties.display as string | undefined
        if (display && DISPLAY_WIDGET_TYPES.has(display)) return display
        return baseType
      },
      set: (v: string) => {
        baseType = v
      }
    })

    Object.defineProperty(widget.options, 'gradient_stops', {
      get: () => properties.gradient_stops,
      set: (v) => {
        properties.gradient_stops = v
      }
    })
  }

  it('returns base type when display property is not set', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    expect(widget.type).toBe('number')
  })

  it('returns gradientslider when display property is set', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    properties.display = 'gradientslider'
    expect(widget.type).toBe('gradientslider')
  })

  it('returns slider when display property is slider', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    properties.display = 'slider'
    expect(widget.type).toBe('slider')
  })

  it('returns base type for unknown display values', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    properties.display = 'unknown'
    expect(widget.type).toBe('number')
  })

  it('bridges gradient_stops from properties to options', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    expect(widget.options.gradient_stops).toBeUndefined()

    const stops = [
      { offset: 0, color: [255, 0, 0] },
      { offset: 1, color: [0, 0, 255] }
    ]
    properties.gradient_stops = stops
    expect(widget.options.gradient_stops).toBe(stops)
  })

  it('writes gradient_stops back to properties', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    const stops = [
      { offset: 0, color: [0, 0, 0] },
      { offset: 1, color: [255, 255, 255] }
    ]
    widget.options.gradient_stops = stops
    expect(properties.gradient_stops).toBe(stops)
  })

  it('allows type to be set and overridden', () => {
    const { properties, widget } = createMockNodeAndWidget()
    applyFloatPropertyBridges(properties, widget)

    widget.type = 'slider'
    expect(widget.type).toBe('slider')

    properties.display = 'gradientslider'
    expect(widget.type).toBe('gradientslider')
  })
})
