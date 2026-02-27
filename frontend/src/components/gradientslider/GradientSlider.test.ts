import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import GradientSlider from './GradientSlider.vue'
import type { ColorStop } from '@/lib/litegraph/src/interfaces'
import { interpolateStops, stopsToGradient } from './gradients'

const TEST_STOPS: ColorStop[] = [
  { offset: 0, color: [0, 0, 0] },
  { offset: 1, color: [255, 255, 255] }
]

function mountSlider(props: {
  stops?: ColorStop[]
  modelValue: number
  min?: number
  max?: number
  step?: number
}) {
  return mount(GradientSlider, {
    props: { stops: TEST_STOPS, ...props }
  })
}

describe('GradientSlider', () => {
  it('passes min, max, step to SliderRoot', () => {
    const wrapper = mountSlider({
      modelValue: 50,
      min: -100,
      max: 100,
      step: 5
    })
    const thumb = wrapper.find('[role="slider"]')
    expect(thumb.attributes('aria-valuemin')).toBe('-100')
    expect(thumb.attributes('aria-valuemax')).toBe('100')
  })

  it('renders slider root with track and thumb', () => {
    const wrapper = mountSlider({ modelValue: 0 })
    expect(wrapper.find('[data-slider-impl]').exists()).toBe(true)
    expect(wrapper.find('[role="slider"]').exists()).toBe(true)
  })

  it('does not render SliderRange', () => {
    const wrapper = mountSlider({ modelValue: 50 })
    expect(wrapper.find('[data-slot="slider-range"]').exists()).toBe(false)
  })
})

describe('stopsToGradient', () => {
  it('returns transparent for empty stops', () => {
    expect(stopsToGradient([])).toBe('transparent')
  })
})

describe('interpolateStops', () => {
  it('returns transparent for empty stops', () => {
    expect(interpolateStops([], 0.5)).toBe('transparent')
  })

  it('returns start color at t=0', () => {
    expect(interpolateStops(TEST_STOPS, 0)).toBe('rgb(0,0,0)')
  })

  it('returns end color at t=1', () => {
    expect(interpolateStops(TEST_STOPS, 1)).toBe('rgb(255,255,255)')
  })

  it('returns midpoint color at t=0.5', () => {
    expect(interpolateStops(TEST_STOPS, 0.5)).toBe('rgb(128,128,128)')
  })

  it('clamps values below 0', () => {
    expect(interpolateStops(TEST_STOPS, -1)).toBe('rgb(0,0,0)')
  })

  it('clamps values above 1', () => {
    expect(interpolateStops(TEST_STOPS, 2)).toBe('rgb(255,255,255)')
  })
})
