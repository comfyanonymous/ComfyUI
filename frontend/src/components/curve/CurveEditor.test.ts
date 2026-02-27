import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CurvePoint } from './types'

import CurveEditor from './CurveEditor.vue'

function mountEditor(points: CurvePoint[], extraProps = {}) {
  return mount(CurveEditor, {
    props: { modelValue: points, ...extraProps }
  })
}

function getCurvePath(wrapper: ReturnType<typeof mount>) {
  return wrapper.find('[data-testid="curve-path"]')
}

describe('CurveEditor', () => {
  it('renders SVG with curve path', () => {
    const wrapper = mountEditor([
      [0, 0],
      [1, 1]
    ])
    expect(wrapper.find('svg').exists()).toBe(true)
    const curvePath = getCurvePath(wrapper)
    expect(curvePath.exists()).toBe(true)
    expect(curvePath.attributes('d')).toBeTruthy()
  })

  it('renders a circle for each control point', () => {
    const wrapper = mountEditor([
      [0, 0],
      [0.5, 0.7],
      [1, 1]
    ])
    expect(wrapper.findAll('circle')).toHaveLength(3)
  })

  it('renders histogram path when provided', () => {
    const histogram = new Uint32Array(256)
    for (let i = 0; i < 256; i++) histogram[i] = i + 1
    const wrapper = mountEditor(
      [
        [0, 0],
        [1, 1]
      ],
      { histogram }
    )
    const histogramPath = wrapper.find('[data-testid="histogram-path"]')
    expect(histogramPath.exists()).toBe(true)
    expect(histogramPath.attributes('d')).toContain('M0,1')
  })

  it('does not render histogram path when not provided', () => {
    const wrapper = mountEditor([
      [0, 0],
      [1, 1]
    ])
    expect(wrapper.find('[data-testid="histogram-path"]').exists()).toBe(false)
  })

  it('returns empty path with fewer than 2 points', () => {
    const wrapper = mountEditor([[0.5, 0.5]])
    expect(getCurvePath(wrapper).attributes('d')).toBe('')
  })

  it('generates path starting with M and containing L segments', () => {
    const wrapper = mountEditor([
      [0, 0],
      [0.5, 0.8],
      [1, 1]
    ])
    const d = getCurvePath(wrapper).attributes('d')!
    expect(d).toMatch(/^M/)
    expect(d).toContain('L')
  })

  it('curve path only spans the x-range of control points', () => {
    const wrapper = mountEditor([
      [0.2, 0.3],
      [0.8, 0.9]
    ])
    const d = getCurvePath(wrapper).attributes('d')!
    const xValues = d
      .split(/[ML]/)
      .filter(Boolean)
      .map((s) => parseFloat(s.split(',')[0]))
    expect(Math.min(...xValues)).toBeCloseTo(0.2, 2)
    expect(Math.max(...xValues)).toBeCloseTo(0.8, 2)
  })

  it('deletes a point on right-click but keeps minimum 2', async () => {
    const points: CurvePoint[] = [
      [0, 0],
      [0.5, 0.5],
      [1, 1]
    ]
    const wrapper = mountEditor(points)
    expect(wrapper.findAll('circle')).toHaveLength(3)

    await wrapper.findAll('circle')[1].trigger('pointerdown', {
      button: 2,
      pointerId: 1
    })
    expect(wrapper.findAll('circle')).toHaveLength(2)

    await wrapper.findAll('circle')[0].trigger('pointerdown', {
      button: 2,
      pointerId: 1
    })
    expect(wrapper.findAll('circle')).toHaveLength(2)
  })
})
