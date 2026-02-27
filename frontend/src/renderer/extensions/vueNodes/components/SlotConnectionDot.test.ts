import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { INodeSlot } from '@/lib/litegraph/src/litegraph'
import { RenderShape } from '@/lib/litegraph/src/types/globalEnums'

import SlotConnectionDot from './SlotConnectionDot.vue'

const defaultSlot: INodeSlot = {
  name: 'output',
  type: 'IMAGE',
  boundingRect: [0, 0, 0, 0]
}

function mountDot(slotData?: INodeSlot) {
  return mount(SlotConnectionDot, {
    props: { slotData }
  })
}

describe('SlotConnectionDot', () => {
  it('renders circle shape by default', () => {
    const wrapper = mountDot(defaultSlot)

    const dot = wrapper.find('.slot-dot')
    expect(dot.classes()).toContain('rounded-full')
    expect(dot.element.tagName).toBe('DIV')
  })

  it('renders rounded square for GRID shape', () => {
    const wrapper = mountDot({
      ...defaultSlot,
      shape: RenderShape.GRID
    })

    const dot = wrapper.find('.slot-dot')
    expect(dot.classes()).toContain('rounded-[1px]')
    expect(dot.classes()).not.toContain('rounded-full')
    expect(dot.element.tagName).toBe('DIV')
  })
})
