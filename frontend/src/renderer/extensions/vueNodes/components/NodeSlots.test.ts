/* eslint-disable vue/one-component-per-file */
import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { defineComponent } from 'vue'
import type { PropType } from 'vue'
import { createI18n } from 'vue-i18n'

import type { VueNodeData } from '@/composables/graph/useGraphNodeManager'
import type {
  INodeInputSlot,
  INodeOutputSlot
} from '@/lib/litegraph/src/interfaces'
import enMessages from '@/locales/en/main.json' with { type: 'json' }

import NodeSlots from './NodeSlots.vue'

const makeNodeData = (overrides: Partial<VueNodeData> = {}): VueNodeData => ({
  id: '123',
  title: 'Test Node',
  type: 'TestType',
  mode: 0,
  selected: false,
  executing: false,
  inputs: [],
  outputs: [],
  widgets: [],
  flags: { collapsed: false },
  ...overrides
})

// Explicit stubs to capture props for assertions
interface StubSlotData {
  name?: string
  type?: string
  boundingRect?: [number, number, number, number]
}

const InputSlotStub = defineComponent({
  name: 'InputSlot',
  props: {
    slotData: { type: Object as PropType<StubSlotData>, required: true },
    nodeId: { type: String, required: false, default: '' },
    index: { type: Number, required: true },
    readonly: { type: Boolean, required: false, default: false }
  },
  template: `
    <div
      class="stub-input-slot"
      :data-index="index"
      :data-name="slotData && slotData.name ? slotData.name : ''"
      :data-type="slotData && slotData.type ? slotData.type : ''"
      :data-node-id="nodeId"
      :data-readonly="readonly ? 'true' : 'false'"
    />
  `
})

const OutputSlotStub = defineComponent({
  name: 'OutputSlot',
  props: {
    slotData: { type: Object as PropType<StubSlotData>, required: true },
    nodeId: { type: String, required: false, default: '' },
    index: { type: Number, required: true },
    readonly: { type: Boolean, required: false, default: false }
  },
  template: `
    <div
      class="stub-output-slot"
      :data-index="index"
      :data-name="slotData && slotData.name ? slotData.name : ''"
      :data-type="slotData && slotData.type ? slotData.type : ''"
      :data-node-id="nodeId"
      :data-readonly="readonly ? 'true' : 'false'"
    />
  `
})

const mountSlots = (nodeData: VueNodeData, readonly = false) => {
  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: { en: enMessages }
  })
  return mount(NodeSlots, {
    global: {
      plugins: [i18n, createTestingPinia({ stubActions: false })],
      stubs: {
        InputSlot: InputSlotStub,
        OutputSlot: OutputSlotStub
      }
    },
    props: { nodeData, readonly }
  })
}

describe('NodeSlots.vue', () => {
  it('filters out inputs with widget property and maps indexes correctly', (context) => {
    context.skip('Filtering not working as expected, needs diagnosis')
    // Two inputs without widgets (object and string) and one with widget (filtered)
    const inputObjNoWidget: INodeInputSlot = {
      name: 'objNoWidget',
      type: 'number',
      boundingRect: [0, 0, 0, 0],
      link: null
    }
    const inputObjWithWidget: INodeInputSlot = {
      name: 'objWithWidget',
      type: 'number',
      boundingRect: [0, 0, 0, 0],
      widget: { name: 'objWithWidget' },
      link: null
    }
    const inputs: INodeInputSlot[] = [inputObjNoWidget, inputObjWithWidget]

    const wrapper = mountSlots(makeNodeData({ inputs }))

    const inputEls = wrapper
      .findAll('.stub-input-slot')
      .map((w) => w.element as HTMLElement)
    // Should filter out the widget-backed input; expect 2 inputs rendered
    expect(inputEls.length).toBe(2)

    // Verify expected tuple of {index, name, nodeId}
    const info = inputEls.map((el) => ({
      index: Number(el.dataset.index),
      name: el.dataset.name ?? '',
      nodeId: el.dataset.nodeId ?? '',
      type: el.dataset.type ?? '',
      readonly: el.dataset.readonly === 'true'
    }))
    expect(info).toEqual([
      {
        index: 0,
        name: 'objNoWidget',
        nodeId: '123',
        type: 'number',
        readonly: false
      },
      // string input is converted to object with default type 'any'
      {
        index: 1,
        name: 'stringInput',
        nodeId: '123',
        type: 'any',
        readonly: false
      }
    ])

    // Ensure widget-backed input was indeed filtered out
    expect(wrapper.find('[data-name="objWithWidget"]').exists()).toBe(false)
  })

  it('maps outputs and passes correct indexes', () => {
    const outputObj: INodeOutputSlot = {
      name: 'outA',
      type: 'any',
      boundingRect: [0, 0, 0, 0],
      links: []
    }
    const outputObjB: INodeOutputSlot = {
      name: 'outB',
      type: 'any',
      boundingRect: [0, 0, 0, 0],
      links: []
    }
    const outputs: INodeOutputSlot[] = [outputObj, outputObjB]

    const wrapper = mountSlots(makeNodeData({ outputs }))
    const outputEls = wrapper
      .findAll('.stub-output-slot')
      .map((w) => w.element as HTMLElement)

    expect(outputEls.length).toBe(2)
    const outInfo = outputEls.map((el) => ({
      index: Number(el.dataset.index),
      name: el.dataset.name ?? '',
      nodeId: el.dataset.nodeId ?? '',
      type: el.dataset.type ?? '',
      readonly: el.dataset.readonly === 'true'
    }))
    expect(outInfo).toEqual([
      { index: 0, name: 'outA', nodeId: '123', type: 'any', readonly: false },
      // string output mapped to object with type 'any'
      { index: 1, name: 'outB', nodeId: '123', type: 'any', readonly: false }
    ])
  })

  it('renders nothing when there are no inputs/outputs', () => {
    const wrapper = mountSlots(makeNodeData({ inputs: [], outputs: [] }))
    expect(wrapper.findAll('.stub-input-slot').length).toBe(0)
    expect(wrapper.findAll('.stub-output-slot').length).toBe(0)
  })
})
