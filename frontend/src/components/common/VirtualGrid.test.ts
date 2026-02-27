import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Ref } from 'vue'
import { nextTick, ref } from 'vue'

import VirtualGrid from './VirtualGrid.vue'

type TestItem = { key: string; name: string }

let mockedWidth: Ref<number>
let mockedHeight: Ref<number>
let mockedScrollY: Ref<number>

vi.mock('@vueuse/core', async () => {
  const actual = await vi.importActual<Record<string, unknown>>('@vueuse/core')
  return {
    ...actual,
    useElementSize: () => ({ width: mockedWidth, height: mockedHeight }),
    useScroll: () => ({ y: mockedScrollY })
  }
})

beforeEach(() => {
  mockedWidth = ref(400)
  mockedHeight = ref(200)
  mockedScrollY = ref(0)
})

function createItems(count: number): TestItem[] {
  return Array.from({ length: count }, (_, i) => ({
    key: `item-${i}`,
    name: `Item ${i}`
  }))
}

describe('VirtualGrid', () => {
  const defaultGridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '1rem'
  }

  it('renders items within the visible range', async () => {
    const items = createItems(100)
    mockedWidth.value = 400
    mockedHeight.value = 200
    mockedScrollY.value = 0

    const wrapper = mount(VirtualGrid<TestItem>, {
      props: {
        items,
        gridStyle: defaultGridStyle,
        defaultItemHeight: 100,
        defaultItemWidth: 100,
        maxColumns: 4,
        bufferRows: 1
      },
      slots: {
        item: `<template #item="{ item }">
          <div class="test-item">{{ item.name }}</div>
        </template>`
      },
      attachTo: document.body
    })

    await nextTick()

    const renderedItems = wrapper.findAll('.test-item')
    expect(renderedItems.length).toBeGreaterThan(0)
    expect(renderedItems.length).toBeLessThan(items.length)

    wrapper.unmount()
  })

  it('provides correct index in slot props', async () => {
    const items = createItems(20)
    const receivedIndices: number[] = []
    mockedWidth.value = 400
    mockedHeight.value = 200
    mockedScrollY.value = 0

    const wrapper = mount(VirtualGrid<TestItem>, {
      props: {
        items,
        gridStyle: defaultGridStyle,
        defaultItemHeight: 50,
        defaultItemWidth: 100,
        maxColumns: 1,
        bufferRows: 0
      },
      slots: {
        item: ({ index }: { index: number }) => {
          receivedIndices.push(index)
          return null
        }
      },
      attachTo: document.body
    })

    await nextTick()

    expect(receivedIndices.length).toBeGreaterThan(0)
    expect(receivedIndices[0]).toBe(0)
    for (let i = 1; i < receivedIndices.length; i++) {
      expect(receivedIndices[i]).toBe(receivedIndices[i - 1] + 1)
    }

    wrapper.unmount()
  })

  it('respects maxColumns prop', async () => {
    const items = createItems(10)
    mockedWidth.value = 400
    mockedHeight.value = 200
    mockedScrollY.value = 0

    const wrapper = mount(VirtualGrid<TestItem>, {
      props: {
        items,
        gridStyle: defaultGridStyle,
        maxColumns: 2
      },
      attachTo: document.body
    })

    await nextTick()

    const gridElement = wrapper.find('[style*="display: grid"]')
    expect(gridElement.exists()).toBe(true)

    const gridEl = gridElement.element as HTMLElement
    expect(gridEl.style.gridTemplateColumns).toBe('repeat(2, minmax(0, 1fr))')

    wrapper.unmount()
  })

  it('renders empty when no items provided', async () => {
    const wrapper = mount(VirtualGrid<TestItem>, {
      props: {
        items: [],
        gridStyle: defaultGridStyle
      },
      slots: {
        item: `<template #item="{ item }">
          <div class="test-item">{{ item.name }}</div>
        </template>`
      }
    })

    await nextTick()

    const renderedItems = wrapper.findAll('.test-item')
    expect(renderedItems.length).toBe(0)

    wrapper.unmount()
  })

  it('forces cols to maxColumns when maxColumns is finite', async () => {
    mockedWidth.value = 100
    mockedHeight.value = 200
    mockedScrollY.value = 0

    const items = createItems(20)
    const wrapper = mount(VirtualGrid<TestItem>, {
      props: {
        items,
        gridStyle: defaultGridStyle,
        defaultItemHeight: 50,
        defaultItemWidth: 200,
        maxColumns: 4,
        bufferRows: 0
      },
      slots: {
        item: `<template #item="{ item }">
          <div class="test-item">{{ item.name }}</div>
        </template>`
      },
      attachTo: document.body
    })

    await nextTick()

    const renderedItems = wrapper.findAll('.test-item')
    expect(renderedItems.length).toBeGreaterThan(0)
    expect(renderedItems.length % 4).toBe(0)

    wrapper.unmount()
  })
})
