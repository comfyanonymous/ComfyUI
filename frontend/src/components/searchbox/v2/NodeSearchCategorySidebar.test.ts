import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import NodeSearchCategorySidebar from '@/components/searchbox/v2/NodeSearchCategorySidebar.vue'
import {
  createMockNodeDef,
  setupTestPinia,
  testI18n
} from '@/components/searchbox/v2/__test__/testUtils'
import { useNodeDefStore } from '@/stores/nodeDefStore'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn(() => undefined),
    set: vi.fn()
  }))
}))

describe('NodeSearchCategorySidebar', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    setupTestPinia()
  })

  async function createWrapper(props = {}) {
    const wrapper = mount(NodeSearchCategorySidebar, {
      props: { selectedCategory: 'most-relevant', ...props },
      global: { plugins: [testI18n] }
    })
    await nextTick()
    return wrapper
  }

  async function clickCategory(
    wrapper: ReturnType<typeof mount>,
    text: string,
    exact = false
  ) {
    const btn = wrapper
      .findAll('button')
      .find((b) => (exact ? b.text().trim() === text : b.text().includes(text)))
    expect(btn, `Expected to find a button with text "${text}"`).toBeDefined()
    await btn!.trigger('click')
    await nextTick()
  }

  describe('preset categories', () => {
    it('should render all preset categories', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'EssentialNode',
          essentials_category: 'basic'
        })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      expect(wrapper.text()).toContain('Most relevant')
      expect(wrapper.text()).toContain('Favorites')
      expect(wrapper.text()).toContain('Essentials')
      expect(wrapper.text()).toContain('Custom')
    })

    it('should mark the selected preset category as selected', async () => {
      const wrapper = await createWrapper({ selectedCategory: 'most-relevant' })

      const mostRelevantBtn = wrapper.find(
        '[data-testid="category-most-relevant"]'
      )

      expect(mostRelevantBtn.attributes('aria-current')).toBe('true')
    })

    it('should emit update:selectedCategory when preset is clicked', async () => {
      const wrapper = await createWrapper({ selectedCategory: 'most-relevant' })

      await clickCategory(wrapper, 'Favorites')

      expect(wrapper.emitted('update:selectedCategory')).toBeTruthy()
      expect(wrapper.emitted('update:selectedCategory')![0]).toEqual([
        'favorites'
      ])
    })
  })

  describe('category tree', () => {
    it('should render top-level categories from node definitions', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' }),
        createMockNodeDef({ name: 'Node2', category: 'loaders' }),
        createMockNodeDef({ name: 'Node3', category: 'conditioning' })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      expect(wrapper.text()).toContain('sampling')
      expect(wrapper.text()).toContain('loaders')
      expect(wrapper.text()).toContain('conditioning')
    })

    it('should emit update:selectedCategory when category is clicked', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      await clickCategory(wrapper, 'sampling')

      expect(wrapper.emitted('update:selectedCategory')![0]).toEqual([
        'sampling'
      ])
    })
  })

  describe('expand/collapse functionality', () => {
    it('should expand category when clicked and show subcategories', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' }),
        createMockNodeDef({ name: 'Node2', category: 'sampling/advanced' }),
        createMockNodeDef({ name: 'Node3', category: 'sampling/basic' })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      expect(wrapper.text()).not.toContain('advanced')

      await clickCategory(wrapper, 'sampling')

      expect(wrapper.text()).toContain('advanced')
      expect(wrapper.text()).toContain('basic')
    })

    it('should collapse sibling category when another is expanded', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' }),
        createMockNodeDef({ name: 'Node2', category: 'sampling/advanced' }),
        createMockNodeDef({ name: 'Node3', category: 'image' }),
        createMockNodeDef({ name: 'Node4', category: 'image/upscale' })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      // Expand sampling
      await clickCategory(wrapper, 'sampling', true)
      expect(wrapper.text()).toContain('advanced')

      // Expand image — sampling should collapse
      await clickCategory(wrapper, 'image', true)

      expect(wrapper.text()).toContain('upscale')
      expect(wrapper.text()).not.toContain('advanced')
    })

    it('should emit update:selectedCategory when subcategory is clicked', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' }),
        createMockNodeDef({ name: 'Node2', category: 'sampling/advanced' })
      ])
      await nextTick()

      const wrapper = await createWrapper()

      // Expand sampling category
      await clickCategory(wrapper, 'sampling', true)

      // Click on advanced subcategory
      await clickCategory(wrapper, 'advanced')

      const emitted = wrapper.emitted('update:selectedCategory')!
      expect(emitted[emitted.length - 1]).toEqual(['sampling/advanced'])
    })
  })

  describe('category selection highlighting', () => {
    it('should mark selected top-level category as selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' })
      ])
      await nextTick()

      const wrapper = await createWrapper({ selectedCategory: 'sampling' })

      expect(
        wrapper
          .find('[data-testid="category-sampling"]')
          .attributes('aria-current')
      ).toBe('true')
    })

    it('should emit selected subcategory when expanded', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', category: 'sampling' }),
        createMockNodeDef({ name: 'Node2', category: 'sampling/advanced' })
      ])
      await nextTick()

      const wrapper = await createWrapper({ selectedCategory: 'most-relevant' })

      // Expand and click subcategory
      await clickCategory(wrapper, 'sampling', true)
      await clickCategory(wrapper, 'advanced')

      const emitted = wrapper.emitted('update:selectedCategory')!
      expect(emitted[emitted.length - 1]).toEqual(['sampling/advanced'])
    })
  })

  it('should support deeply nested categories (3+ levels)', async () => {
    useNodeDefStore().updateNodeDefs([
      createMockNodeDef({ name: 'Node1', category: 'api' }),
      createMockNodeDef({ name: 'Node2', category: 'api/image' }),
      createMockNodeDef({ name: 'Node3', category: 'api/image/BFL' })
    ])
    await nextTick()

    const wrapper = await createWrapper()

    // Only top-level visible initially
    expect(wrapper.text()).toContain('api')
    expect(wrapper.text()).not.toContain('image')
    expect(wrapper.text()).not.toContain('BFL')

    // Expand api
    await clickCategory(wrapper, 'api', true)

    expect(wrapper.text()).toContain('image')
    expect(wrapper.text()).not.toContain('BFL')

    // Expand image
    await clickCategory(wrapper, 'image', true)

    expect(wrapper.text()).toContain('BFL')

    // Click BFL and verify emission
    await clickCategory(wrapper, 'BFL', true)

    const emitted = wrapper.emitted('update:selectedCategory')!
    expect(emitted[emitted.length - 1]).toEqual(['api/image/BFL'])
  })

  it('should emit category without root/ prefix', async () => {
    useNodeDefStore().updateNodeDefs([
      createMockNodeDef({ name: 'Node1', category: 'sampling' })
    ])
    await nextTick()

    const wrapper = await createWrapper()

    await clickCategory(wrapper, 'sampling')

    expect(wrapper.emitted('update:selectedCategory')![0][0]).toBe('sampling')
  })
})
