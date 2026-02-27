import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import NodeSearchCategorySidebar from '@/components/searchbox/v2/NodeSearchCategorySidebar.vue'
import NodeSearchContent from '@/components/searchbox/v2/NodeSearchContent.vue'
import {
  createMockNodeDef,
  setupTestPinia,
  testI18n
} from '@/components/searchbox/v2/__test__/testUtils'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore, useNodeFrequencyStore } from '@/stores/nodeDefStore'
import { NodeSourceType } from '@/types/nodeSource'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn((key: string) => {
      if (key === 'Comfy.NodeLibrary.Bookmarks.V2') return []
      if (key === 'Comfy.NodeLibrary.BookmarksCustomization') return {}
      return undefined
    }),
    set: vi.fn()
  }))
}))

describe('NodeSearchContent', () => {
  beforeEach(() => {
    setupTestPinia()
    vi.restoreAllMocks()
  })

  async function createWrapper(props = {}) {
    const wrapper = mount(NodeSearchContent, {
      props: { filters: [], ...props },
      global: {
        plugins: [testI18n],
        stubs: {
          NodeSearchListItem: {
            template: '<div class="node-item">{{ nodeDef.display_name }}</div>',
            props: [
              'nodeDef',
              'currentQuery',
              'showDescription',
              'showSourceBadge',
              'hideBookmarkIcon'
            ]
          }
        }
      }
    })
    await nextTick()
    return wrapper
  }

  async function setupFavorites(
    nodes: Parameters<typeof createMockNodeDef>[0][]
  ) {
    useNodeDefStore().updateNodeDefs(nodes.map(createMockNodeDef))
    vi.spyOn(useNodeBookmarkStore(), 'isBookmarked').mockReturnValue(true)
    const wrapper = await createWrapper()
    await wrapper.find('[data-testid="category-favorites"]').trigger('click')
    await nextTick()
    return wrapper
  }

  function getResultItems(wrapper: VueWrapper) {
    return wrapper.findAll('[data-testid="result-item"]')
  }

  function getNodeItems(wrapper: VueWrapper) {
    return wrapper.findAll('.node-item')
  }

  describe('category selection', () => {
    it('should show top nodes when Most relevant is selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'FrequentNode',
          display_name: 'Frequent Node'
        }),
        createMockNodeDef({ name: 'RareNode', display_name: 'Rare Node' })
      ])

      vi.spyOn(useNodeFrequencyStore(), 'topNodeDefs', 'get').mockReturnValue([
        useNodeDefStore().nodeDefsByName['FrequentNode']
      ])

      const wrapper = await createWrapper()

      const items = getNodeItems(wrapper)
      expect(items).toHaveLength(1)
      expect(items[0].text()).toContain('Frequent Node')
    })

    it('should show only bookmarked nodes when Favorites is selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'BookmarkedNode',
          display_name: 'Bookmarked Node'
        }),
        createMockNodeDef({
          name: 'RegularNode',
          display_name: 'Regular Node'
        })
      ])
      vi.spyOn(useNodeBookmarkStore(), 'isBookmarked').mockImplementation(
        (node: ComfyNodeDefImpl) => node.name === 'BookmarkedNode'
      )

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-favorites"]').trigger('click')
      await nextTick()

      const items = getNodeItems(wrapper)
      expect(items).toHaveLength(1)
      expect(items[0].text()).toContain('Bookmarked')
    })

    it('should show empty state when no bookmarks exist', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'Node1', display_name: 'Node One' })
      ])
      vi.spyOn(useNodeBookmarkStore(), 'isBookmarked').mockReturnValue(false)

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-favorites"]').trigger('click')
      await nextTick()

      expect(wrapper.text()).toContain('No results')
    })

    it('should show only non-Core nodes when Custom is selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'CoreNode',
          display_name: 'Core Node',
          python_module: 'nodes'
        }),
        createMockNodeDef({
          name: 'CustomNode',
          display_name: 'Custom Node',
          python_module: 'custom_nodes.my_extension'
        })
      ])
      await nextTick()

      expect(useNodeDefStore().nodeDefsByName['CoreNode'].nodeSource.type).toBe(
        NodeSourceType.Core
      )
      expect(
        useNodeDefStore().nodeDefsByName['CustomNode'].nodeSource.type
      ).toBe(NodeSourceType.CustomNodes)

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-custom"]').trigger('click')
      await nextTick()

      const items = getNodeItems(wrapper)
      expect(items).toHaveLength(1)
      expect(items[0].text()).toContain('Custom Node')
    })

    it('should hide Essentials category when no essential nodes exist', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'RegularNode',
          display_name: 'Regular Node'
        })
      ])

      const wrapper = await createWrapper()
      expect(wrapper.find('[data-testid="category-essentials"]').exists()).toBe(
        false
      )
    })

    it('should show only essential nodes when Essentials is selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'EssentialNode',
          display_name: 'Essential Node',
          essentials_category: 'basic'
        }),
        createMockNodeDef({
          name: 'RegularNode',
          display_name: 'Regular Node'
        })
      ])
      await nextTick()

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-essentials"]').trigger('click')
      await nextTick()

      const items = getNodeItems(wrapper)
      expect(items).toHaveLength(1)
      expect(items[0].text()).toContain('Essential Node')
    })

    it('should include subcategory nodes when parent category is selected', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'KSampler',
          display_name: 'KSampler',
          category: 'sampling'
        }),
        createMockNodeDef({
          name: 'LoadCheckpoint',
          display_name: 'Load Checkpoint',
          category: 'loaders'
        }),
        createMockNodeDef({
          name: 'KSamplerAdvanced',
          display_name: 'KSampler Advanced',
          category: 'sampling/advanced'
        })
      ])

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-sampling"]').trigger('click')
      await nextTick()

      const texts = getNodeItems(wrapper).map((i) => i.text())
      expect(texts).toHaveLength(2)
      expect(texts).toContain('KSampler')
      expect(texts).toContain('KSampler Advanced')
    })
  })

  describe('search and category interaction', () => {
    it('should override category to most-relevant when search query is active', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'KSampler',
          display_name: 'KSampler',
          category: 'sampling'
        }),
        createMockNodeDef({
          name: 'LoadCheckpoint',
          display_name: 'Load Checkpoint',
          category: 'loaders'
        })
      ])

      const wrapper = await createWrapper()
      await wrapper.find('[data-testid="category-sampling"]').trigger('click')
      await nextTick()

      expect(getNodeItems(wrapper)).toHaveLength(1)

      const input = wrapper.find('input[type="text"]')
      await input.setValue('Load')
      await nextTick()

      const texts = getNodeItems(wrapper).map((i) => i.text())
      expect(texts.some((t) => t.includes('Load Checkpoint'))).toBe(true)
    })

    it('should clear search query when category changes', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({ name: 'TestNode', display_name: 'Test Node' })
      ])

      const wrapper = await createWrapper()

      const input = wrapper.find('input[type="text"]')
      await input.setValue('test query')
      await nextTick()
      expect((input.element as HTMLInputElement).value).toBe('test query')

      await wrapper.find('[data-testid="category-favorites"]').trigger('click')
      await nextTick()
      expect((input.element as HTMLInputElement).value).toBe('')
    })

    it('should reset selected index when search query changes', async () => {
      const wrapper = await setupFavorites([
        { name: 'Node1', display_name: 'Node One' },
        { name: 'Node2', display_name: 'Node Two' }
      ])

      const input = wrapper.find('input[type="text"]')
      await input.trigger('keydown', { key: 'ArrowDown' })
      await nextTick()
      expect(getResultItems(wrapper)[1].attributes('aria-selected')).toBe(
        'true'
      )

      await input.setValue('Node')
      await nextTick()
      expect(getResultItems(wrapper)[0].attributes('aria-selected')).toBe(
        'true'
      )
    })

    it('should reset selected index when category changes', async () => {
      const wrapper = await setupFavorites([
        { name: 'Node1', display_name: 'Node One' },
        { name: 'Node2', display_name: 'Node Two' }
      ])

      const input = wrapper.find('input[type="text"]')
      await input.trigger('keydown', { key: 'ArrowDown' })
      await nextTick()

      await wrapper
        .find('[data-testid="category-most-relevant"]')
        .trigger('click')
      await nextTick()
      await wrapper.find('[data-testid="category-favorites"]').trigger('click')
      await nextTick()

      expect(getResultItems(wrapper)[0].attributes('aria-selected')).toBe(
        'true'
      )
    })
  })

  describe('keyboard and mouse interaction', () => {
    it('should navigate results with ArrowDown/ArrowUp and clamp to bounds', async () => {
      const wrapper = await setupFavorites([
        { name: 'Node1', display_name: 'Node One' },
        { name: 'Node2', display_name: 'Node Two' },
        { name: 'Node3', display_name: 'Node Three' }
      ])

      const input = wrapper.find('input[type="text"]')
      const selectedIndex = () =>
        getResultItems(wrapper).findIndex(
          (r) => r.attributes('aria-selected') === 'true'
        )

      expect(selectedIndex()).toBe(0)

      await input.trigger('keydown', { key: 'ArrowDown' })
      await nextTick()
      expect(selectedIndex()).toBe(1)

      await input.trigger('keydown', { key: 'ArrowDown' })
      await nextTick()
      expect(selectedIndex()).toBe(2)

      await input.trigger('keydown', { key: 'ArrowUp' })
      await nextTick()
      expect(selectedIndex()).toBe(1)

      // Navigate to first, then try going above — should clamp
      await input.trigger('keydown', { key: 'ArrowUp' })
      await nextTick()
      expect(selectedIndex()).toBe(0)

      await input.trigger('keydown', { key: 'ArrowUp' })
      await nextTick()
      expect(selectedIndex()).toBe(0)
    })

    it('should select current result with Enter key', async () => {
      const wrapper = await setupFavorites([
        { name: 'TestNode', display_name: 'Test Node' }
      ])

      await wrapper
        .find('input[type="text"]')
        .trigger('keydown', { key: 'Enter' })
      await nextTick()

      expect(wrapper.emitted('addNode')).toBeTruthy()
      expect(wrapper.emitted('addNode')![0][0]).toMatchObject({
        name: 'TestNode'
      })
    })

    it('should select item on hover', async () => {
      const wrapper = await setupFavorites([
        { name: 'Node1', display_name: 'Node One' },
        { name: 'Node2', display_name: 'Node Two' }
      ])

      const results = getResultItems(wrapper)
      await results[1].trigger('mouseenter')
      await nextTick()

      expect(results[1].attributes('aria-selected')).toBe('true')
    })

    it('should add node on click', async () => {
      const wrapper = await setupFavorites([
        { name: 'TestNode', display_name: 'Test Node' }
      ])

      await getResultItems(wrapper)[0].trigger('click')
      await nextTick()

      expect(wrapper.emitted('addNode')![0][0]).toMatchObject({
        name: 'TestNode'
      })
    })
  })

  describe('hoverNode emission', () => {
    it('should emit hoverNode with the currently selected node', async () => {
      const wrapper = await setupFavorites([
        { name: 'HoverNode', display_name: 'Hover Node' }
      ])

      const emitted = wrapper.emitted('hoverNode')!
      expect(emitted[emitted.length - 1][0]).toMatchObject({
        name: 'HoverNode'
      })
    })

    it('should emit null hoverNode when no results', async () => {
      const wrapper = await createWrapper()

      vi.spyOn(useNodeBookmarkStore(), 'isBookmarked').mockReturnValue(false)
      await wrapper.find('[data-testid="category-favorites"]').trigger('click')
      await nextTick()

      const emitted = wrapper.emitted('hoverNode')!
      expect(emitted[emitted.length - 1][0]).toBeNull()
    })
  })

  describe('filter integration', () => {
    it('should display active filters in the input area', async () => {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'ImageNode',
          display_name: 'Image Node',
          input: { required: { image: ['IMAGE', {}] } }
        })
      ])

      const wrapper = await createWrapper({
        filters: [
          {
            filterDef: useNodeDefStore().nodeSearchService.inputTypeFilter,
            value: 'IMAGE'
          }
        ]
      })

      expect(
        wrapper.findAll('[data-testid="filter-chip"]').length
      ).toBeGreaterThan(0)
    })
  })

  describe('chip removal', () => {
    function createFilters(count: number) {
      const types = ['IMAGE', 'LATENT', 'MODEL']
      useNodeDefStore().updateNodeDefs(
        types.slice(0, count).map((type) =>
          createMockNodeDef({
            name: `${type}Node`,
            display_name: `${type} Node`,
            input: {
              required: { [type.toLowerCase()]: [type, {}] }
            }
          })
        )
      )
      return types.slice(0, count).map((type) => ({
        filterDef: useNodeDefStore().nodeSearchService.inputTypeFilter,
        value: type
      }))
    }

    it('should emit removeFilter on backspace', async () => {
      const filters = createFilters(1)
      const wrapper = await createWrapper({ filters })

      const input = wrapper.find('input[type="text"]')
      await input.trigger('keydown', { key: 'Backspace' })
      await nextTick()
      await input.trigger('keydown', { key: 'Backspace' })
      await nextTick()

      expect(wrapper.emitted('removeFilter')).toHaveLength(1)
      expect(wrapper.emitted('removeFilter')![0][0]).toMatchObject({
        value: 'IMAGE'
      })
    })

    it('should not interact with chips when no filters exist', async () => {
      const wrapper = await createWrapper({ filters: [] })

      const input = wrapper.find('input[type="text"]')
      await input.trigger('keydown', { key: 'Backspace' })
      await nextTick()

      expect(wrapper.emitted('removeFilter')).toBeUndefined()
    })

    it('should remove chip when clicking its delete button', async () => {
      const filters = createFilters(1)
      const wrapper = await createWrapper({ filters })

      const deleteBtn = wrapper.find('[data-testid="chip-delete"]')
      await deleteBtn.trigger('click')
      await nextTick()

      expect(wrapper.emitted('removeFilter')).toHaveLength(1)
      expect(wrapper.emitted('removeFilter')![0][0]).toMatchObject({
        value: 'IMAGE'
      })
    })
  })

  describe('filter selection mode', () => {
    function setupNodesWithTypes() {
      useNodeDefStore().updateNodeDefs([
        createMockNodeDef({
          name: 'ImageNode',
          display_name: 'Image Node',
          input: { required: { image: ['IMAGE', {}] } },
          output: ['IMAGE']
        }),
        createMockNodeDef({
          name: 'LatentNode',
          display_name: 'Latent Node',
          input: { required: { latent: ['LATENT', {}] } },
          output: ['LATENT']
        }),
        createMockNodeDef({
          name: 'ModelNode',
          display_name: 'Model Node',
          input: { required: { model: ['MODEL', {}] } },
          output: ['MODEL']
        })
      ])
    }

    function findFilterBarButton(wrapper: VueWrapper, label: string) {
      return wrapper
        .findAll('button[aria-pressed]')
        .find((b) => b.text() === label)
    }

    async function enterFilterMode(wrapper: VueWrapper) {
      await findFilterBarButton(wrapper, 'Input')!.trigger('click')
      await nextTick()
    }

    function getFilterOptions(wrapper: VueWrapper) {
      return wrapper.findAll('[data-testid="filter-option"]')
    }

    function getFilterOptionTexts(wrapper: VueWrapper) {
      return getFilterOptions(wrapper).map(
        (o) =>
          o
            .findAll('span')[0]
            ?.text()
            .replace(/^[•·]\s*/, '')
            .trim() ?? ''
      )
    }

    function hasSidebar(wrapper: VueWrapper) {
      return wrapper.findComponent(NodeSearchCategorySidebar).exists()
    }

    it('should enter filter mode when a filter chip is selected', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()

      expect(hasSidebar(wrapper)).toBe(true)

      await enterFilterMode(wrapper)

      expect(hasSidebar(wrapper)).toBe(false)
      expect(getFilterOptions(wrapper).length).toBeGreaterThan(0)
    })

    it('should show available filter options sorted alphabetically', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      const texts = getFilterOptionTexts(wrapper)
      expect(texts).toContain('IMAGE')
      expect(texts).toContain('LATENT')
      expect(texts).toContain('MODEL')
      expect(texts).toEqual([...texts].sort())
    })

    it('should filter options when typing in filter mode', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      await wrapper.find('input[type="text"]').setValue('IMAGE')
      await nextTick()

      const texts = getFilterOptionTexts(wrapper)
      expect(texts).toContain('IMAGE')
      expect(texts).not.toContain('MODEL')
    })

    it('should show no results when filter query has no matches', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      await wrapper.find('input[type="text"]').setValue('NONEXISTENT_TYPE')
      await nextTick()

      expect(wrapper.text()).toContain('No results')
    })

    it('should emit addFilter when a filter option is clicked', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      const imageOption = getFilterOptions(wrapper).find((o) =>
        o.text().includes('IMAGE')
      )
      await imageOption!.trigger('click')
      await nextTick()

      expect(wrapper.emitted('addFilter')![0][0]).toMatchObject({
        filterDef: expect.objectContaining({ id: 'input' }),
        value: 'IMAGE'
      })
    })

    it('should exit filter mode after applying a filter', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      await getFilterOptions(wrapper)[0].trigger('click')
      await nextTick()
      await nextTick()

      expect(hasSidebar(wrapper)).toBe(true)
    })

    it('should emit addFilter when Enter is pressed on selected option', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      await wrapper
        .find('input[type="text"]')
        .trigger('keydown', { key: 'Enter' })
      await nextTick()

      expect(wrapper.emitted('addFilter')![0][0]).toMatchObject({
        filterDef: expect.objectContaining({ id: 'input' }),
        value: 'IMAGE'
      })
    })

    it('should navigate filter options with ArrowDown/ArrowUp', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      const input = wrapper.find('input[type="text"]')

      expect(getFilterOptions(wrapper)[0].attributes('aria-selected')).toBe(
        'true'
      )

      await input.trigger('keydown', { key: 'ArrowDown' })
      await nextTick()
      expect(getFilterOptions(wrapper)[1].attributes('aria-selected')).toBe(
        'true'
      )

      await input.trigger('keydown', { key: 'ArrowUp' })
      await nextTick()
      expect(getFilterOptions(wrapper)[0].attributes('aria-selected')).toBe(
        'true'
      )
    })

    it('should toggle filter mode off when same chip is clicked again', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      await findFilterBarButton(wrapper, 'Input')!.trigger('click')
      await nextTick()
      await nextTick()

      expect(hasSidebar(wrapper)).toBe(true)
    })

    it('should reset filter query when re-entering filter mode', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      const input = wrapper.find('input[type="text"]')
      await input.setValue('IMAGE')
      await nextTick()

      await findFilterBarButton(wrapper, 'Input')!.trigger('click')
      await nextTick()
      await nextTick()

      await enterFilterMode(wrapper)

      expect((input.element as HTMLInputElement).value).toBe('')
    })

    it('should exit filter mode when cancel button is clicked', async () => {
      setupNodesWithTypes()
      const wrapper = await createWrapper()
      await enterFilterMode(wrapper)

      expect(hasSidebar(wrapper)).toBe(false)

      const cancelBtn = wrapper.find('[data-testid="cancel-filter"]')
      await cancelBtn.trigger('click')
      await nextTick()
      await nextTick()

      expect(hasSidebar(wrapper)).toBe(true)
    })
  })
})
