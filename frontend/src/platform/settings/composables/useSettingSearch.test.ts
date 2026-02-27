import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { st } from '@/i18n'
import { useSettingSearch } from '@/platform/settings/composables/useSettingSearch'
import {
  getSettingInfo,
  useSettingStore
} from '@/platform/settings/settingStore'
import type { SettingTreeNode } from '@/platform/settings/settingStore'

// Test-specific type for mock settings
interface MockSettingParams {
  id: string
  name: string
  type: string
  defaultValue: unknown
  category?: string[]
  deprecated?: boolean
}

// Mock dependencies
vi.mock('@/i18n', () => ({
  st: vi.fn((_: string, fallback: string) => fallback)
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(),
  getSettingInfo: vi.fn()
}))

describe('useSettingSearch', () => {
  let mockSettingStore: ReturnType<typeof useSettingStore>
  let mockSettings: Record<string, MockSettingParams>

  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()

    // Mock settings data
    mockSettings = {
      'Category.Setting1': {
        id: 'Category.Setting1',
        name: 'Setting One',
        type: 'text',
        defaultValue: 'default',
        category: ['Category', 'Basic']
      },
      'Category.Setting2': {
        id: 'Category.Setting2',
        name: 'Setting Two',
        type: 'boolean',
        defaultValue: false,
        category: ['Category', 'Advanced']
      },
      'Category.HiddenSetting': {
        id: 'Category.HiddenSetting',
        name: 'Hidden Setting',
        type: 'hidden',
        defaultValue: 'hidden',
        category: ['Category', 'Basic']
      },
      'Category.DeprecatedSetting': {
        id: 'Category.DeprecatedSetting',
        name: 'Deprecated Setting',
        type: 'text',
        defaultValue: 'deprecated',
        deprecated: true,
        category: ['Category', 'Advanced']
      },
      'Other.Setting3': {
        id: 'Other.Setting3',
        name: 'Other Setting',
        type: 'select',
        defaultValue: 'option1',
        category: ['Other', 'SubCategory']
      }
    }

    // Mock setting store
    mockSettingStore = {
      settingsById: mockSettings
    } as ReturnType<typeof useSettingStore>
    vi.mocked(useSettingStore).mockReturnValue(mockSettingStore)

    // Mock getSettingInfo function
    vi.mocked(getSettingInfo).mockImplementation((setting) => {
      const parts = setting.category || setting.id.split('.')
      return {
        category: parts[0] ?? 'Other',
        subCategory: parts[1] ?? 'Other'
      }
    })

    // Mock st function to return fallback value
    vi.mocked(st).mockImplementation((_: string, fallback: string) => fallback)
  })

  describe('initialization', () => {
    it('initializes with default state', () => {
      const search = useSettingSearch()

      expect(search.searchQuery.value).toBe('')
      expect(search.filteredSettingIds.value).toEqual([])
      expect(search.searchInProgress.value).toBe(false)
      expect(search.queryIsEmpty.value).toBe(true)
      expect(search.inSearch.value).toBe(false)
      expect(search.searchResultsCategories.value).toEqual(new Set())
    })
  })

  describe('reactive properties', () => {
    it('queryIsEmpty computed property works correctly', () => {
      const search = useSettingSearch()

      expect(search.queryIsEmpty.value).toBe(true)

      search.searchQuery.value = 'test'
      expect(search.queryIsEmpty.value).toBe(false)

      search.searchQuery.value = ''
      expect(search.queryIsEmpty.value).toBe(true)
    })

    it('inSearch computed property works correctly', () => {
      const search = useSettingSearch()

      // Empty query, not in search
      expect(search.inSearch.value).toBe(false)

      // Has query but search in progress
      search.searchQuery.value = 'test'
      search.searchInProgress.value = true
      expect(search.inSearch.value).toBe(false)

      // Has query and search complete
      search.searchInProgress.value = false
      expect(search.inSearch.value).toBe(true)
    })

    it('searchResultsCategories computed property works correctly', () => {
      const search = useSettingSearch()

      // No results
      expect(search.searchResultsCategories.value).toEqual(new Set())

      // Add some filtered results
      search.filteredSettingIds.value = ['Category.Setting1', 'Other.Setting3']
      expect(search.searchResultsCategories.value).toEqual(
        new Set(['Category', 'Other'])
      )
    })

    it('watches searchQuery and sets searchInProgress to true', async () => {
      const search = useSettingSearch()

      expect(search.searchInProgress.value).toBe(false)

      search.searchQuery.value = 'test'
      await nextTick()

      expect(search.searchInProgress.value).toBe(true)
    })
  })

  describe('handleSearch', () => {
    it('clears results when query is empty', () => {
      const search = useSettingSearch()
      search.filteredSettingIds.value = ['Category.Setting1']

      search.handleSearch('')

      expect(search.filteredSettingIds.value).toEqual([])
    })

    it('filters settings by ID (case insensitive)', () => {
      const search = useSettingSearch()

      search.handleSearch('category.setting1')

      expect(search.filteredSettingIds.value).toContain('Category.Setting1')
      expect(search.filteredSettingIds.value).not.toContain('Other.Setting3')
    })

    it('filters settings by name (case insensitive)', () => {
      const search = useSettingSearch()

      search.handleSearch('setting one')

      expect(search.filteredSettingIds.value).toContain('Category.Setting1')
      expect(search.filteredSettingIds.value).not.toContain('Category.Setting2')
    })

    it('filters settings by category', () => {
      const search = useSettingSearch()

      search.handleSearch('other')

      expect(search.filteredSettingIds.value).toContain('Other.Setting3')
      expect(search.filteredSettingIds.value).not.toContain('Category.Setting1')
    })

    it('excludes hidden settings from results', () => {
      const search = useSettingSearch()

      search.handleSearch('hidden')

      expect(search.filteredSettingIds.value).not.toContain(
        'Category.HiddenSetting'
      )
    })

    it('excludes deprecated settings from results', () => {
      const search = useSettingSearch()

      search.handleSearch('deprecated')

      expect(search.filteredSettingIds.value).not.toContain(
        'Category.DeprecatedSetting'
      )
    })

    it('sets searchInProgress to false after search', () => {
      const search = useSettingSearch()
      search.searchInProgress.value = true

      search.handleSearch('test')

      expect(search.searchInProgress.value).toBe(false)
    })

    it('includes visible settings in results', () => {
      const search = useSettingSearch()

      search.handleSearch('setting')

      expect(search.filteredSettingIds.value).toEqual(
        expect.arrayContaining([
          'Category.Setting1',
          'Category.Setting2',
          'Other.Setting3'
        ])
      )
      expect(search.filteredSettingIds.value).not.toContain(
        'Category.HiddenSetting'
      )
      expect(search.filteredSettingIds.value).not.toContain(
        'Category.DeprecatedSetting'
      )
    })

    it('includes all visible settings in comprehensive search', () => {
      const search = useSettingSearch()

      // Search for a partial match that should include multiple settings
      search.handleSearch('setting')

      // Should find all visible settings (not hidden/deprecated)
      expect(search.filteredSettingIds.value.length).toBeGreaterThan(0)
      expect(search.filteredSettingIds.value).toEqual(
        expect.arrayContaining([
          'Category.Setting1',
          'Category.Setting2',
          'Other.Setting3'
        ])
      )
    })

    it('uses translated categories for search', () => {
      const search = useSettingSearch()

      // Mock st to return translated category names
      vi.mocked(st).mockImplementation((key: string, fallback: string) => {
        if (key === 'settingsCategories.Category') {
          return 'Translated Category'
        }
        return fallback
      })

      search.handleSearch('translated category')

      expect(search.filteredSettingIds.value).toEqual(
        expect.arrayContaining(['Category.Setting1', 'Category.Setting2'])
      )
    })
  })

  describe('getSearchResults', () => {
    it('groups results by subcategory', () => {
      const search = useSettingSearch()
      search.filteredSettingIds.value = [
        'Category.Setting1',
        'Category.Setting2'
      ]

      const results = search.getSearchResults(null)

      expect(results).toEqual([
        {
          label: 'Basic',
          category: 'Category',
          settings: [mockSettings['Category.Setting1']]
        },
        {
          label: 'Advanced',
          category: 'Category',
          settings: [mockSettings['Category.Setting2']]
        }
      ])
    })

    it('filters results by active category', () => {
      const search = useSettingSearch()
      search.filteredSettingIds.value = ['Category.Setting1', 'Other.Setting3']

      const activeCategory: Partial<SettingTreeNode> = { label: 'Category' }
      const results = search.getSearchResults(activeCategory as SettingTreeNode)

      expect(results).toEqual([
        {
          label: 'Basic',
          settings: [mockSettings['Category.Setting1']]
        }
      ])
    })

    it('returns all results when no active category', () => {
      const search = useSettingSearch()
      search.filteredSettingIds.value = ['Category.Setting1', 'Other.Setting3']

      const results = search.getSearchResults(null)

      expect(results).toEqual([
        {
          label: 'Basic',
          category: 'Category',
          settings: [mockSettings['Category.Setting1']]
        },
        {
          label: 'SubCategory',
          category: 'Other',
          settings: [mockSettings['Other.Setting3']]
        }
      ])
    })

    it('returns results from all categories when searching cross-category term', () => {
      // Simulates the "badge" scenario: same term matches settings in
      // multiple categories (e.g. LiteGraph and Comfy)
      mockSettings['LiteGraph.BadgeSetting'] = {
        id: 'LiteGraph.BadgeSetting',
        name: 'Node source badge mode',
        type: 'combo',
        defaultValue: 'default',
        category: ['LiteGraph', 'Node']
      }
      mockSettings['Comfy.BadgeSetting'] = {
        id: 'Comfy.BadgeSetting',
        name: 'Show API node pricing badge',
        type: 'boolean',
        defaultValue: true,
        category: ['Comfy', 'API Nodes']
      }

      const search = useSettingSearch()
      search.handleSearch('badge')

      expect(search.filteredSettingIds.value).toContain(
        'LiteGraph.BadgeSetting'
      )
      expect(search.filteredSettingIds.value).toContain('Comfy.BadgeSetting')

      // getSearchResults(null) should return both categories' results
      const results = search.getSearchResults(null)
      const labels = results.map((g) => g.label)
      expect(labels).toContain('Node')
      expect(labels).toContain('API Nodes')
    })

    it('returns empty array when no filtered results', () => {
      const search = useSettingSearch()
      search.filteredSettingIds.value = []

      const results = search.getSearchResults(null)

      expect(results).toEqual([])
    })

    it('handles multiple settings in same subcategory', () => {
      const search = useSettingSearch()

      // Add another setting to Basic subcategory
      mockSettings['Category.Setting4'] = {
        id: 'Category.Setting4',
        name: 'Setting Four',
        type: 'text',
        defaultValue: 'default',
        category: ['Category', 'Basic']
      }

      search.filteredSettingIds.value = [
        'Category.Setting1',
        'Category.Setting4'
      ]

      const results = search.getSearchResults(null)

      expect(results).toEqual([
        {
          label: 'Basic',
          category: 'Category',
          settings: [
            mockSettings['Category.Setting1'],
            mockSettings['Category.Setting4']
          ]
        }
      ])
    })
  })

  describe('nav item matching', () => {
    const navItems = [
      { key: 'keybinding', label: 'Keybinding' },
      { key: 'about', label: 'About' },
      { key: 'extension', label: 'Extension' },
      { key: 'Comfy', label: 'Comfy' }
    ]

    it('matches nav items by key', () => {
      const search = useSettingSearch()

      search.handleSearch('keybinding', navItems)

      expect(search.matchedNavItemKeys.value.has('keybinding')).toBe(true)
    })

    it('matches nav items by translated label (case insensitive)', () => {
      const search = useSettingSearch()

      search.handleSearch('ABOUT', navItems)

      expect(search.matchedNavItemKeys.value.has('about')).toBe(true)
    })

    it('matches partial nav item labels', () => {
      const search = useSettingSearch()

      search.handleSearch('ext', navItems)

      expect(search.matchedNavItemKeys.value.has('extension')).toBe(true)
    })

    it('clears matched nav item keys on empty query', () => {
      const search = useSettingSearch()

      search.handleSearch('keybinding', navItems)
      expect(search.matchedNavItemKeys.value.size).toBeGreaterThan(0)

      search.handleSearch('', navItems)
      expect(search.matchedNavItemKeys.value.size).toBe(0)
    })

    it('can match both settings and nav items simultaneously', () => {
      const search = useSettingSearch()

      search.handleSearch('other', navItems)

      expect(search.filteredSettingIds.value).toContain('Other.Setting3')
      expect(search.matchedNavItemKeys.value.size).toBe(0)
    })

    it('matches nav items with translated labels different from key', () => {
      const translatedNavItems = [{ key: 'keybinding', label: '키 바인딩' }]
      const search = useSettingSearch()

      search.handleSearch('키 바인딩', translatedNavItems)

      expect(search.matchedNavItemKeys.value.has('keybinding')).toBe(true)
    })

    it('does not match nav items when no nav items provided', () => {
      const search = useSettingSearch()

      search.handleSearch('keybinding')

      expect(search.matchedNavItemKeys.value.size).toBe(0)
    })
  })

  describe('edge cases', () => {
    it('handles empty settings store', () => {
      mockSettingStore.settingsById = {}
      const search = useSettingSearch()

      search.handleSearch('test')

      expect(search.filteredSettingIds.value).toEqual([])
    })

    it('handles settings with undefined category', () => {
      mockSettings['NoCategorySetting'] = {
        id: 'NoCategorySetting',
        name: 'No Category',
        type: 'text',
        defaultValue: 'default'
      }

      const search = useSettingSearch()

      search.handleSearch('category')

      expect(search.filteredSettingIds.value).toContain('NoCategorySetting')
    })

    it('handles special characters in search query', () => {
      const search = useSettingSearch()

      // Search for part of the ID that contains a dot
      search.handleSearch('category.setting')

      expect(search.filteredSettingIds.value).toContain('Category.Setting1')
    })

    it('handles very long search queries', () => {
      const search = useSettingSearch()
      const longQuery = 'a'.repeat(1000)

      search.handleSearch(longQuery)

      expect(search.filteredSettingIds.value).toEqual([])
    })

    it('handles rapid consecutive searches', async () => {
      const search = useSettingSearch()

      search.handleSearch('setting')
      search.handleSearch('other')
      search.handleSearch('category')

      expect(search.filteredSettingIds.value).toEqual(
        expect.arrayContaining(['Category.Setting1', 'Category.Setting2'])
      )
    })
  })
})
