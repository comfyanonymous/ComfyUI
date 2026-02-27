import { describe, it, expect } from 'vitest'

import type { MenuOption } from './useMoreOptionsMenu'
import {
  buildStructuredMenu,
  convertContextMenuToOptions
} from './contextMenuConverter'

describe('contextMenuConverter', () => {
  describe('buildStructuredMenu', () => {
    it('should order core items before extension items', () => {
      const options: MenuOption[] = [
        { label: 'Custom Extension Item', source: 'litegraph' },
        { label: 'Copy', source: 'vue' },
        { label: 'Rename', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      // Core items (Rename, Copy) should come before extension items
      const renameIndex = result.findIndex((opt) => opt.label === 'Rename')
      const copyIndex = result.findIndex((opt) => opt.label === 'Copy')
      const extensionIndex = result.findIndex(
        (opt) => opt.label === 'Custom Extension Item'
      )

      expect(renameIndex).toBeLessThan(extensionIndex)
      expect(copyIndex).toBeLessThan(extensionIndex)
    })

    it('should add Extensions category label before extension items', () => {
      const options: MenuOption[] = [
        { label: 'Copy', source: 'vue' },
        { label: 'My Custom Extension', source: 'litegraph' }
      ]

      const result = buildStructuredMenu(options)

      const extensionsLabel = result.find(
        (opt) => opt.label === 'Extensions' && opt.type === 'category'
      )
      expect(extensionsLabel).toBeDefined()
      expect(extensionsLabel?.disabled).toBe(true)
    })

    it('should place Delete at the very end', () => {
      const options: MenuOption[] = [
        { label: 'Delete', action: () => {}, source: 'vue' },
        { label: 'Copy', source: 'vue' },
        { label: 'Rename', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      const lastNonDivider = [...result]
        .reverse()
        .find((opt) => opt.type !== 'divider')
      expect(lastNonDivider?.label).toBe('Delete')
    })

    it('should deduplicate items with same label, preferring vue source', () => {
      const options: MenuOption[] = [
        { label: 'Copy', action: () => {}, source: 'litegraph' },
        { label: 'Copy', action: () => {}, source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      const copyItems = result.filter((opt) => opt.label === 'Copy')
      expect(copyItems).toHaveLength(1)
      expect(copyItems[0].source).toBe('vue')
    })

    it('should preserve dividers between sections', () => {
      const options: MenuOption[] = [
        { label: 'Rename', source: 'vue' },
        { label: 'Copy', source: 'vue' },
        { label: 'Pin', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      const dividers = result.filter((opt) => opt.type === 'divider')
      expect(dividers.length).toBeGreaterThan(0)
    })

    it('should handle empty input', () => {
      const result = buildStructuredMenu([])
      expect(result).toEqual([])
    })

    it('should handle only dividers', () => {
      const options: MenuOption[] = [{ type: 'divider' }, { type: 'divider' }]

      const result = buildStructuredMenu(options)

      // Should be empty since dividers are filtered initially
      expect(result).toEqual([])
    })

    it('should recognize Remove as equivalent to Delete', () => {
      const options: MenuOption[] = [
        { label: 'Remove', action: () => {}, source: 'vue' },
        { label: 'Copy', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      // Remove should be placed at the end like Delete
      const lastNonDivider = [...result]
        .reverse()
        .find((opt) => opt.type !== 'divider')
      expect(lastNonDivider?.label).toBe('Remove')
    })

    it('should group core items in correct section order', () => {
      const options: MenuOption[] = [
        { label: 'Color', source: 'vue' },
        { label: 'Node Info', source: 'vue' },
        { label: 'Pin', source: 'vue' },
        { label: 'Rename', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      // Get indices of items (excluding dividers and categories)
      const getIndex = (label: string) =>
        result.findIndex((opt) => opt.label === label)

      // Rename (section 1) should come before Pin (section 2)
      expect(getIndex('Rename')).toBeLessThan(getIndex('Pin'))
      // Pin (section 2) should come before Node Info (section 4)
      expect(getIndex('Pin')).toBeLessThan(getIndex('Node Info'))
      // Node Info (section 4) should come before or with Color (section 4)
      expect(getIndex('Node Info')).toBeLessThanOrEqual(getIndex('Color'))
    })

    it('should recognize Frame Nodes as a core menu item', () => {
      const options: MenuOption[] = [
        { label: 'Rename', source: 'vue' },
        { label: 'Frame Nodes', source: 'vue' },
        { label: 'Custom Extension', source: 'vue' }
      ]

      const result = buildStructuredMenu(options)

      // Frame Nodes should appear in the core items section (before Extensions)
      const frameNodesIndex = result.findIndex(
        (opt) => opt.label === 'Frame Nodes'
      )
      const extensionsCategoryIndex = result.findIndex(
        (opt) => opt.label === 'Extensions' && opt.type === 'category'
      )

      // Frame Nodes should come before Extensions category
      expect(frameNodesIndex).toBeLessThan(extensionsCategoryIndex)
    })
  })

  describe('convertContextMenuToOptions', () => {
    it('should convert empty array to empty result', () => {
      const result = convertContextMenuToOptions([])
      expect(result).toEqual([])
    })

    it('should convert null items to dividers', () => {
      const result = convertContextMenuToOptions([null], undefined, false)
      expect(result).toHaveLength(1)
      expect(result[0].type).toBe('divider')
    })

    it('should skip blacklisted items like Properties', () => {
      const items = [{ content: 'Properties', callback: () => {} }]
      const result = convertContextMenuToOptions(items, undefined, false)
      expect(result.find((opt) => opt.label === 'Properties')).toBeUndefined()
    })

    it('should convert basic menu items with content', () => {
      const items = [{ content: 'Test Item', callback: () => {} }]
      const result = convertContextMenuToOptions(items, undefined, false)
      expect(result).toHaveLength(1)
      expect(result[0].label).toBe('Test Item')
    })

    it('should mark items as litegraph source', () => {
      const items = [{ content: 'Test Item', callback: () => {} }]
      const result = convertContextMenuToOptions(items, undefined, false)
      expect(result[0].source).toBe('litegraph')
    })

    it('should pass through disabled state', () => {
      const items = [{ content: 'Disabled Item', disabled: true }]
      const result = convertContextMenuToOptions(items, undefined, false)
      expect(result[0].disabled).toBe(true)
    })

    it('should apply structuring by default', () => {
      const items = [
        { content: 'Copy', callback: () => {} },
        { content: 'Custom Extension', callback: () => {} }
      ]
      const result = convertContextMenuToOptions(items)

      // With structuring, there should be Extensions category
      const hasExtensionsCategory = result.some(
        (opt) => opt.label === 'Extensions' && opt.type === 'category'
      )
      expect(hasExtensionsCategory).toBe(true)
    })
  })
})
