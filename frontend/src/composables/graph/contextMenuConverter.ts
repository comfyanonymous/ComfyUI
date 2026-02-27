import { default as DOMPurify } from 'dompurify'

import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type {
  IContextMenuValue,
  LGraphNode,
  IContextMenuOptions,
  ContextMenu
} from '@/lib/litegraph/src/litegraph'

import type { MenuOption, SubMenuOption } from './useMoreOptionsMenu'
import type { ContextMenuDivElement } from '@/lib/litegraph/src/interfaces'

/**
 * Hard blacklist - items that should NEVER be included
 */
const HARD_BLACKLIST = new Set([
  'Properties', // Never include Properties submenu
  'Colors', // Use singular "Color" instead
  'Shapes', // Use singular "Shape" instead
  'Title',
  'Mode',
  'Properties Panel',
  'Copy (Clipspace)'
])

/**
 * Core menu items - items that should appear in the main menu, not under Extensions
 * Includes both LiteGraph base menu items and ComfyUI built-in functionality
 */
const CORE_MENU_ITEMS = new Set([
  // Basic operations
  'Rename',
  'Copy',
  'Duplicate',
  'Clone',
  // Node state operations
  'Run Branch',
  'Pin',
  'Unpin',
  'Bypass',
  'Remove Bypass',
  'Mute',
  // Structure operations
  'Convert to Subgraph',
  'Frame selection',
  'Frame Nodes',
  'Minimize Node',
  'Expand',
  'Collapse',
  // Info and adjustments
  'Node Info',
  'Resize',
  'Title',
  'Properties Panel',
  'Adjust Size',
  // Visual
  'Color',
  'Colors',
  'Shape',
  'Shapes',
  'Mode',
  // Built-in node operations (node-specific)
  'Open Image',
  'Copy Image',
  'Save Image',
  'Open in Mask Editor',
  'Edit Subgraph Widgets',
  'Unpack Subgraph',
  'Copy (Clipspace)',
  'Paste (Clipspace)',
  // Selection and alignment
  'Align Selected To',
  'Distribute Nodes',
  // Deletion
  'Delete',
  'Remove',
  // LiteGraph base items
  'Show Advanced',
  'Hide Advanced'
])

/**
 * Normalize menu item label for duplicate detection
 * Handles variations like Colors/Color, Shapes/Shape, Pin/Unpin, Remove/Delete
 */
function normalizeLabel(label: string): string {
  return label
    .toLowerCase()
    .replace(/^un/, '') // Remove 'un' prefix (Unpin -> Pin)
    .trim()
}

/**
 * Check if a similar menu item already exists in the results
 * Returns true if an item with the same normalized label exists
 */
function isDuplicateItem(label: string, existingItems: MenuOption[]): boolean {
  const normalizedLabel = normalizeLabel(label)

  // Map of equivalent items
  const equivalents: Record<string, string[]> = {
    color: ['color', 'colors'],
    shape: ['shape', 'shapes'],
    pin: ['pin', 'unpin'],
    delete: ['remove', 'delete'],
    duplicate: ['clone', 'duplicate'],
    frame: ['frame selection', 'frame nodes']
  }

  return existingItems.some((item) => {
    if (!item.label) return false

    const existingNormalized = normalizeLabel(item.label)

    // Check direct match
    if (existingNormalized === normalizedLabel) return true

    // Check if they're in the same equivalence group
    for (const values of Object.values(equivalents)) {
      if (
        values.includes(normalizedLabel) &&
        values.includes(existingNormalized)
      ) {
        return true
      }
    }

    return false
  })
}

/**
 * Check if a menu item is a core menu item (not an extension)
 * Core items include LiteGraph base items and ComfyUI built-in functionality
 */
function isCoreMenuItem(label: string): boolean {
  return CORE_MENU_ITEMS.has(label)
}

/**
 * Filter out duplicate menu items based on label
 * Gives precedence to Vue hardcoded options over LiteGraph options
 */
function removeDuplicateMenuOptions(options: MenuOption[]): MenuOption[] {
  // Group items by label
  const itemsByLabel = new Map<string, MenuOption[]>()
  const itemsWithoutLabel: MenuOption[] = []

  for (const opt of options) {
    // Always keep dividers and category items
    if (opt.type === 'divider' || opt.type === 'category') {
      itemsWithoutLabel.push(opt)
      continue
    }

    // Items without labels are kept as-is
    if (!opt.label) {
      itemsWithoutLabel.push(opt)
      continue
    }

    // Group by label
    if (!itemsByLabel.has(opt.label)) {
      itemsByLabel.set(opt.label, [])
    }
    itemsByLabel.get(opt.label)!.push(opt)
  }

  // Select best item for each label (prefer vue over litegraph)
  const result: MenuOption[] = []
  const seenLabels = new Set<string>()

  for (const opt of options) {
    // Add non-labeled items in original order
    if (opt.type === 'divider' || opt.type === 'category' || !opt.label) {
      if (itemsWithoutLabel.includes(opt)) {
        result.push(opt)
        const idx = itemsWithoutLabel.indexOf(opt)
        itemsWithoutLabel.splice(idx, 1)
      }
      continue
    }

    // Skip if we already processed this label
    if (seenLabels.has(opt.label)) {
      continue
    }
    seenLabels.add(opt.label)

    // Get all items with this label
    const duplicates = itemsByLabel.get(opt.label)!

    // If only one item, add it
    if (duplicates.length === 1) {
      result.push(duplicates[0])
      continue
    }

    // Multiple items: prefer vue source over litegraph
    const vueItem = duplicates.find((item) => item.source === 'vue')
    if (vueItem) {
      result.push(vueItem)
    } else {
      // No vue item, just take the first one
      result.push(duplicates[0])
    }
  }

  return result
}

/**
 * Order groups for menu items - defines the display order of sections
 */
const MENU_ORDER: string[] = [
  // Section 1: Basic operations
  'Rename',
  'Copy',
  'Duplicate',
  // Section 2: Node actions
  'Run Branch',
  'Pin',
  'Unpin',
  'Bypass',
  'Remove Bypass',
  'Mute',
  // Section 3: Structure operations
  'Convert to Subgraph',
  'Frame selection',
  'Frame Nodes',
  'Minimize Node',
  'Expand',
  'Collapse',
  'Resize',
  'Clone',
  // Section 4: Node properties
  'Node Info',
  'Color',
  // Section 5: Node-specific operations
  'Open in Mask Editor',
  'Open Image',
  'Copy Image',
  'Save Image',
  'Copy (Clipspace)',
  'Paste (Clipspace)',
  // Fallback for other core items
  'Convert to Group Node (Deprecated)'
]

/**
 * Get the order index for a menu item (lower = earlier in menu)
 */
function getMenuItemOrder(label: string): number {
  const index = MENU_ORDER.indexOf(label)
  return index === -1 ? 999 : index
}

/**
 * Build structured menu with core items first, then extensions under a labeled section
 * Ensures Delete always appears at the bottom
 */
export function buildStructuredMenu(options: MenuOption[]): MenuOption[] {
  // First, remove duplicates (giving precedence to Vue hardcoded options)
  const deduplicated = removeDuplicateMenuOptions(options)
  const coreItemsMap = new Map<string, MenuOption>()
  const extensionItems: MenuOption[] = []
  let deleteItem: MenuOption | undefined

  // Separate items into core and extension categories
  for (const option of deduplicated) {
    // Skip dividers for now - we'll add them between sections later
    if (option.type === 'divider') {
      continue
    }

    // Skip category labels (they'll be added separately)
    if (option.type === 'category') {
      continue
    }

    // Check if this is the Delete/Remove item - save it for the end
    const isDeleteItem = option.label === 'Delete' || option.label === 'Remove'
    if (isDeleteItem && !option.hasSubmenu) {
      deleteItem = option
      continue
    }

    // Categorize based on label
    if (option.label && isCoreMenuItem(option.label)) {
      coreItemsMap.set(option.label, option)
    } else {
      extensionItems.push(option)
    }
  }
  // Build ordered core items based on MENU_ORDER
  const orderedCoreItems: MenuOption[] = []
  const coreLabels = Array.from(coreItemsMap.keys())
  coreLabels.sort((a, b) => getMenuItemOrder(a) - getMenuItemOrder(b))

  // Section boundaries based on MENU_ORDER indices
  // Section 1: 0-2 (Rename, Copy, Duplicate)
  // Section 2: 3-8 (Run Branch, Pin, Unpin, Bypass, Remove Bypass, Mute)
  // Section 3: 9-15 (Convert to Subgraph, Frame selection, Minimize Node, Expand, Collapse, Resize, Clone)
  // Section 4: 16-17 (Node Info, Color)
  // Section 5: 18+ (Image operations and fallback items)
  const getSectionNumber = (index: number): number => {
    if (index <= 2) return 1
    if (index <= 8) return 2
    if (index <= 15) return 3
    if (index <= 17) return 4
    return 5
  }

  let lastSection = 0
  for (const label of coreLabels) {
    const item = coreItemsMap.get(label)!
    const itemIndex = getMenuItemOrder(label)
    const currentSection = getSectionNumber(itemIndex)

    // Add divider when moving to a new section
    if (lastSection > 0 && currentSection !== lastSection) {
      orderedCoreItems.push({ type: 'divider' })
    }

    orderedCoreItems.push(item)
    lastSection = currentSection
  }

  // Build the final menu structure
  const result: MenuOption[] = []

  // Add ordered core items with their dividers
  result.push(...orderedCoreItems)

  // Add extensions section if there are extension items
  if (extensionItems.length > 0) {
    // Add divider before Extensions section
    result.push({ type: 'divider' })

    // Add non-clickable Extensions label
    result.push({
      label: 'Extensions',
      type: 'category',
      disabled: true
    })

    // Add extension items
    result.push(...extensionItems)
  }

  // Add Delete at the bottom if it exists
  if (deleteItem) {
    result.push({ type: 'divider' })
    result.push(deleteItem)
  }

  return result
}

/**
 * Convert LiteGraph IContextMenuValue items to Vue MenuOption format
 * Used to bridge LiteGraph context menus into Vue node menus
 * @param items - The LiteGraph menu items to convert
 * @param node - The node context (optional)
 * @param applyStructuring - Whether to apply menu structuring (core/extensions separation). Defaults to true.
 */
export function convertContextMenuToOptions(
  items: (IContextMenuValue | null)[],
  node?: LGraphNode,
  applyStructuring: boolean = true
): MenuOption[] {
  const result: MenuOption[] = []

  for (const item of items) {
    // Null items are separators in LiteGraph
    if (item === null) {
      result.push({ type: 'divider' })
      continue
    }

    // Skip items without content (shouldn't happen, but be safe)
    if (!item.content) {
      continue
    }

    // Skip hard blacklisted items
    if (HARD_BLACKLIST.has(item.content)) {
      continue
    }

    // Skip if a similar item already exists in results
    if (isDuplicateItem(item.content, result)) {
      continue
    }

    const option: MenuOption = {
      label: item.content,
      source: 'litegraph'
    }

    // Pass through disabled state
    if (item.disabled) {
      option.disabled = true
    }

    // Handle submenus
    if (item.has_submenu) {
      // Static submenu with pre-defined options
      if (item.submenu?.options) {
        option.hasSubmenu = true
        option.submenu = convertSubmenuToOptions(item.submenu.options)
      }
      // Dynamic submenu - callback creates it on-demand
      else if (item.callback && !item.disabled) {
        option.hasSubmenu = true
        // Intercept the callback to capture dynamic submenu items
        const capturedSubmenu = captureDynamicSubmenu(item, node)
        if (capturedSubmenu) {
          option.submenu = capturedSubmenu
        } else {
          console.warn(
            '[ContextMenuConverter] Failed to capture submenu for:',
            item.content
          )
        }
      }
    }
    // Handle callback (only if not disabled and not a submenu)
    else if (item.callback && !item.disabled) {
      // Wrap the callback to match the () => void signature
      option.action = () => {
        try {
          void item.callback?.call(
            item as unknown as ContextMenuDivElement,
            item.value,
            {},
            undefined,
            undefined,
            item
          )
        } catch (error) {
          console.error('Error executing context menu callback:', error)
        }
      }
    }

    result.push(option)
  }

  // Apply structured menu with core items and extensions section (if requested)
  if (applyStructuring) {
    return buildStructuredMenu(result)
  }

  return result
}

/**
 * Capture submenu items from a dynamic submenu callback
 * Intercepts ContextMenu constructor to extract items without creating HTML menu
 */
function captureDynamicSubmenu(
  item: IContextMenuValue,
  node?: LGraphNode
): SubMenuOption[] | undefined {
  let capturedItems: readonly (IContextMenuValue | string | null)[] | undefined
  let capturedOptions: IContextMenuOptions | undefined

  // Store original ContextMenu constructor
  const OriginalContextMenu = LiteGraph.ContextMenu

  try {
    // Mock ContextMenu constructor to capture submenu items and options
    LiteGraph.ContextMenu = function (
      items: readonly (IContextMenuValue | string | null)[],
      options?: IContextMenuOptions
    ) {
      // Capture both items and options
      capturedItems = items
      capturedOptions = options
      // Return a minimal mock object to prevent errors
      return {
        close: () => {},
        root: document.createElement('div')
      } as unknown as ContextMenu
    } as unknown as typeof ContextMenu

    // Execute the callback to trigger submenu creation
    try {
      // Create a mock MouseEvent for the callback
      const mockEvent = new MouseEvent('click', {
        bubbles: true,
        cancelable: true,
        clientX: 0,
        clientY: 0
      })

      // Create a mock parent menu
      const mockMenu = {
        close: () => {},
        root: document.createElement('div')
      } as unknown as ContextMenu

      // Call the callback which should trigger ContextMenu constructor
      // Callback signature varies, but typically: (value, options, event, menu, node)
      void item.callback?.call(
        item as unknown as ContextMenuDivElement,
        item.value,
        {},
        mockEvent,
        mockMenu,
        node // Pass the node context for callbacks that need it
      )
    } catch (error) {
      console.warn(
        '[ContextMenuConverter] Error executing callback for:',
        item.content,
        error
      )
    }
  } finally {
    // Always restore original constructor
    LiteGraph.ContextMenu = OriginalContextMenu
  }

  // Convert captured items to Vue submenu format
  if (capturedItems) {
    const converted = convertSubmenuToOptions(capturedItems, capturedOptions)
    return converted
  }

  console.warn('[ContextMenuConverter] No items captured for:', item.content)
  return undefined
}

/**
 * Convert LiteGraph submenu items to Vue SubMenuOption format
 */
function convertSubmenuToOptions(
  items: readonly (IContextMenuValue | string | null)[],
  options?: IContextMenuOptions
): SubMenuOption[] {
  const result: SubMenuOption[] = []

  for (const item of items) {
    // Skip null separators
    if (item === null) {
      continue
    }

    // Handle string items (simple labels like in Mode/Shapes menus)
    if (typeof item === 'string') {
      const subOption: SubMenuOption = {
        label: item,
        action: () => {
          try {
            // Call the options callback with the string value
            if (options?.callback) {
              void options.callback.call(
                null,
                item,
                options,
                undefined,
                undefined,
                options.extra
              )
            }
          } catch (error) {
            console.error('Error executing string item callback:', error)
          }
        }
      }
      result.push(subOption)
      continue
    }

    // Handle object items
    if (!item.content) {
      continue
    }

    // Extract text content from HTML if present
    const content = stripHtmlTags(item.content)

    const subOption: SubMenuOption = {
      label: content,
      action: () => {
        try {
          void item.callback?.call(
            item as unknown as ContextMenuDivElement,
            item.value,
            {},
            undefined,
            undefined,
            item
          )
        } catch (error) {
          console.error('Error executing submenu callback:', error)
        }
      }
    }

    // Pass through disabled state
    if (item.disabled) {
      subOption.disabled = true
    }

    result.push(subOption)
  }
  return result
}

/**
 * Strip HTML tags from content string safely
 * LiteGraph menu items often include HTML for styling
 */
function stripHtmlTags(html: string): string {
  // Use DOMPurify to sanitize and strip all HTML tags
  const sanitized = DOMPurify.sanitize(html, { ALLOWED_TAGS: [] })
  const result = sanitized.trim()
  return result || html.replace(/<[^>]*>/g, '').trim() || html
}
