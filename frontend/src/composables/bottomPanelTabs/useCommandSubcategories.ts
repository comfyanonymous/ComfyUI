import { computed } from 'vue'
import type { ComputedRef } from 'vue'

import type { ComfyCommandImpl } from '@/stores/commandStore'

type SubcategoryRule = {
  pattern: string | RegExp
  subcategory: string
}

type SubcategoryConfig = {
  defaultSubcategory: string
  rules: SubcategoryRule[]
}

/**
 * Composable for grouping commands by subcategory based on configurable rules
 */
export function useCommandSubcategories(
  commands: ComputedRef<ComfyCommandImpl[]>,
  config: SubcategoryConfig
) {
  const subcategories = computed(() => {
    const result: Record<string, ComfyCommandImpl[]> = {}

    for (const command of commands.value) {
      let subcategory = config.defaultSubcategory

      // Find the first matching rule
      for (const rule of config.rules) {
        const matches =
          typeof rule.pattern === 'string'
            ? command.id.includes(rule.pattern)
            : rule.pattern.test(command.id)

        if (matches) {
          subcategory = rule.subcategory
          break
        }
      }

      if (!result[subcategory]) {
        result[subcategory] = []
      }
      result[subcategory].push(command)
    }

    return result
  })

  return {
    subcategories
  }
}

/**
 * Predefined configuration for view controls subcategories
 */
export const VIEW_CONTROLS_CONFIG: SubcategoryConfig = {
  defaultSubcategory: 'view',
  rules: [
    { pattern: 'Zoom', subcategory: 'view' },
    { pattern: 'Fit', subcategory: 'view' },
    { pattern: 'Panel', subcategory: 'panel-controls' },
    { pattern: 'Sidebar', subcategory: 'panel-controls' }
  ]
}

/**
 * Predefined configuration for essentials subcategories
 */
export const ESSENTIALS_CONFIG: SubcategoryConfig = {
  defaultSubcategory: 'workflow',
  rules: [
    { pattern: 'Workflow', subcategory: 'workflow' },
    { pattern: 'Node', subcategory: 'node' },
    { pattern: 'Queue', subcategory: 'queue' }
  ]
}
