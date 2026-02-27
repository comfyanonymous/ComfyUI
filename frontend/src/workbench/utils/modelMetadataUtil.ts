import type { ModelFile } from '@/platform/workflow/validation/schemas/workflowSchema'

/**
 * Gets models from the node's `properties.models` field, excluding those
 * not currently selected in at least 1 of the node's widget values.
 *
 * @example
 * ```ts
 * const node = {
 *   type: 'CheckpointLoaderSimple',
 *   widgets_values: ['model1', 'model2'],
 *   properties: { models: [{ name: 'model1' }, { name: 'model2' }, { name: 'model3' }] }
 *   ... other properties
 * }
 * const selectedModels = getSelectedModelsMetadata(node)
 * // selectedModels = [{ name: 'model1' }, { name: 'model2' }]
 * ```
 *
 * @param node - The workflow node to process
 * @returns Filtered array containing only models that are currently selected
 */
export function getSelectedModelsMetadata(node: {
  type: string
  widgets_values?: unknown[] | Record<string, unknown>
  properties?: { models?: ModelFile[] }
}): ModelFile[] | undefined {
  try {
    if (!node.properties?.models?.length) return
    if (!node.widgets_values) return

    const widgetValues = Array.isArray(node.widgets_values)
      ? node.widgets_values
      : Object.values(node.widgets_values)

    if (!widgetValues.length) return

    const stringWidgetValues = new Set<string>()
    for (const widgetValue of widgetValues) {
      if (typeof widgetValue === 'string' && widgetValue.trim()) {
        stringWidgetValues.add(widgetValue)
      }
    }

    // Return the node's models that are present in the widget values
    return node.properties.models.filter((model) =>
      stringWidgetValues.has(model.name)
    )
  } catch (error) {
    console.error('Error filtering models by current selection:', error)
  }
}
