import { useToast } from 'primevue/usetoast'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { clearPreservedQuery } from '@/platform/navigation/preservedQueryManager'
import { PRESERVED_QUERY_NAMESPACES } from '@/platform/navigation/preservedQueryNamespaces'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

import { useTemplateWorkflows } from './useTemplateWorkflows'

/**
 * Composable for loading templates from URL query parameters
 *
 * Supports URLs like:
 * - /?template=flux_simple (loads with default source)
 * - /?template=flux_simple&source=custom (loads from custom source)
 * - /?template=flux_simple&mode=linear (loads template in linear mode)
 *
 * Input validation:
 * - Template, source, and mode parameters must match: ^[a-zA-Z0-9_-]+$
 * - Invalid formats are rejected with console warnings
 */
export function useTemplateUrlLoader() {
  const route = useRoute()
  const router = useRouter()
  const { t } = useI18n()
  const toast = useToast()
  const templateWorkflows = useTemplateWorkflows()
  const canvasStore = useCanvasStore()
  const TEMPLATE_NAMESPACE = PRESERVED_QUERY_NAMESPACES.TEMPLATE
  const SUPPORTED_MODES = ['linear'] as const
  type SupportedMode = (typeof SUPPORTED_MODES)[number]

  /**
   * Validates parameter format to prevent path traversal and injection attacks
   * Allows: letters, numbers, underscores, hyphens, and dots (for version numbers)
   * Blocks: path separators (/, \), special chars that could enable injection
   */
  const isValidParameter = (param: string): boolean => {
    return /^[a-zA-Z0-9_.-]+$/.test(param)
  }

  /**
   * Type guard to check if a value is a supported mode
   */
  const isSupportedMode = (mode: string): mode is SupportedMode => {
    return SUPPORTED_MODES.includes(mode as SupportedMode)
  }

  /**
   * Removes template, source, and mode parameters from URL
   */
  const cleanupUrlParams = () => {
    const newQuery = { ...route.query }
    delete newQuery.template
    delete newQuery.source
    delete newQuery.mode
    void router.replace({ query: newQuery })
  }

  /**
   * Loads template from URL query parameters if present
   * Handles errors internally and shows appropriate user feedback
   */
  const loadTemplateFromUrl = async () => {
    const templateParam = route.query.template

    if (!templateParam || typeof templateParam !== 'string') {
      return
    }

    if (!isValidParameter(templateParam)) {
      console.warn(
        `[useTemplateUrlLoader] Invalid template parameter format: ${templateParam}`
      )
      return
    }

    const sourceParam = (route.query.source as string | undefined) || 'default'

    if (!isValidParameter(sourceParam)) {
      console.warn(
        `[useTemplateUrlLoader] Invalid source parameter format: ${sourceParam}`
      )
      return
    }

    const modeParam = route.query.mode as string | undefined

    if (
      modeParam &&
      (typeof modeParam !== 'string' || !isValidParameter(modeParam))
    ) {
      console.warn(
        `[useTemplateUrlLoader] Invalid mode parameter format: ${modeParam}`
      )
      return
    }

    if (modeParam && !isSupportedMode(modeParam)) {
      console.warn(
        `[useTemplateUrlLoader] Unsupported mode parameter: ${modeParam}. Supported modes: ${SUPPORTED_MODES.join(', ')}`
      )
    }

    try {
      await templateWorkflows.loadTemplates()

      const success = await templateWorkflows.loadWorkflowTemplate(
        templateParam,
        sourceParam
      )

      if (!success) {
        toast.add({
          severity: 'error',
          summary: t('g.error'),
          detail: t('templateWorkflows.error.templateNotFound', {
            templateName: templateParam
          }),
          life: 3000
        })
      } else if (modeParam === 'linear') {
        // Set linear mode after successful template load
        canvasStore.linearMode = true
      }
    } catch (error) {
      console.error(
        '[useTemplateUrlLoader] Failed to load template from URL:',
        error
      )
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('g.errorLoadingTemplate'),
        life: 3000
      })
    } finally {
      cleanupUrlParams()
      clearPreservedQuery(TEMPLATE_NAMESPACE)
    }
  }

  return {
    loadTemplateFromUrl
  }
}
