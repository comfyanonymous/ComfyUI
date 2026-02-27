import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useWorkflowTemplatesStore } from '@/platform/workflow/templates/repositories/workflowTemplatesStore'
import type {
  TemplateGroup,
  TemplateInfo,
  WorkflowTemplates
} from '@/platform/workflow/templates/types/template'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { useDialogStore } from '@/stores/dialogStore'

export function useTemplateWorkflows() {
  const { t } = useI18n()
  const workflowTemplatesStore = useWorkflowTemplatesStore()
  const dialogStore = useDialogStore()

  // State
  const selectedTemplate = ref<WorkflowTemplates | null>(null)
  const loadingTemplateId = ref<string | null>(null)

  // Computed
  const isTemplatesLoaded = computed(() => workflowTemplatesStore.isLoaded)
  const allTemplateGroups = computed<TemplateGroup[]>(
    () => workflowTemplatesStore.groupedTemplates
  )

  /**
   * Loads all template workflows from the API
   */
  const loadTemplates = async () => {
    if (!workflowTemplatesStore.isLoaded) {
      await workflowTemplatesStore.loadWorkflowTemplates()
    }
    return workflowTemplatesStore.isLoaded
  }

  /**
   * Selects the first template category as default
   */
  const selectFirstTemplateCategory = () => {
    if (allTemplateGroups.value.length > 0) {
      const firstCategory = allTemplateGroups.value[0].modules[0]
      selectTemplateCategory(firstCategory)
    }
  }

  /**
   * Selects a template category
   */
  const selectTemplateCategory = (category: WorkflowTemplates | null) => {
    selectedTemplate.value = category
    return category !== null
  }

  /**
   * Gets template thumbnail URL
   */
  const getTemplateThumbnailUrl = (
    template: TemplateInfo,
    sourceModule: string,
    index = '1'
  ) => {
    const basePath =
      sourceModule === 'default'
        ? api.fileURL(`/templates/${template.name}`)
        : api.apiURL(`/workflow_templates/${sourceModule}/${template.name}`)

    const indexSuffix = sourceModule === 'default' && index ? `-${index}` : ''
    return `${basePath}${indexSuffix}.${template.mediaSubtype}`
  }

  /**
   * Gets formatted template title
   */
  const getTemplateTitle = (template: TemplateInfo, sourceModule: string) => {
    const fallback =
      template.title ?? template.name ?? `${sourceModule} Template`
    return sourceModule === 'default'
      ? (template.localizedTitle ?? fallback)
      : fallback
  }

  /**
   * Gets formatted template description
   */
  const getTemplateDescription = (template: TemplateInfo) => {
    return (
      (template.localizedDescription || template.description)
        ?.replace(/[-_]/g, ' ')
        .trim() ?? ''
    )
  }

  /**
   * Loads a workflow template
   */
  const loadWorkflowTemplate = async (id: string, sourceModule: string) => {
    if (!isTemplatesLoaded.value) return false

    loadingTemplateId.value = id
    let json

    try {
      // Handle "All" category as a special case
      if (sourceModule === 'all') {
        // Find "All" category in the ComfyUI Examples group
        const comfyExamplesGroup = allTemplateGroups.value.find(
          (g) =>
            g.label ===
            t('templateWorkflows.category.ComfyUI Examples', 'ComfyUI Examples')
        )
        const allCategory = comfyExamplesGroup?.modules.find(
          (m) => m.moduleName === 'all'
        )
        const template = allCategory?.templates.find((t) => t.name === id)

        if (!template || !template.sourceModule) return false

        // Use the stored source module for loading
        sourceModule = template.sourceModule
      }

      // Regular case for normal categories
      json = await fetchTemplateJson(id, sourceModule)

      const workflowName =
        sourceModule === 'default'
          ? t(`templateWorkflows.template.${id}`, id)
          : id

      if (isCloud) {
        useTelemetry()?.trackTemplate({
          workflow_name: id,
          template_source: sourceModule
        })
      }

      dialogStore.closeDialog()
      await app.loadGraphData(json, true, true, workflowName, {
        openSource: 'template'
      })

      return true
    } catch (error) {
      console.error('Error loading workflow template:', error)
      return false
    } finally {
      loadingTemplateId.value = null
    }
  }

  /**
   * Fetches template JSON from the appropriate endpoint
   */
  const fetchTemplateJson = async (id: string, sourceModule: string) => {
    if (sourceModule === 'default') {
      // Default templates provided by frontend are served on this separate endpoint
      return fetch(api.fileURL(`/templates/${id}.json`)).then((r) => r.json())
    } else {
      return fetch(
        api.apiURL(`/workflow_templates/${sourceModule}/${id}.json`)
      ).then((r) => r.json())
    }
  }

  return {
    // State
    selectedTemplate,
    loadingTemplateId,

    // Computed
    isTemplatesLoaded,
    allTemplateGroups,

    // Methods
    loadTemplates,
    selectFirstTemplateCategory,
    selectTemplateCategory,
    getTemplateThumbnailUrl,
    getTemplateTitle,
    getTemplateDescription,
    loadWorkflowTemplate
  }
}
