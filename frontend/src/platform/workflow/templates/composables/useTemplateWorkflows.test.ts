import { flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTemplateWorkflows } from '@/platform/workflow/templates/composables/useTemplateWorkflows'
import { useWorkflowTemplatesStore } from '@/platform/workflow/templates/repositories/workflowTemplatesStore'

// Mock the store
vi.mock(
  '@/platform/workflow/templates/repositories/workflowTemplatesStore',
  () => ({
    useWorkflowTemplatesStore: vi.fn()
  })
)

// Mock the API
vi.mock('@/scripts/api', () => ({
  api: {
    fileURL: vi.fn((path) => `mock-file-url${path}`),
    apiURL: vi.fn((path) => `mock-api-url${path}`)
  }
}))

// Mock the app
vi.mock('@/scripts/app', () => ({
  app: {
    loadGraphData: vi.fn()
  }
}))

// Mock Vue I18n
vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: vi.fn((key, fallback) => fallback || key)
  }),
  createI18n: () => ({
    global: {
      t: (key: string) => key
    }
  })
}))

// Mock the dialog store
vi.mock('@/stores/dialogStore', () => ({
  useDialogStore: vi.fn(() => ({
    closeDialog: vi.fn()
  }))
}))

// Mock fetch
global.fetch = vi.fn()

type MockWorkflowTemplatesStore = ReturnType<typeof useWorkflowTemplatesStore>

describe('useTemplateWorkflows', () => {
  let mockWorkflowTemplatesStore: MockWorkflowTemplatesStore

  beforeEach(() => {
    mockWorkflowTemplatesStore = {
      isLoaded: false,
      loadWorkflowTemplates: vi.fn().mockResolvedValue(true),
      groupedTemplates: [
        {
          label: 'ComfyUI Examples',
          modules: [
            {
              moduleName: 'all',
              title: 'All',
              localizedTitle: 'All Templates',
              templates: [
                {
                  name: 'template1',
                  mediaType: 'image',
                  mediaSubtype: 'jpg',
                  sourceModule: 'default',
                  localizedTitle: 'Template 1',
                  description: 'Template 1 description'
                },
                {
                  name: 'template2',
                  mediaType: 'image',
                  mediaSubtype: 'jpg',
                  sourceModule: 'custom-module',
                  description: 'A custom template'
                }
              ]
            },
            {
              moduleName: 'default',
              title: 'Default',
              localizedTitle: 'Default Templates',
              templates: [
                {
                  name: 'template1',
                  mediaType: 'image',
                  mediaSubtype: 'jpg',
                  localizedTitle: 'Template 1',
                  localizedDescription: 'A default template',
                  description: 'Template 1 description'
                }
              ]
            }
          ]
        }
      ]
    } as Partial<MockWorkflowTemplatesStore> as MockWorkflowTemplatesStore

    vi.mocked(useWorkflowTemplatesStore).mockReturnValue(
      mockWorkflowTemplatesStore
    )

    // Mock fetch response
    vi.mocked(fetch).mockResolvedValue({
      json: vi.fn().mockResolvedValue({ workflow: 'data' })
    } as Partial<Response> as Response)
  })

  it('should load templates from store', async () => {
    const { loadTemplates, isTemplatesLoaded } = useTemplateWorkflows()

    expect(isTemplatesLoaded.value).toBe(false)

    await loadTemplates()

    expect(mockWorkflowTemplatesStore.loadWorkflowTemplates).toHaveBeenCalled()
  })

  it('should select the first template category', () => {
    const { selectFirstTemplateCategory, selectedTemplate } =
      useTemplateWorkflows()

    selectFirstTemplateCategory()

    expect(selectedTemplate.value).toEqual(
      mockWorkflowTemplatesStore.groupedTemplates[0].modules[0]
    )
  })

  it('should select a template category', () => {
    const { selectTemplateCategory, selectedTemplate } = useTemplateWorkflows()
    const category = mockWorkflowTemplatesStore.groupedTemplates[0].modules[1] // Default category

    const result = selectTemplateCategory(category)

    expect(result).toBe(true)
    expect(selectedTemplate.value).toEqual(category)
  })

  it('should format template thumbnails correctly for default templates', () => {
    const { getTemplateThumbnailUrl } = useTemplateWorkflows()
    const template = {
      name: 'test-template',
      mediaSubtype: 'jpg',
      mediaType: 'image',
      description: 'Test template'
    }

    const url = getTemplateThumbnailUrl(template, 'default', '1')

    expect(url).toBe('mock-file-url/templates/test-template-1.jpg')
  })

  it('should format template thumbnails correctly for custom templates', () => {
    const { getTemplateThumbnailUrl } = useTemplateWorkflows()
    const template = {
      name: 'test-template',
      mediaSubtype: 'jpg',
      mediaType: 'image',
      description: 'Test template'
    }

    const url = getTemplateThumbnailUrl(template, 'custom-module')

    expect(url).toBe(
      'mock-api-url/workflow_templates/custom-module/test-template.jpg'
    )
  })

  it('should format template titles correctly', () => {
    const { getTemplateTitle } = useTemplateWorkflows()

    // Default template with localized title
    const titleWithLocalized = getTemplateTitle(
      {
        name: 'test',
        localizedTitle: 'Localized Title',
        mediaType: 'image',
        mediaSubtype: 'jpg',
        description: 'Test'
      },
      'default'
    )
    expect(titleWithLocalized).toBe('Localized Title')

    // Default template without localized title
    const titleWithFallback = getTemplateTitle(
      {
        name: 'test',
        title: 'Title',
        mediaType: 'image',
        mediaSubtype: 'jpg',
        description: 'Test'
      },
      'default'
    )
    expect(titleWithFallback).toBe('Title')

    // Custom template
    const customTitle = getTemplateTitle(
      {
        name: 'test-template',
        title: 'Custom Title',
        mediaType: 'image',
        mediaSubtype: 'jpg',
        description: 'Test'
      },
      'custom-module'
    )
    expect(customTitle).toBe('Custom Title')

    // Fallback to name
    const nameOnly = getTemplateTitle(
      {
        name: 'name-only',
        mediaType: 'image',
        mediaSubtype: 'jpg',
        description: 'Test'
      },
      'custom-module'
    )
    expect(nameOnly).toBe('name-only')
  })

  it('should format template descriptions correctly', () => {
    const { getTemplateDescription } = useTemplateWorkflows()

    // Default template with localized description
    const descWithLocalized = getTemplateDescription({
      name: 'test',
      localizedDescription: 'Localized Description',
      mediaType: 'image',
      mediaSubtype: 'jpg',
      description: 'Test'
    })
    expect(descWithLocalized).toBe('Localized Description')

    // Custom template with description
    const customDesc = getTemplateDescription({
      name: 'test',
      description: 'custom-template_description',
      mediaType: 'image',
      mediaSubtype: 'jpg'
    })
    expect(customDesc).toBe('custom template description')
  })

  it('should load a template from the "All" category', async () => {
    const { loadWorkflowTemplate, loadingTemplateId } = useTemplateWorkflows()

    // Set the store as loaded
    mockWorkflowTemplatesStore.isLoaded = true

    // Load a template from the "All" category
    const result = await loadWorkflowTemplate('template1', 'all')
    await flushPromises()

    expect(result).toBe(true)
    expect(fetch).toHaveBeenCalledWith('mock-file-url/templates/template1.json')
    expect(loadingTemplateId.value).toBe(null) // Should reset after loading
  })

  it('should load a template from a regular category', async () => {
    const { loadWorkflowTemplate } = useTemplateWorkflows()

    // Set the store as loaded
    mockWorkflowTemplatesStore.isLoaded = true

    // Load a template from the default category
    const result = await loadWorkflowTemplate('template1', 'default')
    await flushPromises()

    expect(result).toBe(true)
    expect(fetch).toHaveBeenCalledWith('mock-file-url/templates/template1.json')
  })

  it('should handle errors when loading templates', async () => {
    const { loadWorkflowTemplate, loadingTemplateId } = useTemplateWorkflows()

    // Set the store as loaded
    mockWorkflowTemplatesStore.isLoaded = true

    // Mock fetch to throw an error
    vi.mocked(fetch).mockRejectedValueOnce(new Error('Failed to fetch'))

    // Spy on console.error
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    // Load a template that will fail
    const result = await loadWorkflowTemplate('error-template', 'default')

    expect(result).toBe(false)
    expect(consoleSpy).toHaveBeenCalled()
    expect(loadingTemplateId.value).toBe(null) // Should reset even after error

    // Restore console.error
    consoleSpy.mockRestore()
  })
})
