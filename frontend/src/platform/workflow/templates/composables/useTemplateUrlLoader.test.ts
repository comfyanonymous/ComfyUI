import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTemplateUrlLoader } from '@/platform/workflow/templates/composables/useTemplateUrlLoader'

/**
 * Unit tests for useTemplateUrlLoader composable
 *
 * Tests the behavior of loading templates via URL query parameters:
 * - ?template=flux_simple loads the template
 * - ?template=flux_simple&source=custom loads from custom source
 * - ?template=flux_simple&mode=linear loads template in linear mode
 * - Invalid template shows error toast
 * - Input validation for template, source, and mode parameters
 */

const preservedQueryMocks = vi.hoisted(() => ({
  clearPreservedQuery: vi.fn()
}))

// Mock vue-router
let mockQueryParams: Record<string, string | string[] | undefined> = {}
const mockRouterReplace = vi.fn()

vi.mock('vue-router', () => ({
  useRoute: vi.fn(() => ({
    query: mockQueryParams
  })),
  useRouter: vi.fn(() => ({
    replace: mockRouterReplace
  }))
}))

vi.mock(
  '@/platform/navigation/preservedQueryManager',
  () => preservedQueryMocks
)

// Mock template workflows composable
const mockLoadTemplates = vi.fn().mockResolvedValue(true)
const mockLoadWorkflowTemplate = vi.fn().mockResolvedValue(true)

vi.mock(
  '@/platform/workflow/templates/composables/useTemplateWorkflows',
  () => ({
    useTemplateWorkflows: () => ({
      loadTemplates: mockLoadTemplates,
      loadWorkflowTemplate: mockLoadWorkflowTemplate
    })
  })
)

// Mock toast
const mockToastAdd = vi.fn()
vi.mock('primevue/usetoast', () => ({
  useToast: () => ({
    add: mockToastAdd
  })
}))

// Mock i18n
vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: vi.fn((key: string, params?: unknown) => {
      if (key === 'g.error') return 'Error'
      if (key === 'templateWorkflows.error.templateNotFound') {
        return `Template "${(params as { templateName?: string })?.templateName}" not found`
      }
      if (key === 'g.errorLoadingTemplate') return 'Failed to load template'
      return key
    })
  })
}))

// Mock canvas store
const mockCanvasStore = {
  linearMode: false
}

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => mockCanvasStore
}))

describe('useTemplateUrlLoader', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockQueryParams = {}
    mockCanvasStore.linearMode = false
  })

  it('does not load template when no query param present', () => {
    mockQueryParams = {}

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    expect(mockLoadTemplates).not.toHaveBeenCalled()
    expect(mockLoadWorkflowTemplate).not.toHaveBeenCalled()
  })

  it('loads template when query param is present', async () => {
    mockQueryParams = { template: 'flux_simple' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadTemplates).toHaveBeenCalledTimes(1)
    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
    expect(preservedQueryMocks.clearPreservedQuery).toHaveBeenCalledTimes(1)
  })

  it('uses default source when source param is not provided', async () => {
    mockQueryParams = { template: 'flux_simple' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
  })

  it('uses custom source when source param is provided', async () => {
    mockQueryParams = { template: 'custom-template', source: 'custom-module' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'custom-template',
      'custom-module'
    )
  })

  it('shows error toast when template loading fails', async () => {
    mockQueryParams = { template: 'invalid-template' }
    mockLoadWorkflowTemplate.mockResolvedValueOnce(false)

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockToastAdd).toHaveBeenCalledWith({
      severity: 'error',
      summary: 'Error',
      detail: 'Template "invalid-template" not found',
      life: 3000
    })
  })

  it('handles array query params correctly', () => {
    // Vue Router can return string[] for duplicate params
    mockQueryParams = { template: ['first', 'second'] }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    // Should not load when param is an array
    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('rejects invalid template parameter with special characters', () => {
    // Test path traversal attempt
    mockQueryParams = { template: '../../../etc/passwd' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    // Should not load invalid template
    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('rejects invalid template parameter with slash', () => {
    mockQueryParams = { template: 'path/to/template' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    // Should not load invalid template
    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('accepts valid template parameter formats', async () => {
    const validTemplates = [
      'flux_simple',
      'flux-kontext-dev',
      'template123',
      'My_Template-2',
      'templates-1_click_multiple_scene_angles-v1.0' // template with version number containing dot
    ]

    for (const template of validTemplates) {
      vi.clearAllMocks()
      mockQueryParams = { template }

      const { loadTemplateFromUrl } = useTemplateUrlLoader()
      await loadTemplateFromUrl()

      expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(template, 'default')
    }
  })

  it('rejects invalid source parameter with special characters', () => {
    mockQueryParams = { template: 'flux_simple', source: '../malicious' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    // Should not load with invalid source
    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('accepts valid source parameter formats', async () => {
    const validSources = ['default', 'custom-module', 'my_source', 'source123']

    for (const source of validSources) {
      vi.clearAllMocks()
      mockQueryParams = { template: 'flux_simple', source }

      const { loadTemplateFromUrl } = useTemplateUrlLoader()
      await loadTemplateFromUrl()

      expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
        'flux_simple',
        source
      )
    }
  })

  it('shows error toast when exception is thrown', async () => {
    mockQueryParams = { template: 'flux_simple' }
    mockLoadTemplates.mockRejectedValueOnce(new Error('Network error'))

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockToastAdd).toHaveBeenCalledWith({
      severity: 'error',
      summary: 'Error',
      detail: 'Failed to load template',
      life: 3000
    })
  })

  it('removes template params from URL after successful load', async () => {
    mockQueryParams = {
      template: 'flux_simple',
      source: 'custom',
      mode: 'linear',
      other: 'param'
    }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockRouterReplace).toHaveBeenCalledWith({
      query: { other: 'param' }
    })
  })

  it('removes template params from URL even on error', async () => {
    mockQueryParams = { template: 'invalid', source: 'custom', other: 'param' }
    mockLoadWorkflowTemplate.mockResolvedValueOnce(false)

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockRouterReplace).toHaveBeenCalledWith({
      query: { other: 'param' }
    })
  })

  it('removes template params from URL even on exception', async () => {
    mockQueryParams = { template: 'flux_simple', other: 'param' }
    mockLoadTemplates.mockRejectedValueOnce(new Error('Network error'))

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockRouterReplace).toHaveBeenCalledWith({
      query: { other: 'param' }
    })
  })

  it('sets linear mode when mode=linear and template loads successfully', async () => {
    mockQueryParams = { template: 'flux_simple', mode: 'linear' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
    expect(mockCanvasStore.linearMode).toBe(true)
  })

  it('does not set linear mode when template loading fails', async () => {
    mockQueryParams = { template: 'invalid-template', mode: 'linear' }
    mockLoadWorkflowTemplate.mockResolvedValueOnce(false)

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockCanvasStore.linearMode).toBe(false)
  })

  it('does not set linear mode when mode parameter is not linear', async () => {
    mockQueryParams = { template: 'flux_simple', mode: 'graph' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
    expect(mockCanvasStore.linearMode).toBe(false)
  })

  it('rejects invalid mode parameter with special characters', () => {
    mockQueryParams = { template: 'flux_simple', mode: '../malicious' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('handles array mode params correctly', () => {
    // Vue Router can return string[] for duplicate params
    mockQueryParams = {
      template: 'flux_simple',
      mode: ['linear', 'graph']
    }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    void loadTemplateFromUrl()

    // Should not load when mode param is an array
    expect(mockLoadTemplates).not.toHaveBeenCalled()
  })

  it('warns about unsupported mode values but continues loading', async () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    mockQueryParams = { template: 'flux_simple', mode: 'unsupported' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(consoleSpy).toHaveBeenCalledWith(
      '[useTemplateUrlLoader] Unsupported mode parameter: unsupported. Supported modes: linear'
    )
    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
    expect(mockCanvasStore.linearMode).toBe(false)

    consoleSpy.mockRestore()
  })

  it('accepts supported mode parameter: linear', async () => {
    mockQueryParams = { template: 'flux_simple', mode: 'linear' }

    const { loadTemplateFromUrl } = useTemplateUrlLoader()
    await loadTemplateFromUrl()

    expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
      'flux_simple',
      'default'
    )
    expect(mockCanvasStore.linearMode).toBe(true)
  })

  it('accepts valid format but warns about unsupported modes', async () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const unsupportedModes = ['graph', 'mode123', 'my_mode-2']

    for (const mode of unsupportedModes) {
      vi.clearAllMocks()
      consoleSpy.mockClear()
      mockCanvasStore.linearMode = false
      mockQueryParams = { template: 'flux_simple', mode }

      const { loadTemplateFromUrl } = useTemplateUrlLoader()
      await loadTemplateFromUrl()

      expect(consoleSpy).toHaveBeenCalledWith(
        `[useTemplateUrlLoader] Unsupported mode parameter: ${mode}. Supported modes: linear`
      )
      expect(mockLoadWorkflowTemplate).toHaveBeenCalledWith(
        'flux_simple',
        'default'
      )
      expect(mockCanvasStore.linearMode).toBe(false)
    }

    consoleSpy.mockRestore()
  })
})
