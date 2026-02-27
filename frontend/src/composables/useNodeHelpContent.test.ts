import { flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import { useNodeHelpContent } from '@/composables/useNodeHelpContent'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

function createMockNode(
  overrides: Partial<ComfyNodeDefImpl>
): ComfyNodeDefImpl {
  return {
    name: 'TestNode',
    display_name: 'Test Node',
    description: 'A test node',
    category: 'test',
    python_module: 'comfy.test_node',
    inputs: {},
    outputs: [],
    deprecated: false,
    experimental: false,
    output_node: false,
    api_node: false,
    ...overrides
  } as ComfyNodeDefImpl
}

vi.mock('@/scripts/api', () => ({
  api: {
    fileURL: vi.fn((url) => url)
  }
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    locale: ref('en')
  })
}))

vi.mock('@/types/nodeSource', () => ({
  NodeSourceType: {
    Core: 'core',
    CustomNodes: 'custom_nodes'
  },
  getNodeSource: vi.fn((pythonModule) => {
    if (pythonModule?.startsWith('custom_nodes.')) {
      return { type: 'custom_nodes' }
    }
    return { type: 'core' }
  })
}))

describe('useNodeHelpContent', () => {
  const mockCoreNode = createMockNode({
    name: 'TestNode',
    display_name: 'Test Node',
    description: 'A test node',
    python_module: 'comfy.test_node'
  })

  const mockCustomNode = createMockNode({
    name: 'CustomNode',
    display_name: 'Custom Node',
    description: 'A custom node',
    python_module: 'custom_nodes.test_module.custom@1.0.0'
  })

  const mockFetch = vi.fn()

  beforeEach(() => {
    mockFetch.mockReset()
    vi.stubGlobal('fetch', mockFetch)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('should generate correct baseUrl for core nodes', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '# Test'
    })

    const { baseUrl } = useNodeHelpContent(nodeRef)
    await nextTick()

    expect(baseUrl.value).toBe(`/docs/${mockCoreNode.name}/`)
  })

  it('should generate correct baseUrl for custom nodes', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '# Test'
    })

    const { baseUrl } = useNodeHelpContent(nodeRef)
    await nextTick()

    expect(baseUrl.value).toBe('/extensions/test_module/docs/')
  })

  it('should render markdown content correctly', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '# Test Help\nThis is test help content'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain('This is test help content')
  })

  it('should handle fetch errors and fall back to description', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: false,
      statusText: 'Not Found'
    })

    const { error, renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(error.value).toBe('Not Found')
    expect(renderedHelpHtml.value).toContain(mockCoreNode.description)
  })

  it('should include alt attribute for images', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '![image](test.jpg)'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain('alt="image"')
  })

  it('should prefix relative video src in custom nodes', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '<video src="video.mp4"></video>'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      'src="/extensions/test_module/docs/video.mp4"'
    )
  })

  it('should prefix relative video src for core nodes with node-specific base URL', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '<video src="video.mp4"></video>'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video.mp4"`
    )
  })

  it('should handle loading state', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockImplementationOnce(() => new Promise(() => {})) // Never resolves

    const { isLoading } = useNodeHelpContent(nodeRef)
    await nextTick()

    expect(isLoading.value).toBe(true)
  })

  it('should try fallback URL for custom nodes', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        statusText: 'Not Found'
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => '# Fallback content'
      })

    useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(mockFetch).toHaveBeenCalledTimes(2)
    expect(mockFetch).toHaveBeenCalledWith(
      '/extensions/test_module/docs/CustomNode/en.md'
    )
    expect(mockFetch).toHaveBeenCalledWith(
      '/extensions/test_module/docs/CustomNode.md'
    )
  })

  it('should prefix relative source src in custom nodes', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        '<video><source src="video.mp4" type="video/mp4" /></video>'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      'src="/extensions/test_module/docs/video.mp4"'
    )
  })

  it('should prefix relative source src for core nodes with node-specific base URL', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        '<video><source src="video.webm" type="video/webm" /></video>'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video.webm"`
    )
  })

  it('should prefix relative img src in raw HTML for custom nodes', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '# Test\n<img src="image.png" alt="Test image">'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      'src="/extensions/test_module/docs/image.png"'
    )
    expect(renderedHelpHtml.value).toContain('alt="Test image"')
  })

  it('should prefix relative img src in raw HTML for core nodes', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '# Test\n<img src="image.png" alt="Test image">'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/image.png"`
    )
    expect(renderedHelpHtml.value).toContain('alt="Test image"')
  })

  it('should not prefix absolute img src in raw HTML', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => '<img src="/absolute/image.png" alt="Absolute">'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain('src="/absolute/image.png"')
    expect(renderedHelpHtml.value).toContain('alt="Absolute"')
  })

  it('should not prefix external img src in raw HTML', async () => {
    const nodeRef = ref(mockCustomNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () =>
        '<img src="https://example.com/image.png" alt="External">'
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    expect(renderedHelpHtml.value).toContain(
      'src="https://example.com/image.png"'
    )
    expect(renderedHelpHtml.value).toContain('alt="External"')
  })

  it('should handle various quote styles in media src attributes', async () => {
    const nodeRef = ref(mockCoreNode)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: async () => `# Media Test

Testing quote styles in properly formed HTML:

<video src="video1.mp4" controls></video>
<video src='video2.mp4' controls></video>
<img src="image1.png" alt="Double quotes">
<img src='image2.png' alt='Single quotes'>

<video controls>
  <source src="video3.mp4" type="video/mp4">
  <source src='video3.webm' type='video/webm'>
</video>

The MEDIA_SRC_REGEX handles both single and double quotes in img, video and source tags.`
    })

    const { renderedHelpHtml } = useNodeHelpContent(nodeRef)
    await flushPromises()

    // All media src attributes should be prefixed correctly
    // Note: marked normalizes quotes to double quotes in output
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video1.mp4"`
    )
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video2.mp4"`
    )
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/image1.png"`
    )
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/image2.png"`
    )
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video3.mp4"`
    )
    expect(renderedHelpHtml.value).toContain(
      `src="/docs/${mockCoreNode.name}/video3.webm"`
    )
  })

  it('should ignore stale requests when node changes', async () => {
    const nodeRef = ref(mockCoreNode)
    let resolveFirst: (value: unknown) => void
    const firstRequest = new Promise((resolve) => {
      resolveFirst = resolve
    })

    mockFetch
      .mockImplementationOnce(() => firstRequest)
      .mockResolvedValueOnce({
        ok: true,
        text: async () => '# Second node content'
      })

    const { helpContent } = useNodeHelpContent(nodeRef)
    await nextTick()

    // Change node before first request completes
    nodeRef.value = mockCustomNode
    await nextTick()
    await flushPromises()

    // Now resolve the first (stale) request
    resolveFirst!({
      ok: true,
      text: async () => '# First node content'
    })
    await flushPromises()

    // Should have second node's content, not first
    expect(helpContent.value).toBe('# Second node content')
  })
})
