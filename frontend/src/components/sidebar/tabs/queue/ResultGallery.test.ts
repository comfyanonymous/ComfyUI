import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import Galleria from 'primevue/galleria'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, nextTick } from 'vue'

import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { ResultItemImpl } from '@/stores/queueStore'

import ResultGallery from './ResultGallery.vue'

type MockResultItem = Partial<ResultItemImpl> & {
  filename: string
  subfolder: string
  type: string
  nodeId: NodeId
  mediaType: string
  id?: string
  url?: string
  isImage?: boolean
  isVideo?: boolean
}

describe('ResultGallery', () => {
  // Mock ComfyImage and ResultVideo components
  const mockComfyImage = {
    name: 'ComfyImage',
    template: '<div class="mock-comfy-image" data-testid="comfy-image"></div>',
    props: ['src', 'contain', 'alt']
  }

  const mockResultVideo = {
    name: 'ResultVideo',
    template:
      '<div class="mock-result-video" data-testid="result-video"></div>',
    props: ['result']
  }

  // Sample gallery items - using mock instances with only required properties
  const mockGalleryItems: MockResultItem[] = [
    {
      filename: 'image1.jpg',
      subfolder: 'outputs',
      type: 'output',
      nodeId: '123' as NodeId,
      mediaType: 'images',
      isImage: true,
      isVideo: false,
      url: 'image1.jpg',
      id: '1'
    },
    {
      filename: 'image2.jpg',
      subfolder: 'outputs',
      type: 'output',
      nodeId: '456' as NodeId,
      mediaType: 'images',
      isImage: true,
      isVideo: false,
      url: 'image2.jpg',
      id: '2'
    }
  ]

  beforeEach(() => {
    const app = createApp({})
    app.use(PrimeVue)

    // Create mock elements for Galleria to find
    document.body.innerHTML = `
      <div id="app"></div>
    `
  })

  afterEach(() => {
    // Clean up any elements added to body
    document.body.innerHTML = ''
    vi.restoreAllMocks()
  })

  const mountGallery = (props = {}) => {
    return mount(ResultGallery, {
      global: {
        plugins: [PrimeVue],
        components: {
          Galleria,
          ComfyImage: mockComfyImage,
          ResultVideo: mockResultVideo
        },
        stubs: {
          teleport: true
        }
      },
      props: {
        allGalleryItems: mockGalleryItems as ResultItemImpl[],
        activeIndex: 0,
        ...props
      },
      attachTo: document.getElementById('app') || undefined
    })
  }

  it('renders Galleria component with correct props', async () => {
    const wrapper = mountGallery()

    await nextTick() // Wait for component to mount

    const galleria = wrapper.findComponent(Galleria)
    expect(galleria.exists()).toBe(true)
    expect(galleria.props('value')).toEqual(mockGalleryItems)
    expect(galleria.props('showIndicators')).toBe(false)
    expect(galleria.props('showItemNavigators')).toBe(true)
    expect(galleria.props('fullScreen')).toBe(true)
  })

  it('shows gallery when activeIndex changes from -1', async () => {
    const wrapper = mountGallery({ activeIndex: -1 })

    // Initially galleryVisible should be false
    type GalleryVM = typeof wrapper.vm & {
      galleryVisible: boolean
    }
    const vm = wrapper.vm as GalleryVM
    expect(vm.galleryVisible).toBe(false)

    // Change activeIndex
    await wrapper.setProps({ activeIndex: 0 })
    await nextTick()

    // galleryVisible should become true
    expect(vm.galleryVisible).toBe(true)
  })

  it('should render the component properly', () => {
    // This is a meta-test to confirm the component mounts properly
    const wrapper = mountGallery()

    // We can't directly test the compiled CSS, but we can verify the component renders
    expect(wrapper.exists()).toBe(true)

    // Verify that the Galleria component exists and is properly mounted
    const galleria = wrapper.findComponent(Galleria)
    expect(galleria.exists()).toBe(true)
  })

  it('ensures correct configuration for mobile viewport', async () => {
    // Mock window.matchMedia to simulate mobile viewport
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation((query) => ({
        matches: query.includes('max-width: 768px'),
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })

    const wrapper = mountGallery()
    await nextTick()

    // Verify mobile media query is working
    expect(window.matchMedia('(max-width: 768px)').matches).toBe(true)

    // Check if the component renders with Galleria
    const galleria = wrapper.findComponent(Galleria)
    expect(galleria.exists()).toBe(true)

    // Check that our PT props for positioning work correctly
    interface GalleriaPT {
      prevButton?: { style?: string }
      nextButton?: { style?: string }
    }
    const pt = galleria.props('pt') as GalleriaPT
    expect(pt?.prevButton?.style).toContain('position: fixed')
    expect(pt?.nextButton?.style).toContain('position: fixed')
  })

  // Additional tests for interaction could be added once we can reliably
  // test Galleria component in fullscreen mode
})
