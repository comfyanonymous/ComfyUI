import { beforeEach, describe, expect, it, vi } from 'vitest'

const mockData = vi.hoisted(() => ({ isDesktop: false }))

vi.mock('@/platform/distribution/types', () => ({
  get isDesktop() {
    return mockData.isDesktop
  }
}))

// Mock the environment utilities
vi.mock('@/utils/envUtil', () => ({
  electronAPI: vi.fn()
}))

// Provide a minimal i18n instance for the composable
const i18n = vi.hoisted(() => ({
  global: {
    locale: {
      value: 'en'
    }
  }
}))
vi.mock('@/i18n', () => ({
  i18n
}))

// Import after mocking to get the mocked versions
import { useExternalLink } from '@/composables/useExternalLink'
import { electronAPI } from '@/utils/envUtil'

describe('useExternalLink', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset to default state
    i18n.global.locale.value = 'en'
    mockData.isDesktop = false
  })

  describe('staticUrls', () => {
    it('should provide common external URLs', () => {
      const { staticUrls } = useExternalLink()

      // Static URLs
      expect(staticUrls.discord).toBe('https://www.comfy.org/discord')
      expect(staticUrls.github).toBe(
        'https://github.com/comfyanonymous/ComfyUI'
      )
      expect(staticUrls.githubIssues).toBe(
        'https://github.com/comfyanonymous/ComfyUI/issues'
      )
      expect(staticUrls.githubFrontend).toBe(
        'https://github.com/Comfy-Org/ComfyUI_frontend'
      )
      expect(staticUrls.githubElectron).toBe(
        'https://github.com/Comfy-Org/electron'
      )
      expect(staticUrls.forum).toBe('https://forum.comfy.org/')
      expect(staticUrls.comfyOrg).toBe('https://www.comfy.org/')
    })
  })

  describe('buildDocsUrl', () => {
    it('should build basic docs URL without locale', () => {
      i18n.global.locale.value = 'en'
      const { buildDocsUrl } = useExternalLink()

      const url = buildDocsUrl('/changelog')
      expect(url).toBe('https://docs.comfy.org/changelog')
    })

    it('should build docs URL with Chinese (zh) locale when requested', () => {
      i18n.global.locale.value = 'zh'
      const { buildDocsUrl } = useExternalLink()

      const url = buildDocsUrl('/changelog', { includeLocale: true })
      expect(url).toBe('https://docs.comfy.org/zh-CN/changelog')
    })

    it('should build docs URL with Chinese (zh-TW) locale when requested', () => {
      i18n.global.locale.value = 'zh-TW'
      const { buildDocsUrl } = useExternalLink()

      const url = buildDocsUrl('/changelog', { includeLocale: true })
      expect(url).toBe('https://docs.comfy.org/zh-CN/changelog')
    })

    it('should not include locale for English when requested', () => {
      i18n.global.locale.value = 'en'
      const { buildDocsUrl } = useExternalLink()

      const url = buildDocsUrl('/changelog', { includeLocale: true })
      expect(url).toBe('https://docs.comfy.org/changelog')
    })

    it('should handle path without leading slash', () => {
      const { buildDocsUrl } = useExternalLink()

      const url = buildDocsUrl('changelog')
      expect(url).toBe('https://docs.comfy.org/changelog')
    })

    it('should add platform suffix when requested', () => {
      i18n.global.locale.value = 'en'
      mockData.isDesktop = true
      vi.mocked(electronAPI).mockReturnValue({
        getPlatform: () => 'darwin'
      } as ReturnType<typeof electronAPI>)

      const { buildDocsUrl } = useExternalLink()
      const url = buildDocsUrl('/installation/desktop', { platform: true })
      expect(url).toBe('https://docs.comfy.org/installation/desktop/macos')
    })

    it('should add platform suffix with trailing slash', () => {
      i18n.global.locale.value = 'en'
      mockData.isDesktop = true
      vi.mocked(electronAPI).mockReturnValue({
        getPlatform: () => 'win32'
      } as ReturnType<typeof electronAPI>)

      const { buildDocsUrl } = useExternalLink()
      const url = buildDocsUrl('/installation/desktop/', { platform: true })
      expect(url).toBe('https://docs.comfy.org/installation/desktop/windows')
    })

    it('should combine locale and platform', () => {
      i18n.global.locale.value = 'zh'
      mockData.isDesktop = true
      vi.mocked(electronAPI).mockReturnValue({
        getPlatform: () => 'darwin'
      } as ReturnType<typeof electronAPI>)

      const { buildDocsUrl } = useExternalLink()
      const url = buildDocsUrl('/installation/desktop', {
        includeLocale: true,
        platform: true
      })
      expect(url).toBe(
        'https://docs.comfy.org/zh-CN/installation/desktop/macos'
      )
    })

    it('should not add platform when not desktop', () => {
      i18n.global.locale.value = 'en'
      mockData.isDesktop = false

      const { buildDocsUrl } = useExternalLink()
      const url = buildDocsUrl('/installation/desktop', { platform: true })
      expect(url).toBe('https://docs.comfy.org/installation/desktop')
    })
  })
})
