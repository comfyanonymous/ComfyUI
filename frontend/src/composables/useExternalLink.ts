import { computed } from 'vue'

import { isDesktop } from '@/platform/distribution/types'
import { electronAPI } from '@/utils/envUtil'
import { i18n } from '@/i18n'

/**
 * Composable for building docs.comfy.org URLs with automatic locale and platform detection
 *
 * @example
 * ```ts
 * const { buildDocsUrl } = useExternalLink()
 *
 * // Simple usage
 * const changelogUrl = buildDocsUrl('/changelog', { includeLocale: true })
 * // => 'https://docs.comfy.org/zh-CN/changelog' (if Chinese)
 *
 * // With platform detection
 * const desktopUrl = buildDocsUrl('/installation/desktop', {
 *   includeLocale: true,
 *   platform: true
 * })
 * // => 'https://docs.comfy.org/zh-CN/installation/desktop/macos' (if Chinese + macOS)
 * ```
 */
export function useExternalLink() {
  const locale = computed(() => String(i18n.global.locale.value))

  const isChinese = computed(() => {
    return locale.value === 'zh' || locale.value === 'zh-TW'
  })

  const platform = computed(() => {
    if (!isDesktop) {
      return null
    }

    const electronPlatform = electronAPI().getPlatform()
    return electronPlatform === 'darwin' ? 'macos' : 'windows'
  })

  /**
   * Build a docs.comfy.org URL with optional locale and platform
   *
   * @param path - The path after the domain (e.g., '/installation/desktop')
   * @param options - Options for building the URL
   * @param options.includeLocale - Whether to include locale prefix (default: false)
   * @param options.platform - Whether to include platform suffix (default: false)
   * @returns The complete docs URL
   *
   * @example
   * ```ts
   * buildDocsUrl('/changelog') // => 'https://docs.comfy.org/changelog'
   * buildDocsUrl('/changelog', { includeLocale: true }) // => 'https://docs.comfy.org/zh-CN/changelog' (if Chinese)
   * buildDocsUrl('/installation/desktop', { includeLocale: true, platform: true })
   * // => 'https://docs.comfy.org/zh-CN/installation/desktop/macos' (if Chinese + macOS)
   * ```
   */
  const buildDocsUrl = (
    path: string,
    options: {
      includeLocale?: boolean
      platform?: boolean
    } = {}
  ): string => {
    const { includeLocale = false, platform: includePlatform = false } = options

    let url = 'https://docs.comfy.org'

    if (includeLocale && isChinese.value) {
      url += '/zh-CN'
    }

    const normalizedPath = path.startsWith('/') ? path : `/${path}`
    url += normalizedPath

    if (includePlatform && platform.value) {
      url = url.endsWith('/') ? url : `${url}/`
      url += platform.value
    }

    return url
  }

  const staticUrls = {
    // Static external URLs
    discord: 'https://www.comfy.org/discord',
    github: 'https://github.com/comfyanonymous/ComfyUI',
    githubIssues: 'https://github.com/comfyanonymous/ComfyUI/issues',
    githubFrontend: 'https://github.com/Comfy-Org/ComfyUI_frontend',
    githubElectron: 'https://github.com/Comfy-Org/electron',
    forum: 'https://forum.comfy.org/',
    comfyOrg: 'https://www.comfy.org/'
  }

  /** Common doc paths for use with buildDocsUrl */
  const docsPaths = {
    partnerNodesPricing: '/tutorials/partner-nodes/pricing'
  }

  return {
    buildDocsUrl,
    staticUrls,
    docsPaths
  }
}
