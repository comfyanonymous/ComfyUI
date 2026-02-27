import { beforeEach, describe, expect, it, vi } from 'vitest'
const { i18n, loadLocale, mergeCustomNodesI18n } = await import('./i18n')

// Mock the JSON imports before importing i18n module
vi.mock('./locales/en/main.json', () => ({ default: { welcome: 'Welcome' } }))
vi.mock('./locales/en/nodeDefs.json', () => ({
  default: { testNode: 'Test Node' }
}))
vi.mock('./locales/en/commands.json', () => ({
  default: { save: 'Save' }
}))
vi.mock('./locales/en/settings.json', () => ({
  default: { theme: 'Theme' }
}))

// Mock lazy-loaded locales
vi.mock('./locales/zh/main.json', () => ({ default: { welcome: '欢迎' } }))
vi.mock('./locales/zh/nodeDefs.json', () => ({
  default: { testNode: '测试节点' }
}))
vi.mock('./locales/zh/commands.json', () => ({ default: { save: '保存' } }))
vi.mock('./locales/zh/settings.json', () => ({ default: { theme: '主题' } }))

describe('i18n', () => {
  beforeEach(async () => {
    vi.resetModules()
  })

  describe('mergeCustomNodesI18n', () => {
    it('should immediately merge data for already loaded locales (en)', async () => {
      // English is pre-loaded, so merge should work immediately
      mergeCustomNodesI18n({
        en: {
          customNode: {
            title: 'Custom Node Title'
          }
        }
      })

      // Verify the custom node data was merged
      const messages = i18n.global.getLocaleMessage('en') as Record<
        string,
        unknown
      >
      expect(messages.customNode).toEqual({ title: 'Custom Node Title' })
    })

    it('should store data for not-yet-loaded locales', async () => {
      const { i18n, mergeCustomNodesI18n } = await import('./i18n')

      // Chinese is not pre-loaded, data should be stored but not merged yet
      mergeCustomNodesI18n({
        zh: {
          customNode: {
            title: '自定义节点标题'
          }
        }
      })

      // zh locale should not exist yet (not loaded)
      const zhMessages = i18n.global.getLocaleMessage('zh') as Record<
        string,
        unknown
      >
      // Either empty or doesn't have our custom data merged directly
      // (since zh wasn't loaded yet, mergeLocaleMessage on non-existent locale
      // may create an empty locale or do nothing useful)
      expect(zhMessages.customNode).toBeUndefined()
    })

    it('should merge stored data when locale is lazily loaded', async () => {
      // First, store custom nodes i18n data (before locale is loaded)
      mergeCustomNodesI18n({
        zh: {
          customNode: {
            title: '自定义节点标题'
          }
        }
      })

      await loadLocale('zh')

      // Verify both the base locale data and custom node data are present
      const zhMessages = i18n.global.getLocaleMessage('zh') as Record<
        string,
        unknown
      >
      expect(zhMessages.welcome).toBe('欢迎')
      expect(zhMessages.customNode).toEqual({ title: '自定义节点标题' })
    })

    it('should preserve custom node data when locale is loaded after merge', async () => {
      // Simulate the real scenario:
      // 1. Custom nodes i18n is loaded first
      mergeCustomNodesI18n({
        zh: {
          customNode: {
            title: '自定义节点标题'
          },
          settingsCategories: {
            Hotkeys: '快捷键'
          }
        }
      })

      // 2. Then locale is lazily loaded (this would previously overwrite custom data)
      await loadLocale('zh')

      // 3. Verify custom node data is still present
      const zhMessages = i18n.global.getLocaleMessage('zh') as Record<
        string,
        unknown
      >
      expect(zhMessages.customNode).toEqual({ title: '自定义节点标题' })
      expect(zhMessages.settingsCategories).toEqual({ Hotkeys: '快捷键' })

      // 4. Also verify base locale data is present
      expect(zhMessages.welcome).toBe('欢迎')
      expect(zhMessages.nodeDefs).toEqual({ testNode: '测试节点' })
    })

    it('should handle multiple locales in custom nodes i18n data', async () => {
      // Merge data for multiple locales
      mergeCustomNodesI18n({
        en: {
          customPlugin: { name: 'Easy Use' }
        },
        zh: {
          customPlugin: { name: '简单使用' }
        }
      })

      // English should be merged immediately (pre-loaded)
      const enMessages = i18n.global.getLocaleMessage('en') as Record<
        string,
        unknown
      >
      expect(enMessages.customPlugin).toEqual({ name: 'Easy Use' })

      await loadLocale('zh')
      const zhMessages = i18n.global.getLocaleMessage('zh') as Record<
        string,
        unknown
      >
      expect(zhMessages.customPlugin).toEqual({ name: '简单使用' })
    })

    it('should handle calling mergeCustomNodesI18n multiple times', async () => {
      // Use fresh module instance to ensure clean state
      vi.resetModules()
      const { i18n, loadLocale, mergeCustomNodesI18n } = await import('./i18n')

      mergeCustomNodesI18n({
        zh: { plugin1: { name: '插件1' } }
      })

      mergeCustomNodesI18n({
        zh: { plugin2: { name: '插件2' } }
      })

      await loadLocale('zh')

      const zhMessages = i18n.global.getLocaleMessage('zh') as Record<
        string,
        unknown
      >
      // Only the second call's data should be present
      expect(zhMessages.plugin2).toEqual({ name: '插件2' })
      // First call's data is overwritten
      expect(zhMessages.plugin1).toBeUndefined()
    })
  })

  describe('loadLocale', () => {
    it('should not reload already loaded locale', async () => {
      await loadLocale('zh')
      await loadLocale('zh')

      // Should complete without error (second call returns early)
    })

    it('should warn for unsupported locale', async () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      await loadLocale('unsupported-locale')

      expect(consoleSpy).toHaveBeenCalledWith(
        'Locale "unsupported-locale" is not supported'
      )
      consoleSpy.mockRestore()
    })

    it('should handle concurrent load requests for same locale', async () => {
      // Start multiple loads concurrently
      const promises = [loadLocale('zh'), loadLocale('zh'), loadLocale('zh')]

      await Promise.all(promises)
    })
  })
})
