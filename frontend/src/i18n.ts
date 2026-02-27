import { createI18n } from 'vue-i18n'

// ESLint cannot statically resolve dynamic imports with relative paths in template strings,
// but these are valid ES module imports that Vite processes correctly at build time.

// Import only English locale eagerly as the default/fallback
import enCommands from './locales/en/commands.json' with { type: 'json' }
import en from './locales/en/main.json' with { type: 'json' }
import enNodes from './locales/en/nodeDefs.json' with { type: 'json' }
import enSettings from './locales/en/settings.json' with { type: 'json' }

function buildLocale<
  M extends Record<string, unknown>,
  N extends Record<string, unknown>,
  C extends Record<string, unknown>,
  S extends Record<string, unknown>
>(main: M, nodes: N, commands: C, settings: S) {
  return {
    ...main,
    nodeDefs: nodes,
    commands: commands,
    settings: settings
  } as M & { nodeDefs: N; commands: C; settings: S }
}

// Locale loader map - dynamically import locales only when needed
const localeLoaders: Record<
  string,
  () => Promise<{ default: Record<string, unknown> }>
> = {
  ar: () => import('./locales/ar/main.json'),
  es: () => import('./locales/es/main.json'),
  fa: () => import('./locales/fa/main.json'),
  fr: () => import('./locales/fr/main.json'),
  ja: () => import('./locales/ja/main.json'),
  ko: () => import('./locales/ko/main.json'),
  ru: () => import('./locales/ru/main.json'),
  tr: () => import('./locales/tr/main.json'),
  zh: () => import('./locales/zh/main.json'),
  'zh-TW': () => import('./locales/zh-TW/main.json'),
  'pt-BR': () => import('./locales/pt-BR/main.json')
}

const nodeDefsLoaders: Record<
  string,
  () => Promise<{ default: Record<string, unknown> }>
> = {
  ar: () => import('./locales/ar/nodeDefs.json'),
  es: () => import('./locales/es/nodeDefs.json'),
  fa: () => import('./locales/fa/nodeDefs.json'),
  fr: () => import('./locales/fr/nodeDefs.json'),
  ja: () => import('./locales/ja/nodeDefs.json'),
  ko: () => import('./locales/ko/nodeDefs.json'),
  ru: () => import('./locales/ru/nodeDefs.json'),
  tr: () => import('./locales/tr/nodeDefs.json'),
  zh: () => import('./locales/zh/nodeDefs.json'),
  'zh-TW': () => import('./locales/zh-TW/nodeDefs.json'),
  'pt-BR': () => import('./locales/pt-BR/nodeDefs.json')
}

const commandsLoaders: Record<
  string,
  () => Promise<{ default: Record<string, unknown> }>
> = {
  ar: () => import('./locales/ar/commands.json'),
  es: () => import('./locales/es/commands.json'),
  fa: () => import('./locales/fa/commands.json'),
  fr: () => import('./locales/fr/commands.json'),
  ja: () => import('./locales/ja/commands.json'),
  ko: () => import('./locales/ko/commands.json'),
  ru: () => import('./locales/ru/commands.json'),
  tr: () => import('./locales/tr/commands.json'),
  zh: () => import('./locales/zh/commands.json'),
  'zh-TW': () => import('./locales/zh-TW/commands.json'),
  'pt-BR': () => import('./locales/pt-BR/commands.json')
}

const settingsLoaders: Record<
  string,
  () => Promise<{ default: Record<string, unknown> }>
> = {
  ar: () => import('./locales/ar/settings.json'),
  es: () => import('./locales/es/settings.json'),
  fa: () => import('./locales/fa/settings.json'),
  fr: () => import('./locales/fr/settings.json'),
  ja: () => import('./locales/ja/settings.json'),
  ko: () => import('./locales/ko/settings.json'),
  ru: () => import('./locales/ru/settings.json'),
  tr: () => import('./locales/tr/settings.json'),
  zh: () => import('./locales/zh/settings.json'),
  'zh-TW': () => import('./locales/zh-TW/settings.json'),
  'pt-BR': () => import('./locales/pt-BR/settings.json')
}

// Track which locales have been loaded
const loadedLocales = new Set<string>(['en'])

// Track locales currently being loaded to prevent race conditions
const loadingLocales = new Map<string, Promise<void>>()

// Store custom nodes i18n data for merging when locales are lazily loaded
const customNodesI18nData: Record<string, unknown> = {}

/**
 * Dynamically load a locale and its associated files (nodeDefs, commands, settings)
 */
export async function loadLocale(locale: string): Promise<void> {
  if (loadedLocales.has(locale)) {
    return
  }

  // If already loading, return the existing promise to prevent duplicate loads
  const existingLoad = loadingLocales.get(locale)
  if (existingLoad) {
    return existingLoad
  }

  const loader = localeLoaders[locale]
  const nodeDefsLoader = nodeDefsLoaders[locale]
  const commandsLoader = commandsLoaders[locale]
  const settingsLoader = settingsLoaders[locale]

  if (!loader || !nodeDefsLoader || !commandsLoader || !settingsLoader) {
    console.warn(`Locale "${locale}" is not supported`)
    return
  }

  // Create and track the loading promise
  const loadPromise = (async () => {
    try {
      const [main, nodes, commands, settings] = await Promise.all([
        loader(),
        nodeDefsLoader(),
        commandsLoader(),
        settingsLoader()
      ])

      const messages = buildLocale(
        main.default,
        nodes.default,
        commands.default,
        settings.default
      )

      i18n.global.setLocaleMessage(locale, messages as LocaleMessages)
      loadedLocales.add(locale)

      if (customNodesI18nData[locale]) {
        i18n.global.mergeLocaleMessage(locale, customNodesI18nData[locale])
      }
    } catch (error) {
      console.error(`Failed to load locale "${locale}":`, error)
      throw error
    } finally {
      // Clean up the loading promise once complete
      loadingLocales.delete(locale)
    }
  })()

  loadingLocales.set(locale, loadPromise)
  return loadPromise
}

/**
 * Stores the data for later use when locales are lazily loaded,
 * and immediately merges data for already-loaded locales.
 */
export function mergeCustomNodesI18n(i18nData: Record<string, unknown>): void {
  // Clear existing data and replace with new data
  for (const key of Object.keys(customNodesI18nData)) {
    delete customNodesI18nData[key]
  }
  Object.assign(customNodesI18nData, i18nData)

  for (const [locale, message] of Object.entries(i18nData)) {
    if (loadedLocales.has(locale)) {
      i18n.global.mergeLocaleMessage(locale, message)
    }
  }
}

// Only include English in the initial bundle
const messages = {
  en: buildLocale(en, enNodes, enCommands, enSettings)
}

// Type for locale messages - inferred from the English locale structure
type LocaleMessages = typeof messages.en

export const i18n = createI18n({
  // Must set `false`, as Vue I18n Legacy API is for Vue 2
  legacy: false,
  locale: navigator.language.split('-')[0] || 'en',
  fallbackLocale: 'en',
  escapeParameter: true,
  messages,
  // Ignore warnings for locale options as each option is in its own language.
  // e.g. "English", "中文", "Русский", "日本語", "한국어", "Français", "Español"
  missingWarn: /^(?!settings\.Comfy_Locale\.options\.).+/,
  fallbackWarn: /^(?!settings\.Comfy_Locale\.options\.).+/
})

/** Convenience shorthand: i18n.global */
export const { t, te, d } = i18n.global

/**
 * Safe translation function that returns the fallback message if the key is not found.
 *
 * @param key - The key to translate.
 * @param fallbackMessage - The fallback message to use if the key is not found.
 */
export function st(key: string, fallbackMessage: string) {
  // The normal defaultMsg overload fails in some cases for custom nodes
  return te(key) ? t(key) : fallbackMessage
}
