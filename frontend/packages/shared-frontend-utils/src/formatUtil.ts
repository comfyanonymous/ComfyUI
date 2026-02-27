import { default as DOMPurify } from 'dompurify'
import type { operations } from '@comfyorg/registry-types'

export function formatCamelCase(str: string): string {
  // Check if the string is camel case
  const isCamelCase = /^([A-Z][a-z]*)+$/.test(str)

  if (!isCamelCase) {
    return str // Return original string if not camel case
  }

  // Split the string into words, keeping acronyms together
  const words = str.split(/(?=[A-Z][a-z])|\d+/)

  // Process each word
  const processedWords = words.map((word) => {
    // If the word is all uppercase and longer than one character, it's likely an acronym
    if (word.length > 1 && word === word.toUpperCase()) {
      return word // Keep acronyms as is
    }
    // For other words, ensure the first letter is capitalized
    return word.charAt(0).toUpperCase() + word.slice(1)
  })

  // Join the words with spaces
  return processedWords.join(' ')
}

export function appendJsonExt(path: string) {
  if (!path.toLowerCase().endsWith('.json')) {
    path += '.json'
  }
  return path
}

export function highlightQuery(
  text: string,
  query: string,
  sanitize: boolean = true
) {
  if (!query) return text
  if (sanitize) {
    text = DOMPurify.sanitize(text)
  }

  // Escape special regex characters in the query string
  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  const regex = new RegExp(`(${escapedQuery})`, 'gi')
  return text.replace(regex, '<span class="highlight">$1</span>')
}

export function formatNumberWithSuffix(
  num: number,
  {
    precision = 1,
    roundToInt = false
  }: { precision?: number; roundToInt?: boolean } = {}
): string {
  const suffixes = ['', 'k', 'm', 'b', 't']
  const absNum = Math.abs(num)

  if (absNum < 1000) {
    return roundToInt ? Math.round(num).toString() : num.toFixed(precision)
  }

  const exp = Math.min(Math.floor(Math.log10(absNum) / 3), suffixes.length - 1)
  const formattedNum = (num / Math.pow(1000, exp)).toFixed(precision)

  return `${formattedNum}${suffixes[exp]}`
}

export function formatSize(value?: number) {
  if (value === null || value === undefined) {
    return '-'
  }

  const bytes = value
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

/**
 * Formats a commit hash by truncating long (40-char) hashes to 7 chars.
 * Returns the original string if not a valid full commit hash.
 */
export function formatCommitHash(value: string): string {
  if (/^[a-f0-9]{40}$/i.test(value)) {
    return value.slice(0, 7)
  }
  return value
}

/**
 * Returns various filename components.
 * Example:
 * - fullFilename: 'file.txt'
 * - filename: 'file'
 * - suffix: 'txt'
 */
export function getFilenameDetails(fullFilename: string) {
  if (fullFilename.includes('.')) {
    return {
      filename: fullFilename.split('.').slice(0, -1).join('.'),
      suffix: fullFilename.split('.').pop() ?? null
    }
  } else {
    return { filename: fullFilename, suffix: null }
  }
}

/**
 * Returns various path components.
 * Example:
 * - path: 'dir/file.txt'
 * - directory: 'dir'
 * - fullFilename: 'file.txt'
 * - filename: 'file'
 * - suffix: 'txt'
 */
export function getPathDetails(path: string) {
  const directory = path.split('/').slice(0, -1).join('/')
  const fullFilename = path.split('/').pop() ?? path
  return { directory, fullFilename, ...getFilenameDetails(fullFilename) }
}

/**
 * Normalizes a string to be used as an i18n key.
 * Replaces dots with underscores.
 */
export function normalizeI18nKey(key: string) {
  return typeof key === 'string' ? key.replace(/\./g, '_') : ''
}

/**
 * Takes a dynamic prompt in the format {opt1|opt2|{optA|optB}|} and randomly replaces groups. Supports C style comments.
 * @param input The dynamic prompt to process
 * @returns
 */
export function processDynamicPrompt(input: string): string {
  /*
   * Strips C-style line and block comments from a string
   */
  function stripComments(str: string) {
    return str.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g, '')
  }

  let i = 0
  let result = ''
  input = stripComments(input)

  const handleEscape = () => {
    const nextChar = input[i++]
    return '\\' + nextChar
  }

  function parseChoiceBlock() {
    // Parse the content inside {}
    const options: string[] = []
    let choice = ''
    let depth = 0

    while (i < input.length) {
      const char = input[i++]

      if (char === '\\') {
        choice += handleEscape()
        continue
      } else if (char === '{') {
        depth++
      } else if (char === '}') {
        if (!depth) break
        depth--
      } else if (char === '|') {
        if (!depth) {
          options.push(choice)
          choice = ''
          continue
        }
      }
      choice += char
    }

    options.push(choice)

    const chosenOption = options[Math.floor(Math.random() * options.length)]
    return processDynamicPrompt(chosenOption)
  }

  while (i < input.length) {
    const char = input[i++]
    if (char === '\\') {
      result += handleEscape()
    } else if (char === '{') {
      result += parseChoiceBlock()
    } else {
      result += char
    }
  }

  return result.replace(/\\([{}|])/g, '$1')
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

/**
 * Parses a filepath into its filename and subfolder components.
 *
 * @example
 * parseFilePath('folder/file.txt')    // → { filename: 'file.txt', subfolder: 'folder' }
 * parseFilePath('/folder/file.txt')   // → { filename: 'file.txt', subfolder: 'folder' }
 * parseFilePath('file.txt')           // → { filename: 'file.txt', subfolder: '' }
 * parseFilePath('folder//file.txt')   // → { filename: 'file.txt', subfolder: 'folder' }
 *
 * @param filepath The filepath to parse
 * @returns Object containing filename and subfolder
 */
export function parseFilePath(filepath: string): {
  filename: string
  subfolder: string
} {
  if (!filepath?.trim()) return { filename: '', subfolder: '' }

  const normalizedPath = filepath
    .replace(/[\\/]+/g, '/') // Normalize path separators
    .replace(/^\//, '') // Remove leading slash
    .replace(/\/$/, '') // Remove trailing slash

  const lastSlashIndex = normalizedPath.lastIndexOf('/')

  if (lastSlashIndex === -1) {
    return {
      filename: normalizedPath,
      subfolder: ''
    }
  }

  return {
    filename: normalizedPath.slice(lastSlashIndex + 1),
    subfolder: normalizedPath.slice(0, lastSlashIndex)
  }
}

// Simple date formatter
const parts = {
  d: (d: Date) => d.getDate(),
  M: (d: Date) => d.getMonth() + 1,
  h: (d: Date) => d.getHours(),
  m: (d: Date) => d.getMinutes(),
  s: (d: Date) => d.getSeconds()
}
const format =
  Object.keys(parts)
    .map((k) => k + k + '?')
    .join('|') + '|yyy?y?'

export function formatDate(text: string, date: Date) {
  return text.replace(new RegExp(format, 'g'), (text: string): string => {
    if (text === 'yy') return (date.getFullYear() + '').substring(2)
    if (text === 'yyyy') return date.getFullYear().toString()
    if (text[0] in parts) {
      const p = parts[text[0] as keyof typeof parts](date)
      return (p + '').padStart(text.length, '0')
    }
    return text
  })
}

/**
 * Generate a cache key from parameters
 * Sorts the parameters to ensure consistent keys regardless of parameter order
 */
export const paramsToCacheKey = (params: unknown): string => {
  if (typeof params === 'string') return params
  if (typeof params === 'object' && params !== null)
    return Object.keys(params)
      .sort((a, b) => a.localeCompare(b))
      .map((key) => `${key}:${params[key as keyof typeof params]}`)
      .join('&')

  return String(params)
}

/**
 * Generates a RFC4122 compliant UUID v4 using the native crypto API when available
 * @returns A properly formatted UUID string
 */
export const generateUUID = (): string => {
  // Use native crypto.randomUUID() if available (modern browsers)
  if (
    typeof crypto !== 'undefined' &&
    typeof crypto.randomUUID === 'function'
  ) {
    return crypto.randomUUID()
  }

  // Fallback implementation for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Checks if a URL is a Civitai model URL
 * @example
 * isCivitaiModelUrl('https://civitai.com/api/download/models/1234567890') // true
 * isCivitaiModelUrl('https://civitai.com/api/v1/models/1234567890') // true
 * isCivitaiModelUrl('https://civitai.com/api/v1/models-versions/15342') // true
 * isCivitaiModelUrl('https://example.com/model.safetensors') // false
 */
export const isCivitaiModelUrl = (url: string): boolean => {
  if (!isValidUrl(url)) return false
  if (!url.includes('civitai.com')) return false

  const urlObj = new URL(url)
  const pathname = urlObj.pathname

  return (
    /^\/api\/download\/models\/(\d+)$/.test(pathname) ||
    /^\/api\/v1\/models\/(\d+)$/.test(pathname) ||
    /^\/api\/v1\/models-versions\/(\d+)$/.test(pathname)
  )
}

/**
 * Converts a Hugging Face download URL to a repository page URL
 * @param url The download URL to convert
 * @returns The repository page URL or the original URL if conversion fails
 * @example
 * downloadUrlToHfRepoUrl(
 *  'https://huggingface.co/bfl/FLUX.1/resolve/main/flux1-canny-dev.safetensors?download=true'
 * ) // https://huggingface.co/bfl/FLUX.1
 */
export const downloadUrlToHfRepoUrl = (url: string): string => {
  try {
    const urlObj = new URL(url)
    const pathname = urlObj.pathname

    // Use regex to match everything before /resolve/ or /blob/
    const regex = /^(.*?)(?:\/resolve\/|\/blob\/|$)/
    const repoPathMatch = regex.exec(pathname)

    // Extract the repository path and remove leading slash if present
    const repoPath = repoPathMatch?.[1]?.replace(/^\//, '') || ''

    return `https://huggingface.co/${repoPath}`
  } catch (error) {
    return url
  }
}

/**
 * Converts Metronome's integer amount back to a formatted currency string.
 * For USD, converts from cents to dollars.
 * For all other currencies (including custom pricing units), returns the amount as is.
 * This is specific to Metronome's API requirements.
 *
 * @param amount - The amount in Metronome's integer format (cents for USD, base units for others)
 * @param currency - The currency to convert
 * @returns The formatted amount in currency with 2 decimal places for USD
 * @example
 * formatMetronomeCurrency(123, 'usd') // returns "1.23" (cents to USD)
 * formatMetronomeCurrency(1000, 'jpy') // returns "1000" (yen)
 */
export function formatMetronomeCurrency(
  amount: number,
  currency: string
): string {
  if (currency === 'usd') {
    return (amount / 100).toFixed(2)
  }
  return amount.toString()
}

/**
 * Converts a USD amount to microdollars (1/1,000,000 of a dollar).
 * This conversion is commonly used in financial systems to avoid floating-point precision issues
 * by representing monetary values as integers.
 *
 * @param usd - The amount in US dollars to convert
 * @returns The amount in microdollars (multiplied by 1,000,000)
 * @example
 * usdToMicros(1.23) // returns 1230000
 */
export function usdToMicros(usd: number): number {
  return Math.round(usd * 1_000_000)
}

/**
 * Converts URLs in a string to HTML links.
 * @param text - The string to convert
 * @returns The string with URLs converted to HTML links
 * @example
 * linkifyHtml('Visit https://example.com for more info') // returns 'Visit <a href="https://example.com" target="_blank" rel="noopener noreferrer" class="text-primary-400 hover:underline">https://example.com</a> for more info'
 */
export function linkifyHtml(text: string): string {
  if (!text) return ''
  const urlRegex =
    /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#/%?=~_|!:,.;]*[-A-Z0-9+&@#/%?=~_|])|(\bwww\.[-A-Z0-9+&@#/%?=~_|!:,.;]*[-A-Z0-9+&@#/%?=~_|])/gi
  return text.replace(urlRegex, (_match, p1, _p2, p3) => {
    const url = p1 || p3
    const href = p3 ? `http://${url}` : url
    return `<a href="${href}" target="_blank" rel="noopener noreferrer" class="text-primary-400 hover:underline">${url}</a>`
  })
}

/**
 * Converts newline characters to HTML <br> tags.
 * @param text - The string to convert
 * @returns The string with newline characters converted to <br> tags
 * @example
 * nl2br('Hello\nWorld') // returns 'Hello<br />World'
 */
export function nl2br(text: string): string {
  if (!text) return ''
  return text.replace(/\n/g, '<br />')
}

/**
 * Converts a version string to an anchor-safe format by replacing dots with dashes.
 * @param version The version string (e.g., "1.0.0", "2.1.3-beta.1")
 * @returns The anchor-safe version string (e.g., "v1-0-0", "v2-1-3-beta-1")
 * @example
 * formatVersionAnchor("1.0.0") // returns "v1-0-0"
 * formatVersionAnchor("2.1.3-beta.1") // returns "v2-1-3-beta-1"
 */
export function formatVersionAnchor(version: string): string {
  return `v${version.replace(/\./g, '-')}`
}

/**
 * Supported locale types for the application (from OpenAPI schema)
 */
type SupportedLocale = NonNullable<
  operations['getReleaseNotes']['parameters']['query']['locale']
>

/**
 * Converts a string to a valid locale type with 'en' as default
 * @param locale - The locale string to validate and convert
 * @returns A valid SupportedLocale type, defaults to 'en' if invalid
 * @example
 * stringToLocale('fr') // returns 'fr'
 * stringToLocale('invalid') // returns 'en'
 * stringToLocale('') // returns 'en'
 */
export function stringToLocale(locale: string): SupportedLocale {
  const supportedLocales: SupportedLocale[] = [
    'en',
    'es',
    'fr',
    'ja',
    'ko',
    'ru',
    'zh'
  ]
  return supportedLocales.includes(locale as SupportedLocale)
    ? (locale as SupportedLocale)
    : 'en'
}

export function formatDuration(milliseconds: number): string {
  if (!milliseconds || milliseconds < 0) return '0s'

  const totalSeconds = Math.floor(milliseconds / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainingSeconds = Math.floor(totalSeconds % 60)

  const parts: string[] = []

  if (hours > 0) {
    parts.push(`${hours}h`)
  }
  if (minutes > 0) {
    parts.push(`${minutes}m`)
  }
  if (remainingSeconds > 0 || parts.length === 0) {
    parts.push(`${remainingSeconds}s`)
  }

  return parts.join(' ')
}

const IMAGE_EXTENSIONS = [
  'png',
  'jpg',
  'jpeg',
  'gif',
  'webp',
  'bmp',
  'avif',
  'tif',
  'tiff'
] as const
const VIDEO_EXTENSIONS = ['mp4', 'webm', 'mov', 'avi'] as const
const AUDIO_EXTENSIONS = ['mp3', 'wav', 'ogg', 'flac'] as const
const THREE_D_EXTENSIONS = ['obj', 'fbx', 'gltf', 'glb', 'usdz'] as const
const TEXT_EXTENSIONS = [
  'txt',
  'md',
  'markdown',
  'json',
  'csv',
  'yaml',
  'yml',
  'xml',
  'log'
] as const

const MEDIA_TYPES = ['image', 'video', 'audio', '3D', 'text', 'other'] as const
export type MediaType = (typeof MEDIA_TYPES)[number]

// Type guard helper for checking array membership
type ImageExtension = (typeof IMAGE_EXTENSIONS)[number]
type VideoExtension = (typeof VIDEO_EXTENSIONS)[number]
type AudioExtension = (typeof AUDIO_EXTENSIONS)[number]
type ThreeDExtension = (typeof THREE_D_EXTENSIONS)[number]
type TextExtension = (typeof TEXT_EXTENSIONS)[number]

/**
 * Truncates a filename while preserving the extension
 * @param filename The filename to truncate
 * @param maxLength Maximum length for the filename without extension
 * @returns Truncated filename with extension preserved
 */
export function truncateFilename(
  filename: string,
  maxLength: number = 20
): string {
  if (!filename || filename.length <= maxLength) {
    return filename
  }

  const lastDotIndex = filename.lastIndexOf('.')
  const nameWithoutExt =
    lastDotIndex > -1 ? filename.substring(0, lastDotIndex) : filename
  const extension = lastDotIndex > -1 ? filename.substring(lastDotIndex) : ''

  // If the name without extension is short enough, return as is
  if (nameWithoutExt.length <= maxLength) {
    return filename
  }

  // Calculate how to split the truncation
  const halfLength = Math.floor((maxLength - 3) / 2) // -3 for '...'
  const start = nameWithoutExt.substring(0, halfLength)
  const end = nameWithoutExt.substring(nameWithoutExt.length - halfLength)

  return `${start}...${end}${extension}`
}

/**
 * Determines the media type from a filename's extension (singular form)
 * @param filename The filename to analyze
 * @returns The media type: 'image', 'video', 'audio', '3D', 'text', or 'other'
 */
export function getMediaTypeFromFilename(
  filename: string | null | undefined
): MediaType {
  if (!filename) return 'other'
  const ext = filename.split('.').pop()?.toLowerCase()
  if (!ext) return 'other'

  // Type-safe array includes check using type assertion
  if (IMAGE_EXTENSIONS.includes(ext as ImageExtension)) return 'image'
  if (VIDEO_EXTENSIONS.includes(ext as VideoExtension)) return 'video'
  if (AUDIO_EXTENSIONS.includes(ext as AudioExtension)) return 'audio'
  if (THREE_D_EXTENSIONS.includes(ext as ThreeDExtension)) return '3D'
  if (TEXT_EXTENSIONS.includes(ext as TextExtension)) return 'text'

  return 'other'
}

export function isPreviewableMediaType(mediaType: MediaType): boolean {
  return (
    mediaType === 'image' ||
    mediaType === 'video' ||
    mediaType === 'audio' ||
    mediaType === '3D'
  )
}
