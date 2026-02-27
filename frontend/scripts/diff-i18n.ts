import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync
} from 'fs'
import { dirname, join } from 'path'
import type { LocaleData } from './i18n-types'

// Ensure directories exist
function ensureDir(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true })
  }
}

// Read JSON file
function readJsonFile(path: string) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'))
  } catch {
    return {}
  }
}

// Get all JSON files recursively
function getAllJsonFiles(dir: string): string[] {
  const files: string[] = []
  const items = readdirSync(dir, { withFileTypes: true })

  for (const item of items) {
    const path = join(dir, item.name)
    if (item.isDirectory()) {
      files.push(...getAllJsonFiles(path))
    } else if (item.name.endsWith('.json')) {
      files.push(path)
    }
  }
  return files
}

// Find additions in new object compared to base
function findAdditions(base: LocaleData, updated: LocaleData): LocaleData {
  const additions: LocaleData = {}

  for (const key in updated) {
    if (!(key in base)) {
      additions[key] = updated[key]
    } else if (
      typeof updated[key] === 'object' &&
      !Array.isArray(updated[key]) &&
      typeof base[key] === 'object' &&
      !Array.isArray(base[key])
    ) {
      const nestedAdditions = findAdditions(base[key], updated[key])
      if (Object.keys(nestedAdditions).length > 0) {
        additions[key] = nestedAdditions
      }
    }
  }

  return additions
}

// Capture command
function capture(srcLocaleDir: string, tempBaseDir: string) {
  ensureDir(tempBaseDir)
  const files = getAllJsonFiles(srcLocaleDir)

  for (const file of files) {
    const relativePath = file.replace(srcLocaleDir, '')
    const targetPath = join(tempBaseDir, relativePath)
    ensureDir(dirname(targetPath))
    writeFileSync(targetPath, readFileSync(file, 'utf8'))
  }
  console.warn('Captured current locale files to temp/base/')
}

// Diff command
function diff(srcLocaleDir: string, tempBaseDir: string, tempDiffDir: string) {
  ensureDir(tempDiffDir)
  const files = getAllJsonFiles(srcLocaleDir)

  for (const file of files) {
    const relativePath = file.replace(srcLocaleDir, '')
    const basePath = join(tempBaseDir, relativePath)
    const diffPath = join(tempDiffDir, relativePath)

    const baseContent = readJsonFile(basePath)
    const updatedContent = readJsonFile(file)

    const additions = findAdditions(baseContent, updatedContent)
    if (Object.keys(additions).length > 0) {
      ensureDir(dirname(diffPath))
      writeFileSync(diffPath, JSON.stringify(additions, null, 2))
      console.warn(`Wrote diff to ${diffPath}`)
    }
  }
}

// Command handling
const command = process.argv[2]
const SRC_LOCALE_DIR = 'src/locales'
const TEMP_BASE_DIR = 'temp/base'
const TEMP_DIFF_DIR = 'temp/diff'

switch (command) {
  case 'capture':
    capture(SRC_LOCALE_DIR, TEMP_BASE_DIR)
    break
  case 'diff':
    diff(SRC_LOCALE_DIR, TEMP_BASE_DIR, TEMP_DIFF_DIR)
    break
  case 'clean':
    // Remove temp directory recursively
    if (existsSync('temp')) {
      rmSync('temp', { recursive: true, force: true })
      console.warn('Removed temp directory')
    }
    break
  default:
    console.error('Please specify either "capture" or "diff" command')
}
