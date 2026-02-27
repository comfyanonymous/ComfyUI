// @ts-check
import { markdownTable } from 'markdown-table'
import { existsSync } from 'node:fs'
import { readdir } from 'node:fs/promises'
import path from 'node:path'
import prettyBytes from 'pretty-bytes'

import { getCategoryMetadata } from './bundle-categories.js'

/**
 * @typedef {Object} SizeMetrics
 * @property {number} size
 * @property {number} gzip
 * @property {number} brotli
 */

/**
 * @typedef {Object} SizeResult
 * @property {number} size
 * @property {number} gzip
 * @property {number} brotli
 */

/**
 * @typedef {SizeResult & { file: string, category?: string }} BundleResult
 */

/**
 * @typedef {'added' | 'removed' | 'increased' | 'decreased' | 'unchanged'} BundleStatus
 */

/**
 * @typedef {Object} BundleDiff
 * @property {string} fileName
 * @property {BundleResult | undefined} curr
 * @property {BundleResult | undefined} prev
 * @property {SizeMetrics} diff
 * @property {BundleStatus} status
 */

/**
 * @typedef {Object} CountSummary
 * @property {number} added
 * @property {number} removed
 * @property {number} increased
 * @property {number} decreased
 * @property {number} unchanged
 */

/**
 * @typedef {Object} CategoryReport
 * @property {string} name
 * @property {string | undefined} description
 * @property {number} order
 * @property {{ current: SizeMetrics, baseline: SizeMetrics, diff: SizeMetrics }} metrics
 * @property {CountSummary} counts
 * @property {BundleDiff[]} bundles
 */

/**
 * @typedef {Object} BundleReport
 * @property {CategoryReport[]} categories
 * @property {{ currentBundles: number, baselineBundles: number, metrics: { current: SizeMetrics, baseline: SizeMetrics, diff: SizeMetrics }, counts: CountSummary }} overall
 * @property {boolean} hasBaseline
 */

const currDir = path.resolve('temp/size')
const prevDir = path.resolve('temp/size-prev')

run()

/**
 * Main entry for generating the size report
 */
async function run() {
  if (!existsSync(currDir)) {
    console.error('Error: temp/size directory does not exist')
    console.error('Please run "pnpm size:collect" first')
    process.exit(1)
  }

  const report = await buildBundleReport()
  const output = renderReport(report)
  process.stdout.write(output)
}

/**
 * Build bundle comparison data from current and baseline artifacts
 * @returns {Promise<BundleReport>}
 */
async function buildBundleReport() {
  /**
   * @param {string[]} files
   * @returns {string[]}
   */
  const filterFiles = (files) => files.filter((file) => file.endsWith('.json'))

  const currFiles = filterFiles(await readdir(currDir))
  const baselineFiles = existsSync(prevDir)
    ? filterFiles(await readdir(prevDir))
    : []
  const fileList = new Set([...currFiles, ...baselineFiles])

  /** @type {Map<string, CategoryReport>} */
  const categories = new Map()

  const overall = {
    currentBundles: 0,
    baselineBundles: 0,
    metrics: {
      current: createMetrics(),
      baseline: createMetrics(),
      diff: createMetrics()
    },
    counts: createCounts()
  }

  for (const file of fileList) {
    const currPath = path.resolve(currDir, file)
    const prevPath = path.resolve(prevDir, file)

    const curr = await importJSON(currPath)
    const prev = await importJSON(prevPath)
    const fileName = curr?.file || prev?.file
    if (!fileName) continue

    const categoryName = curr?.category || prev?.category || 'Other'
    const category = ensureCategoryEntry(categories, categoryName)

    const currMetrics = toMetrics(curr)
    const baselineMetrics = toMetrics(prev)
    const diffMetrics = subtractMetrics(currMetrics, baselineMetrics)
    const status = getStatus(curr, prev, diffMetrics.size)

    if (curr) {
      overall.currentBundles++
    }
    if (prev) {
      overall.baselineBundles++
    }

    addMetrics(overall.metrics.current, currMetrics)
    addMetrics(overall.metrics.baseline, baselineMetrics)
    addMetrics(overall.metrics.diff, diffMetrics)
    incrementStatus(overall.counts, status)

    addMetrics(category.metrics.current, currMetrics)
    addMetrics(category.metrics.baseline, baselineMetrics)
    addMetrics(category.metrics.diff, diffMetrics)
    incrementStatus(category.counts, status)

    category.bundles.push({
      fileName,
      curr,
      prev,
      diff: diffMetrics,
      status
    })
  }

  const sortedCategories = Array.from(categories.values()).sort(
    (a, b) => a.order - b.order
  )

  return {
    categories: sortedCategories,
    overall,
    hasBaseline: baselineFiles.length > 0
  }
}

/**
 * Render the complete report in markdown
 * @param {BundleReport} report
 * @returns {string}
 */
function renderReport(report) {
  const parts = [renderCompactHeader(report)]

  if (report.categories.length > 0) {
    parts.push('\n' + renderCategoryDetails(report))
  }

  return (
    parts
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trimEnd() + '\n'
  )
}

/**
 * Render compact single-line header with key metrics
 * @param {BundleReport} report
 * @returns {string}
 */
function renderCompactHeader(report) {
  const { overall, hasBaseline } = report

  const gzipSize = prettyBytes(overall.metrics.current.gzip)
  let header = `## 📦 Bundle: ${gzipSize} gzip`

  if (hasBaseline) {
    header += ` ${formatDiffIndicator(overall.metrics.diff.gzip)}`
  }

  return header
}

/**
 * Render overall summary bullets
 * @param {BundleReport} report
 * @returns {string}
 */
function renderSummary(report) {
  const { overall, hasBaseline } = report
  const lines = ['**Summary**']

  const rawLineParts = [
    `- Raw size: ${prettyBytes(overall.metrics.current.size)}`
  ]
  if (hasBaseline) {
    rawLineParts.push(`baseline ${prettyBytes(overall.metrics.baseline.size)}`)
    rawLineParts.push(`— ${formatDiffIndicator(overall.metrics.diff.size)}`)
  }
  lines.push(rawLineParts.join(' '))

  const gzipLineParts = [`- Gzip: ${prettyBytes(overall.metrics.current.gzip)}`]
  if (hasBaseline) {
    gzipLineParts.push(`baseline ${prettyBytes(overall.metrics.baseline.gzip)}`)
    gzipLineParts.push(`— ${formatDiffIndicator(overall.metrics.diff.gzip)}`)
  }
  lines.push(gzipLineParts.join(' '))

  const brotliLineParts = [
    `- Brotli: ${prettyBytes(overall.metrics.current.brotli)}`
  ]
  if (hasBaseline) {
    brotliLineParts.push(
      `baseline ${prettyBytes(overall.metrics.baseline.brotli)}`
    )
    brotliLineParts.push(
      `— ${formatDiffIndicator(overall.metrics.diff.brotli)}`
    )
  }
  lines.push(brotliLineParts.join(' '))

  const bundleStats = [`${overall.currentBundles} current`]
  if (hasBaseline) {
    bundleStats.push(`${overall.baselineBundles} baseline`)
  }

  const statusParts = []
  if (overall.counts.added) statusParts.push(`${overall.counts.added} added`)
  if (overall.counts.removed)
    statusParts.push(`${overall.counts.removed} removed`)
  if (overall.counts.increased)
    statusParts.push(`${overall.counts.increased} grew`)
  if (overall.counts.decreased)
    statusParts.push(`${overall.counts.decreased} shrank`)

  let bundlesLine = `- Bundles: ${bundleStats.join(' • ')}`
  if (statusParts.length > 0) {
    bundlesLine += ` • ${statusParts.join(' / ')}`
  }
  lines.push(bundlesLine)

  if (!hasBaseline) {
    lines.push(
      '_Baseline artifact not found; showing current bundle sizes only._'
    )
  }

  return lines.join('\n')
}

/**
 * Render a compact category glance line
 * @param {BundleReport} report
 * @returns {string}
 */
function renderCategoryGlance(report) {
  const { categories, hasBaseline } = report
  const relevant = categories.filter(
    (category) =>
      category.metrics.current.size > 0 ||
      (hasBaseline && category.metrics.baseline.size > 0)
  )

  if (relevant.length === 0) return ''

  const sorted = relevant.slice().sort((a, b) => {
    if (hasBaseline) {
      return (
        Math.abs(b.metrics.diff.size) - Math.abs(a.metrics.diff.size) ||
        b.metrics.current.size - a.metrics.current.size
      )
    }
    return b.metrics.current.size - a.metrics.current.size
  })

  const limit = 6
  const trimmed = sorted.slice(0, limit)
  const parts = trimmed.map((category) => {
    const currentStr = prettyBytes(category.metrics.current.size)
    if (hasBaseline) {
      return `${category.name} ${formatDiffIndicator(category.metrics.diff.size)} (${currentStr})`
    }
    return `${category.name} ${currentStr}`
  })

  if (sorted.length > limit) {
    parts.push(`+ ${sorted.length - limit} more`)
  }

  return `**Category Glance**\n${parts.join(' · ')}`
}

/**
 * Render per-category detail tables wrapped in collapsible sections
 * @param {BundleReport} report
 * @returns {string}
 */
function renderCategoryDetails(report) {
  const lines = ['<details>', '<summary>Details</summary>', '']

  lines.push(renderSummary(report))
  lines.push('')

  const glance = renderCategoryGlance(report)
  if (glance) {
    lines.push(glance)
    lines.push('')
  }

  for (const category of report.categories) {
    lines.push(renderCategoryBlock(category, report.hasBaseline))
    lines.push('')
  }

  if (report.categories.length > 0) {
    lines.pop()
  }

  lines.push('</details>')
  return lines.join('\n')
}

/**
 * Render a single category block with its table
 * @param {CategoryReport} category
 * @param {boolean} hasBaseline
 * @returns {string}
 */
function renderCategoryBlock(category, hasBaseline) {
  const lines = ['<details>']
  const currentStr = prettyBytes(category.metrics.current.size)
  const summaryParts = [`<summary>${category.name} — ${currentStr}`]

  if (hasBaseline) {
    summaryParts.push(
      ` (baseline ${prettyBytes(category.metrics.baseline.size)}) • ${formatDiffIndicator(category.metrics.diff.size)}`
    )
  }

  summaryParts.push('</summary>')
  lines.push(summaryParts.join(''))
  lines.push('')

  if (category.description) {
    lines.push(`_${category.description}_`)
    lines.push('')
  }

  if (category.bundles.length === 0) {
    lines.push('No bundles matched this category.\n')
    lines.push('</details>\n')
    return lines.join('\n')
  }

  const headers = hasBaseline
    ? ['File', 'Before', 'After', 'Δ Raw', 'Δ Gzip', 'Δ Brotli']
    : ['File', 'Size', 'Gzip', 'Brotli']

  const rows = category.bundles
    .slice()
    .sort((a, b) => {
      const diffMagnitude = Math.abs(b.diff.size) - Math.abs(a.diff.size)
      if (diffMagnitude !== 0) return diffMagnitude
      return a.fileName.localeCompare(b.fileName)
    })
    .map((bundle) => {
      if (hasBaseline) {
        return [
          formatFileLabel(bundle),
          formatSize(bundle.prev?.size),
          formatSize(bundle.curr?.size),
          formatDiffIndicator(bundle.diff.size),
          formatDiffIndicator(bundle.diff.gzip),
          formatDiffIndicator(bundle.diff.brotli)
        ]
      }

      return [
        formatFileLabel(bundle),
        formatSize(bundle.curr?.size),
        formatSize(bundle.curr?.gzip),
        formatSize(bundle.curr?.brotli)
      ]
    })

  lines.push(markdownTable([headers, ...rows]))
  lines.push('')

  const statusParts = []
  if (category.counts.added) statusParts.push(`${category.counts.added} added`)
  if (category.counts.removed)
    statusParts.push(`${category.counts.removed} removed`)
  if (category.counts.increased)
    statusParts.push(`${category.counts.increased} grew`)
  if (category.counts.decreased)
    statusParts.push(`${category.counts.decreased} shrank`)

  if (statusParts.length > 0) {
    lines.push(`_Status:_ ${statusParts.join(' / ')}`)
    lines.push('')
  }

  lines.push('</details>')
  return lines.join('\n')
}

/**
 * Ensure a category entry exists in the map
 * @param {Map<string, CategoryReport>} categories
 * @param {string} categoryName
 * @returns {CategoryReport}
 */
function ensureCategoryEntry(categories, categoryName) {
  if (!categories.has(categoryName)) {
    const meta = getCategoryMetadata(categoryName)
    categories.set(categoryName, {
      name: categoryName,
      description: meta?.description,
      order: meta?.order ?? 99,
      metrics: {
        current: createMetrics(),
        baseline: createMetrics(),
        diff: createMetrics()
      },
      counts: createCounts(),
      bundles: []
    })
  }
  // @ts-expect-error - ensured by check above
  return categories.get(categoryName)
}

/**
 * Convert bundle result to metrics
 * @param {BundleResult | undefined} bundle
 * @returns {SizeMetrics}
 */
function toMetrics(bundle) {
  if (!bundle) return createMetrics()
  return {
    size: bundle.size,
    gzip: bundle.gzip,
    brotli: bundle.brotli
  }
}

/**
 * Create an empty metrics object
 * @returns {SizeMetrics}
 */
function createMetrics() {
  return { size: 0, gzip: 0, brotli: 0 }
}

/**
 * Add source metrics into target metrics
 * @param {SizeMetrics} target
 * @param {SizeMetrics} source
 */
function addMetrics(target, source) {
  target.size += source.size
  target.gzip += source.gzip
  target.brotli += source.brotli
}

/**
 * Subtract baseline metrics from current metrics
 * @param {SizeMetrics} current
 * @param {SizeMetrics} baseline
 * @returns {SizeMetrics}
 */
function subtractMetrics(current, baseline) {
  return {
    size: current.size - baseline.size,
    gzip: current.gzip - baseline.gzip,
    brotli: current.brotli - baseline.brotli
  }
}

/**
 * Create an empty counts object
 * @returns {CountSummary}
 */
function createCounts() {
  return { added: 0, removed: 0, increased: 0, decreased: 0, unchanged: 0 }
}

/**
 * Increment status counters
 * @param {CountSummary} counts
 * @param {BundleStatus} status
 */
function incrementStatus(counts, status) {
  counts[status] += 1
}

/**
 * Determine bundle status for reporting
 * @param {BundleResult | undefined} curr
 * @param {BundleResult | undefined} prev
 * @param {number} sizeDiff
 * @returns {BundleStatus}
 */
function getStatus(curr, prev, sizeDiff) {
  if (curr && prev) {
    if (sizeDiff > 0) return 'increased'
    if (sizeDiff < 0) return 'decreased'
    return 'unchanged'
  }
  if (curr && !prev) return 'added'
  if (!curr && prev) return 'removed'
  return 'unchanged'
}

/**
 * Format file label with status hints
 * @param {BundleDiff} bundle
 * @returns {string}
 */
function formatFileLabel(bundle) {
  if (bundle.status === 'added') {
    return `**${bundle.fileName}** _(new)_`
  }
  if (bundle.status === 'removed') {
    return `~~${bundle.fileName}~~ _(removed)_`
  }
  return bundle.fileName
}

/**
 * Format size for table output
 * @param {number | undefined} value
 * @returns {string}
 */
function formatSize(value) {
  if (value === undefined) return '—'
  return prettyBytes(value)
}

/**
 * Format a diff with an indicator emoji
 * @param {number} diff
 * @returns {string}
 */
function formatDiffIndicator(diff) {
  if (diff > 0) {
    return `:red_circle: +${prettyBytes(diff)}`
  }
  if (diff < 0) {
    return `:green_circle: -${prettyBytes(Math.abs(diff))}`
  }
  return ':white_circle: 0 B'
}

/**
 * Import JSON data if it exists
 * @template T
 * @param {string} filePath
 * @returns {Promise<T | undefined>}
 */
async function importJSON(filePath) {
  if (!existsSync(filePath)) return undefined
  return (await import(filePath, { with: { type: 'json' } })).default
}
