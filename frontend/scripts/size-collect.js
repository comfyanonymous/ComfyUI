// @ts-check
import { existsSync } from 'node:fs'
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { brotliCompressSync, gzipSync } from 'node:zlib'
import pico from 'picocolors'
import prettyBytes from 'pretty-bytes'

import { categorizeBundle } from './bundle-categories.js'

const distDir = path.resolve('dist')
const sizeDir = path.resolve('temp/size')

/**
 * @typedef {Object} SizeResult
 * @property {string} file
 * @property {string} category
 * @property {number} size
 * @property {number} gzip
 * @property {number} brotli
 */

run()

/**
 * Main function to collect bundle size data
 */
async function run() {
  if (!existsSync(distDir)) {
    console.error(pico.red('Error: dist directory does not exist'))
    console.error(pico.yellow('Please run "pnpm build" first'))
    process.exit(1)
  }

  console.log(pico.blue('\nCollecting bundle size data...\n'))

  // Collect main bundle files from dist/assets
  const assetsDir = path.join(distDir, 'assets')
  const bundles = []

  if (existsSync(assetsDir)) {
    const files = await readdir(assetsDir)
    const jsFiles = files.filter(
      (file) => file.endsWith('.js') && !file.includes('legacy')
    )

    for (const file of jsFiles) {
      const filePath = path.join(assetsDir, file)
      const content = await readFile(filePath, 'utf-8')
      const size = Buffer.byteLength(content)
      const gzip = gzipSync(content).length
      const brotli = brotliCompressSync(content).length
      const fileName = `assets/${file}`
      const category = categorizeBundle(fileName)

      bundles.push({
        file: fileName,
        category,
        size,
        gzip,
        brotli
      })

      console.log(
        `${pico.green(file)} ${pico.dim(`[${category}]`)} - ` +
          `Size: ${prettyBytes(size)} / ` +
          `Gzip: ${prettyBytes(gzip)} / ` +
          `Brotli: ${prettyBytes(brotli)}`
      )
    }
  }

  // Create temp/size directory
  await mkdir(sizeDir, { recursive: true })

  // Write individual bundle files
  for (const bundle of bundles) {
    const fileName = bundle.file.replace(/[/\\]/g, '_').replace('.js', '.json')
    await writeFile(
      path.join(sizeDir, fileName),
      JSON.stringify(bundle, null, 2),
      'utf-8'
    )
  }

  console.log(
    pico.green(`\n✓ Collected size data for ${bundles.length} bundles\n`)
  )
  console.log(pico.blue(`Data saved to: ${sizeDir}\n`))
}
