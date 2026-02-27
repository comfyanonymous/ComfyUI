import path from 'path'
import type { Plugin } from 'vite'

interface ShimResult {
  code: string
  exports: string[]
}

const SKIP_WARNING_FILES = new Set(['scripts/app', 'scripts/api'])

/** Files that will be removed in v1.34 */
const DEPRECATED_FILES = [
  'scripts/ui',
  'extensions/core/groupNode',
  'extensions/core/nodeTemplates'
] as const

function getWarningMessage(
  fileKey: string,
  shimFileName: string
): string | null {
  if (SKIP_WARNING_FILES.has(fileKey)) {
    return null
  }

  const isDeprecated = DEPRECATED_FILES.some((deprecatedPath) =>
    fileKey.startsWith(deprecatedPath)
  )

  if (isDeprecated) {
    return `[ComfyUI Deprecated] Importing from "${shimFileName}" is deprecated and will be removed in v1.34.`
  }

  return `[ComfyUI Notice] "${shimFileName}" is an internal module, not part of the public API. Future updates may break this import.`
}

function isLegacyFile(id: string): boolean {
  return (
    id.endsWith('.ts') &&
    (id.includes('src/extensions/core') || id.includes('src/scripts'))
  )
}

function transformExports(code: string, id: string): ShimResult {
  const moduleName = getModuleName(id)
  const exports: string[] = []
  let newCode = code

  // Regex to match different types of exports
  const regex =
    /export\s+(const|let|var|function|class|async function)\s+([a-zA-Z$_][a-zA-Z\d$_]*)(\s|\()/g
  let match

  while ((match = regex.exec(code)) !== null) {
    const name = match[2]
    // All exports should be bind to the window object as new API endpoint.
    if (exports.length == 0) {
      newCode += `\nwindow.comfyAPI = window.comfyAPI || {};`
      newCode += `\nwindow.comfyAPI.${moduleName} = window.comfyAPI.${moduleName} || {};`
    }
    newCode += `\nwindow.comfyAPI.${moduleName}.${name} = ${name};`
    exports.push(
      `export const ${name} = window.comfyAPI.${moduleName}.${name};\n`
    )
  }

  return {
    code: newCode,
    exports
  }
}

function getModuleName(id: string): string {
  // Simple example to derive a module name from the file path
  const parts = id.split('/')
  const fileName = parts[parts.length - 1]
  return fileName.replace(/\.\w+$/, '') // Remove file extension
}

export function comfyAPIPlugin(isDev: boolean): Plugin {
  return {
    name: 'comfy-api-plugin',
    apply: 'build',
    transform(code: string, id: string) {
      if (isDev) return null

      if (isLegacyFile(id)) {
        const result = transformExports(code, id)

        if (result.exports.length > 0) {
          const projectRoot = process.cwd()
          const relativePath = path
            .relative(path.join(projectRoot, 'src'), id)
            .replace(/\\/g, '/')
          const shimFileName = relativePath.replace(/\.ts$/, '.js')

          let shimContent = `// Shim for ${relativePath}\n`

          const fileKey = relativePath.replace(/\.ts$/, '')
          const warningMessage = getWarningMessage(fileKey, shimFileName)

          if (warningMessage) {
            // It will only display once because it is at the root of the file.
            shimContent += `console.warn('${warningMessage}');\n`
          }

          shimContent += result.exports.join('')

          this.emitFile({
            type: 'asset',
            fileName: shimFileName,
            source: shimContent
          })
        }

        return {
          code: result.code,
          map: null // If you're not modifying the source map, return null
        }
      }
    }
  }
}
