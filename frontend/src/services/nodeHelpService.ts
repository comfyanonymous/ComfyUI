import { api } from '@/scripts/api'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { NodeSourceType, getNodeSource } from '@/types/nodeSource'
import { extractCustomNodeName } from '@/workbench/utils/nodeHelpUtil'

class NodeHelpService {
  async fetchNodeHelp(node: ComfyNodeDefImpl, locale: string): Promise<string> {
    const nodeSource = getNodeSource(node.python_module)

    if (nodeSource.type === NodeSourceType.Blueprint) {
      return node.description || ''
    }

    if (nodeSource.type === NodeSourceType.CustomNodes) {
      return this.fetchCustomNodeHelp(node, locale)
    } else {
      return this.fetchCoreNodeHelp(node, locale)
    }
  }

  private async fetchCustomNodeHelp(
    node: ComfyNodeDefImpl,
    locale: string
  ): Promise<string> {
    const customNodeName = extractCustomNodeName(node.python_module)
    let lastError: string | undefined
    if (!customNodeName) {
      throw new Error('Invalid custom node module')
    }

    // Try locale-specific path first
    const localePath = `/extensions/${customNodeName}/docs/${node.name}/${locale}.md`
    const localeDoc = await this.tryFetchMarkdown(localePath)
    if (localeDoc.text) return localeDoc.text
    lastError = localeDoc.errorText

    // Fall back to non-locale path
    const fallbackPath = `/extensions/${customNodeName}/docs/${node.name}.md`
    const fallbackDoc = await this.tryFetchMarkdown(fallbackPath)
    if (fallbackDoc.text) return fallbackDoc.text
    lastError = fallbackDoc.errorText ?? lastError

    throw new Error(lastError ?? 'Help not found')
  }

  private async fetchCoreNodeHelp(
    node: ComfyNodeDefImpl,
    locale: string
  ): Promise<string> {
    const mdUrl = `/docs/${node.name}/${locale}.md`
    const doc = await this.tryFetchMarkdown(mdUrl)
    if (!doc.text) {
      throw new Error(doc.errorText ?? 'Help not found')
    }

    return doc.text
  }

  /**
   * Fetch a markdown file and return its text, guarding against HTML/SPA fallbacks.
   * Returns null when not OK or when the content type indicates HTML.
   */
  private async tryFetchMarkdown(
    path: string
  ): Promise<{ text: string | null; errorText?: string }> {
    const res = await fetch(api.fileURL(path))

    if (!res.ok) {
      return { text: null, errorText: res.statusText }
    }

    const contentType = res.headers?.get?.('content-type') ?? ''
    const text = await res.text()

    const isHtmlContentType = contentType.includes('text/html')

    if (isHtmlContentType) return { text: null, errorText: res.statusText }

    return { text }
  }
}

// Export singleton instance
export const nodeHelpService = new NodeHelpService()
