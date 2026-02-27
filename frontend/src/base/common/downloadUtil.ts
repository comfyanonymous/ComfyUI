/**
 * Utility functions for downloading files
 */
import { t } from '@/i18n'
import { isCloud } from '@/platform/distribution/types'
import { useToastStore } from '@/platform/updates/common/toastStore'

// Constants
const DEFAULT_DOWNLOAD_FILENAME = 'download.png'

/**
 * Trigger a download by creating a temporary anchor element
 * @param href - The URL or blob URL to download
 * @param filename - The filename to suggest to the browser
 */
function triggerLinkDownload(href: string, filename: string): void {
  const link = document.createElement('a')
  link.href = href
  link.download = filename
  link.style.display = 'none'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Download a file from a URL by creating a temporary anchor element
 * @param url - The URL of the file to download (must be a valid URL string)
 * @param filename - Optional filename override (will use URL filename or default if not provided)
 * @throws {Error} If the URL is invalid or empty
 */
export function downloadFile(url: string, filename?: string): void {
  if (!url || typeof url !== 'string' || url.trim().length === 0) {
    throw new Error('Invalid URL provided for download')
  }

  const inferredFilename =
    filename || extractFilenameFromUrl(url) || DEFAULT_DOWNLOAD_FILENAME

  if (isCloud) {
    // Assets from cross-origin (e.g., GCS) cannot be downloaded this way
    void downloadViaBlobFetch(url, inferredFilename).catch((error) => {
      console.error('Failed to download file', error)
    })
    return
  }

  triggerLinkDownload(url, inferredFilename)
}

/**
 * Download a Blob by creating a temporary object URL and anchor element
 * @param filename - The filename to suggest to the browser
 * @param blob - The Blob to download
 */
export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)

  triggerLinkDownload(url, filename)

  // Revoke on the next microtask to give the browser time to start the download
  queueMicrotask(() => URL.revokeObjectURL(url))
}

/**
 * Extract filename from a URL's query parameters
 * @param url - The URL to extract filename from
 * @returns The extracted filename or null if not found
 */
const extractFilenameFromUrl = (url: string): string | null => {
  try {
    const urlObj = new URL(url, window.location.origin)
    return urlObj.searchParams.get('filename')
  } catch {
    return null
  }
}

/**
 * Extract filename from Content-Disposition header
 * Handles both simple format: attachment; filename="name.png"
 * And RFC 5987 format: attachment; filename="fallback.png"; filename*=UTF-8''encoded%20name.png
 * @param header - The Content-Disposition header value
 * @returns The extracted filename or null if not found
 */
export function extractFilenameFromContentDisposition(
  header: string | null
): string | null {
  if (!header) return null

  // Try RFC 5987 extended format first (filename*=UTF-8''...)
  const extendedMatch = header.match(/filename\*=UTF-8''([^;]+)/i)
  if (extendedMatch?.[1]) {
    try {
      return decodeURIComponent(extendedMatch[1])
    } catch {
      // Fall through to simple format
    }
  }

  // Try simple quoted format: filename="..."
  const quotedMatch = header.match(/filename="([^"]+)"/i)
  if (quotedMatch?.[1]) {
    return quotedMatch[1]
  }

  // Try unquoted format: filename=...
  const unquotedMatch = header.match(/filename=([^;\s]+)/i)
  if (unquotedMatch?.[1]) {
    return unquotedMatch[1]
  }

  return null
}

/**
 * Fetch a URL and return its body as a Blob.
 * Shared by download and open-in-new-tab cloud paths.
 */
async function fetchAsBlob(url: string): Promise<Response> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`)
  }
  return response
}

async function downloadViaBlobFetch(
  href: string,
  fallbackFilename: string
): Promise<void> {
  const response = await fetchAsBlob(href)

  // Try to get filename from Content-Disposition header (set by backend)
  const contentDisposition = response.headers.get('Content-Disposition')
  const headerFilename =
    extractFilenameFromContentDisposition(contentDisposition)

  const blob = await response.blob()
  downloadBlob(headerFilename ?? fallbackFilename, blob)
}

/**
 * Open a file URL in a new browser tab.
 * On cloud, fetches the resource as a blob first to avoid GCS redirects
 * that would trigger an auto-download instead of displaying the file.
 *
 * Opens the tab synchronously to preserve the user-gesture context
 * (browsers block window.open after an await), then navigates it to
 * the blob URL once the fetch completes.
 */
export async function openFileInNewTab(url: string): Promise<void> {
  if (!isCloud) {
    window.open(url, '_blank')
    return
  }

  // Open immediately to preserve user-gesture activation.
  const tab = window.open('', '_blank')

  try {
    const response = await fetchAsBlob(url)
    const blob = await response.blob()
    const blobUrl = URL.createObjectURL(blob)

    if (tab && !tab.closed) {
      tab.location.href = blobUrl
      // Revoke after the tab has had time to load the blob.
      setTimeout(() => URL.revokeObjectURL(blobUrl), 60_000)
    } else {
      URL.revokeObjectURL(blobUrl)
    }
  } catch (error) {
    tab?.close()
    console.error('Failed to open image:', error)
    useToastStore().addAlert(
      t('toastMessages.errorOpenImage', {
        error: error instanceof Error ? error.message : String(error)
      })
    )
  }
}
