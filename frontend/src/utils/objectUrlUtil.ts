const objectUrlRefCounts = new Map<string, number>()

const isBlobUrl = (url: string) => url.startsWith('blob:')

export function createSharedObjectUrl(blob: Blob): string {
  const url = URL.createObjectURL(blob)
  objectUrlRefCounts.set(url, 1)
  return url
}

export function retainSharedObjectUrl(url: string | undefined): void {
  if (!url || !isBlobUrl(url)) return
  objectUrlRefCounts.set(url, (objectUrlRefCounts.get(url) ?? 0) + 1)
}

export function releaseSharedObjectUrl(url: string | undefined): void {
  if (!url || !isBlobUrl(url)) return

  const currentCount = objectUrlRefCounts.get(url)
  if (currentCount === undefined || currentCount <= 1) {
    objectUrlRefCounts.delete(url)
    URL.revokeObjectURL(url)
    return
  }

  objectUrlRefCounts.set(url, currentCount - 1)
}
