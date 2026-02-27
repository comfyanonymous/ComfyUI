import { type ComfyMetadata } from '@/types/metadataTypes'

export async function getSvgMetadata(file: File): Promise<ComfyMetadata> {
  const text = await file.text()
  const metadataMatch =
    /<metadata>\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*<\/metadata>/i.exec(text)

  if (metadataMatch && metadataMatch[1]) {
    try {
      return JSON.parse(metadataMatch[1].trim())
    } catch (error) {
      console.error('Error parsing SVG metadata:', error)
      return {}
    }
  }

  return {}
}
