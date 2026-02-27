import type { ImportSource } from '@/platform/assets/types/importSource'

/**
 * Hugging Face model import source configuration
 */
export const huggingfaceImportSource: ImportSource = {
  type: 'huggingface',
  name: 'Hugging Face',
  hostnames: ['huggingface.co']
}
