import type { ImportSource } from '@/platform/assets/types/importSource'

/**
 * Civitai model import source configuration
 */
export const civitaiImportSource: ImportSource = {
  type: 'civitai',
  name: 'Civitai',
  hostnames: ['civitai.com']
}
