/**
 * Supported model import sources
 */
type ImportSourceType = 'civitai' | 'huggingface'

/**
 * Configuration for a model import source
 */
export interface ImportSource {
  /**
   * Unique identifier for this import source
   */
  readonly type: ImportSourceType

  /**
   * Display name for the source
   */
  readonly name: string

  /**
   * Hostname(s) that identify this source
   */
  readonly hostnames: readonly string[]
}
