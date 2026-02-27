/**
 * Supported workflow file formats organized by type category
 */

/**
 * All supported image formats that can contain workflow data
 */
const IMAGE_WORKFLOW_FORMATS = {
  extensions: ['.png', '.webp', '.svg', '.avif'],
  mimeTypes: ['image/png', 'image/webp', 'image/svg+xml', 'image/avif']
}

/**
 * All supported audio formats that can contain workflow data
 */
const AUDIO_WORKFLOW_FORMATS = {
  extensions: ['.mp3', '.ogg', '.flac'],
  mimeTypes: ['audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/x-flac']
}

/**
 * All supported video formats that can contain workflow data
 */
const VIDEO_WORKFLOW_FORMATS = {
  extensions: ['.mp4', '.mov', '.m4v', '.webm'],
  mimeTypes: ['video/mp4', 'video/quicktime', 'video/x-m4v', 'video/webm']
}

/**
 * All supported 3D model formats that can contain workflow data
 */
const MODEL_WORKFLOW_FORMATS = {
  extensions: ['.glb'],
  mimeTypes: ['model/gltf-binary']
}

/**
 * All supported data formats that directly contain workflow data
 */
const DATA_WORKFLOW_FORMATS = {
  extensions: ['.json', '.latent', '.safetensors'],
  mimeTypes: ['application/json']
}

/**
 * Combines all supported formats into a single object
 */
const ALL_WORKFLOW_FORMATS = {
  extensions: [
    ...IMAGE_WORKFLOW_FORMATS.extensions,
    ...AUDIO_WORKFLOW_FORMATS.extensions,
    ...VIDEO_WORKFLOW_FORMATS.extensions,
    ...MODEL_WORKFLOW_FORMATS.extensions,
    ...DATA_WORKFLOW_FORMATS.extensions
  ],
  mimeTypes: [
    ...IMAGE_WORKFLOW_FORMATS.mimeTypes,
    ...AUDIO_WORKFLOW_FORMATS.mimeTypes,
    ...VIDEO_WORKFLOW_FORMATS.mimeTypes,
    ...MODEL_WORKFLOW_FORMATS.mimeTypes,
    ...DATA_WORKFLOW_FORMATS.mimeTypes
  ]
}

/**
 * Generate a comma-separated accept string for file inputs
 * Combines all extensions and mime types
 */
export const WORKFLOW_ACCEPT_STRING = [
  ...ALL_WORKFLOW_FORMATS.extensions,
  ...ALL_WORKFLOW_FORMATS.mimeTypes
].join(',')
