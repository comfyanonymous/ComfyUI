/**
 * Load3D constants that don't require THREE.js
 * This file can be imported without pulling in the entire THREE.js bundle
 */

export const SUPPORTED_EXTENSIONS = new Set([
  '.gltf',
  '.glb',
  '.obj',
  '.fbx',
  '.stl',
  '.spz',
  '.splat',
  '.ply',
  '.ksplat'
])

export const SUPPORTED_EXTENSIONS_ACCEPT = [...SUPPORTED_EXTENSIONS].join(',')
