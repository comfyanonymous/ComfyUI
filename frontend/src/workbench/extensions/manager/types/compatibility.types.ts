/**
 * System compatibility type definitions
 * Registry supports exactly these values, null/undefined means compatible with all
 */

// Registry OS
export type RegistryOS = 'Windows' | 'macOS' | 'Linux'

// Registry Accelerator
export type RegistryAccelerator = 'CUDA' | 'ROCm' | 'Metal' | 'CPU'
