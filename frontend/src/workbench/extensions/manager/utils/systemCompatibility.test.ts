import { describe, expect, it } from 'vitest'

import type {
  RegistryAccelerator,
  RegistryOS
} from '@/workbench/extensions/manager/types/compatibility.types'
import {
  checkAcceleratorCompatibility,
  checkOSCompatibility,
  normalizeOSList
} from '@/workbench/extensions/manager/utils/systemCompatibility'

describe('systemCompatibility', () => {
  describe('checkOSCompatibility', () => {
    it('should return null when supported OS list is null', () => {
      const result = checkOSCompatibility(null, 'darwin')
      expect(result).toBeNull()
    })

    it('should return null when supported OS list is undefined', () => {
      const result = checkOSCompatibility(undefined, 'darwin')
      expect(result).toBeNull()
    })

    it('should return null when supported OS list is empty', () => {
      const result = checkOSCompatibility([], 'darwin')
      expect(result).toBeNull()
    })

    it('should return null when OS is compatible (macOS)', () => {
      const supported: RegistryOS[] = ['macOS', 'Windows']
      const result = checkOSCompatibility(supported, 'darwin')
      expect(result).toBeNull()
    })

    it('should return null when OS is compatible (Windows)', () => {
      const supported: RegistryOS[] = ['Windows', 'Linux']
      const result = checkOSCompatibility(supported, 'win32')
      expect(result).toBeNull()
    })

    it('should return null when OS is compatible (Linux)', () => {
      const supported: RegistryOS[] = ['Linux', 'macOS']
      const result = checkOSCompatibility(supported, 'linux')
      expect(result).toBeNull()
    })

    it('should return conflict when OS is incompatible', () => {
      const supported: RegistryOS[] = ['Windows']
      const result = checkOSCompatibility(supported, 'darwin')
      expect(result).toEqual({
        type: 'os',
        current_value: 'macOS',
        required_value: 'Windows'
      })
    })

    it('should return conflict with Unknown OS when current OS is unrecognized', () => {
      const supported: RegistryOS[] = ['Windows', 'Linux']
      const result = checkOSCompatibility(supported, 'freebsd')
      expect(result).toEqual({
        type: 'os',
        current_value: 'Unknown',
        required_value: 'Windows, Linux'
      })
    })

    it('should handle various OS string formats', () => {
      const supported: RegistryOS[] = ['Windows']

      // Test Windows variations
      expect(checkOSCompatibility(supported, 'win32')).toBeNull()
      expect(checkOSCompatibility(supported, 'windows')).toBeNull()
      expect(checkOSCompatibility(supported, 'Windows_NT')).toBeNull()

      // Test macOS variations
      const macSupported: RegistryOS[] = ['macOS']
      expect(checkOSCompatibility(macSupported, 'darwin')).toBeNull()
      expect(checkOSCompatibility(macSupported, 'Darwin')).toBeNull()
      expect(checkOSCompatibility(macSupported, 'macos')).toBeNull()
      expect(checkOSCompatibility(macSupported, 'mac')).toBeNull()
    })

    it('should handle undefined current OS', () => {
      const supported: RegistryOS[] = ['Windows']
      const result = checkOSCompatibility(supported, undefined)
      expect(result).toEqual({
        type: 'os',
        current_value: 'Unknown',
        required_value: 'Windows'
      })
    })
  })

  describe('checkAcceleratorCompatibility', () => {
    it('should return null when supported accelerator list is null', () => {
      const result = checkAcceleratorCompatibility(null, 'cuda')
      expect(result).toBeNull()
    })

    it('should return null when supported accelerator list is undefined', () => {
      const result = checkAcceleratorCompatibility(undefined, 'cuda')
      expect(result).toBeNull()
    })

    it('should return null when supported accelerator list is empty', () => {
      const result = checkAcceleratorCompatibility([], 'cuda')
      expect(result).toBeNull()
    })

    it('should return null when accelerator is compatible (CUDA)', () => {
      const supported: RegistryAccelerator[] = ['CUDA', 'CPU']
      const result = checkAcceleratorCompatibility(supported, 'cuda')
      expect(result).toBeNull()
    })

    it('should return null when accelerator is compatible (Metal)', () => {
      const supported: RegistryAccelerator[] = ['Metal', 'CPU']
      const result = checkAcceleratorCompatibility(supported, 'mps')
      expect(result).toBeNull()
    })

    it('should return null when accelerator is compatible (ROCm)', () => {
      const supported: RegistryAccelerator[] = ['ROCm', 'CPU']
      const result = checkAcceleratorCompatibility(supported, 'rocm')
      expect(result).toBeNull()
    })

    it('should return null when accelerator is compatible (CPU)', () => {
      const supported: RegistryAccelerator[] = ['CPU']
      const result = checkAcceleratorCompatibility(supported, 'cpu')
      expect(result).toBeNull()
    })

    it('should return conflict when accelerator is incompatible', () => {
      const supported: RegistryAccelerator[] = ['CUDA']
      const result = checkAcceleratorCompatibility(supported, 'mps')
      expect(result).toEqual({
        type: 'accelerator',
        current_value: 'Metal',
        required_value: 'CUDA'
      })
    })

    it('should default to CPU for unknown device types', () => {
      const supported: RegistryAccelerator[] = ['CUDA']
      const result = checkAcceleratorCompatibility(supported, 'unknown')
      expect(result).toEqual({
        type: 'accelerator',
        current_value: 'CPU',
        required_value: 'CUDA'
      })
    })

    it('should default to CPU when device type is undefined', () => {
      const supported: RegistryAccelerator[] = ['CUDA']
      const result = checkAcceleratorCompatibility(supported, undefined)
      expect(result).toEqual({
        type: 'accelerator',
        current_value: 'CPU',
        required_value: 'CUDA'
      })
    })

    it('should handle case-insensitive device types', () => {
      const supported: RegistryAccelerator[] = ['CUDA']

      // CUDA variations
      expect(checkAcceleratorCompatibility(supported, 'cuda')).toBeNull()
      expect(checkAcceleratorCompatibility(supported, 'CUDA')).toBeNull()
      expect(checkAcceleratorCompatibility(supported, 'Cuda')).toBeNull()

      // Metal variations
      const metalSupported: RegistryAccelerator[] = ['Metal']
      expect(checkAcceleratorCompatibility(metalSupported, 'mps')).toBeNull()
      expect(checkAcceleratorCompatibility(metalSupported, 'MPS')).toBeNull()

      // ROCm variations
      const rocmSupported: RegistryAccelerator[] = ['ROCm']
      expect(checkAcceleratorCompatibility(rocmSupported, 'rocm')).toBeNull()
      expect(checkAcceleratorCompatibility(rocmSupported, 'ROCM')).toBeNull()
    })

    it('should handle multiple required accelerators', () => {
      const supported: RegistryAccelerator[] = ['CUDA', 'ROCm']
      const result = checkAcceleratorCompatibility(supported, 'mps')
      expect(result).toEqual({
        type: 'accelerator',
        current_value: 'Metal',
        required_value: 'CUDA, ROCm'
      })
    })
  })

  describe('normalizeOSList', () => {
    it('should return undefined for null input', () => {
      const result = normalizeOSList(null)
      expect(result).toBeUndefined()
    })

    it('should return undefined for undefined input', () => {
      const result = normalizeOSList(undefined)
      expect(result).toBeUndefined()
    })

    it('should return undefined for empty array', () => {
      const result = normalizeOSList([])
      expect(result).toBeUndefined()
    })

    it('should return undefined when OS Independent is present', () => {
      const result = normalizeOSList(['OS Independent', 'Windows'])
      expect(result).toBeUndefined()
    })

    it('should return undefined for case-insensitive OS Independent', () => {
      const result = normalizeOSList(['os independent'])
      expect(result).toBeUndefined()
    })

    it('should filter and return valid OS values', () => {
      const result = normalizeOSList(['Windows', 'Linux', 'macOS'])
      expect(result).toEqual(['Windows', 'Linux', 'macOS'])
    })

    it('should filter out invalid OS values', () => {
      const result = normalizeOSList(['Windows', 'FreeBSD', 'Linux', 'Android'])
      expect(result).toEqual(['Windows', 'Linux'])
    })

    it('should deduplicate OS values', () => {
      const result = normalizeOSList([
        'Windows',
        'Linux',
        'Windows',
        'macOS',
        'Linux'
      ])
      expect(result).toEqual(['Windows', 'Linux', 'macOS'])
    })

    it('should return undefined when no valid OS values remain', () => {
      const result = normalizeOSList(['FreeBSD', 'Android', 'iOS'])
      expect(result).toBeUndefined()
    })

    it('should handle mixed valid and invalid values', () => {
      const result = normalizeOSList([
        'windows',
        'Windows',
        'linux',
        'Linux',
        'macos'
      ])
      // Only exact matches are valid
      expect(result).toEqual(['Windows', 'Linux'])
    })

    it('should preserve order of first occurrence when deduplicating', () => {
      const result = normalizeOSList([
        'Linux',
        'Windows',
        'macOS',
        'Linux',
        'Windows'
      ])
      expect(result).toEqual(['Linux', 'Windows', 'macOS'])
    })
  })
})
