import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  checkVersionCompatibility,
  getFrontendVersion
} from '@/workbench/extensions/manager/utils/versionUtil'

const mockConfig = vi.hoisted((): { app_version: string | undefined } => ({
  app_version: '1.24.0-1'
}))

vi.mock('@/config', () => ({
  default: mockConfig
}))

describe('versionUtil', () => {
  describe('checkVersionCompatibility', () => {
    it('should return null when current version is undefined', () => {
      const result = checkVersionCompatibility(
        'comfyui_version',
        undefined,
        '>=1.0.0'
      )
      expect(result).toBeNull()
    })

    it('should return null when current version is null', () => {
      const result = checkVersionCompatibility(
        'comfyui_version',
        null!,
        '>=1.0.0'
      )
      expect(result).toBeNull()
    })

    it('should return null when current version is empty string', () => {
      const result = checkVersionCompatibility('comfyui_version', '', '>=1.0.0')
      expect(result).toBeNull()
    })

    it('should return null when supported version is undefined', () => {
      const result = checkVersionCompatibility(
        'comfyui_version',
        '1.0.0',
        undefined
      )
      expect(result).toBeNull()
    })

    it('should return null when supported version is null', () => {
      const result = checkVersionCompatibility(
        'comfyui_version',
        '1.0.0',
        null!
      )
      expect(result).toBeNull()
    })

    it('should return null when supported version is empty string', () => {
      const result = checkVersionCompatibility('comfyui_version', '1.0.0', '')
      expect(result).toBeNull()
    })

    it('should return null when supported version is whitespace only', () => {
      const result = checkVersionCompatibility(
        'comfyui_version',
        '1.0.0',
        '   '
      )
      expect(result).toBeNull()
    })

    describe('version compatibility checks', () => {
      it('should return null when version satisfies >= requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '2.0.0',
          '>=1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return null when version exactly matches requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.0.0',
          '1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return null when version satisfies ^ requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.2.3',
          '^1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return null when version satisfies ~ requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.0.5',
          '~1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return null when version satisfies range requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.5.0',
          '1.0.0 - 2.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return conflict when version does not satisfy >= requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '0.9.0',
          '>=1.0.0'
        )
        expect(result).toEqual({
          type: 'comfyui_version',
          current_value: '0.9.0',
          required_value: '>=1.0.0'
        })
      })

      it('should return conflict when version does not satisfy ^ requirement', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '2.0.0',
          '^1.0.0'
        )
        expect(result).toEqual({
          type: 'comfyui_version',
          current_value: '2.0.0',
          required_value: '^1.0.0'
        })
      })

      it('should return conflict when version is outside range', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '3.0.0',
          '1.0.0 - 2.0.0'
        )
        expect(result).toEqual({
          type: 'comfyui_version',
          current_value: '3.0.0',
          required_value: '1.0.0 - 2.0.0'
        })
      })
    })

    describe('version cleaning', () => {
      it('should handle versions with v prefix', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          'v1.0.0',
          '>=1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should handle versions with pre-release tags', () => {
        // Pre-release versions have specific semver rules
        // 1.0.0-alpha satisfies >=1.0.0-alpha but not >=1.0.0
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.0.0-alpha',
          '>=1.0.0-alpha'
        )
        expect(result).toBeNull()

        // This should fail because pre-release < stable
        const result2 = checkVersionCompatibility(
          'comfyui_version',
          '1.0.0-alpha',
          '>=1.0.0'
        )
        expect(result2).toEqual({
          type: 'comfyui_version',
          current_value: '1.0.0-alpha',
          required_value: '>=1.0.0'
        })
      })

      it('should handle versions with build metadata', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.0.0+build123',
          '>=1.0.0'
        )
        expect(result).toBeNull()
      })

      it('should handle malformed versions gracefully', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          'not-a-version',
          '>=1.0.0'
        )
        expect(result).toEqual({
          type: 'comfyui_version',
          current_value: 'not-a-version',
          required_value: '>=1.0.0'
        })
      })
    })

    describe('different conflict types', () => {
      it('should handle comfyui_version type', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '0.5.0',
          '>=1.0.0'
        )
        expect(result?.type).toBe('comfyui_version')
      })

      it('should handle frontend_version type', () => {
        const result = checkVersionCompatibility(
          'frontend_version',
          '0.5.0',
          '>=1.0.0'
        )
        expect(result?.type).toBe('frontend_version')
      })
    })

    describe('complex version ranges', () => {
      it('should handle OR conditions with ||', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.5.0',
          '>=1.0.0 <2.0.0 || >=3.0.0'
        )
        expect(result).toBeNull()
      })

      it('should handle multiple constraints', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '1.5.0',
          '>=1.0.0 <2.0.0'
        )
        expect(result).toBeNull()
      })

      it('should return conflict when no constraints are met', () => {
        const result = checkVersionCompatibility(
          'comfyui_version',
          '2.5.0',
          '>=1.0.0 <2.0.0 || >=3.0.0 <4.0.0'
        )
        expect(result).toEqual({
          type: 'comfyui_version',
          current_value: '2.5.0',
          required_value: '>=1.0.0 <2.0.0 || >=3.0.0 <4.0.0'
        })
      })
    })
  })

  describe('getFrontendVersion', () => {
    let originalEnv: string | undefined

    beforeEach(() => {
      originalEnv = import.meta.env.VITE_APP_VERSION
      mockConfig.app_version = '1.24.0-1'
    })

    afterEach(() => {
      if (originalEnv !== undefined) {
        import.meta.env.VITE_APP_VERSION = originalEnv
      } else {
        delete import.meta.env.VITE_APP_VERSION
      }
    })

    it('should return app_version from config when available', () => {
      const version = getFrontendVersion()
      expect(version).toBe('1.24.0-1')
    })

    it('should fallback to VITE_APP_VERSION when app_version is not available', () => {
      mockConfig.app_version = undefined
      import.meta.env.VITE_APP_VERSION = '2.0.0'

      const version = getFrontendVersion()
      expect(version).toBe('2.0.0')
    })

    it('should return undefined when no version is available', () => {
      mockConfig.app_version = undefined
      delete import.meta.env.VITE_APP_VERSION

      const version = getFrontendVersion()
      expect(version).toBeUndefined()
    })
  })
})
