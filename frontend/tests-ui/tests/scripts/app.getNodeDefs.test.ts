import { beforeEach, describe, expect, test, vi } from 'vitest'

import { st } from '@/i18n'
import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'

vi.mock('@/scripts/api', () => ({
  api: {
    getNodeDefs: vi.fn(),
    apiURL: vi.fn((path: string) => path),
    addEventListener: vi.fn(),
    getUserData: vi.fn(),
    storeUserData: vi.fn()
  }
}))

vi.mock('@/i18n', () => ({
  st: vi.fn((_key: string, fallback: string) => fallback),
  t: vi.fn((key: string) => key),
  te: vi.fn(() => false)
}))

describe('ComfyApp.getNodeDefs', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('should use object info display_name when available', async () => {
    const mockNodeDefs: Record<string, ComfyNodeDefV1> = {
      TestNode: {
        name: 'TestNode',
        display_name: 'Custom Display Name',
        category: 'test',
        description: 'Test description',
        input: {},
        output: [],
        output_node: false,
        python_module: 'test.module'
      }
    }

    vi.mocked(api.getNodeDefs).mockResolvedValue(mockNodeDefs)

    const result = await comfyApp.getNodeDefs()

    expect(result.TestNode.display_name).toBe('Custom Display Name')
  })

  test('should fall back to name when display_name is missing', async () => {
    const mockNodeDefs: Record<string, ComfyNodeDefV1> = {
      TestNode: {
        name: 'TestNode',
        display_name: '',
        category: 'test',
        description: 'Test description',
        input: {},
        output: [],
        output_node: false,
        python_module: 'test.module'
      }
    }

    vi.mocked(api.getNodeDefs).mockResolvedValue(mockNodeDefs)

    const result = await comfyApp.getNodeDefs()

    // When display_name is empty, should fall back to name
    expect(result.TestNode.display_name).toBe('TestNode')
  })

  test('should prioritize translation over object info display_name', async () => {
    const mockNodeDefs: Record<string, ComfyNodeDefV1> = {
      TestNode: {
        name: 'TestNode',
        display_name: 'Object Info Display Name',
        category: 'test',
        description: 'Test description',
        input: {},
        output: [],
        output_node: false,
        python_module: 'test.module'
      }
    }

    vi.mocked(api.getNodeDefs).mockResolvedValue(mockNodeDefs)
    // Mock st to return a translation instead of fallback
    vi.mocked(st).mockReturnValue('Translated Display Name')

    const result = await comfyApp.getNodeDefs()

    expect(result.TestNode.display_name).toBe('Translated Display Name')
  })
})
