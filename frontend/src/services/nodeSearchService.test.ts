import { describe, expect, it } from 'vitest'

import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import { NodeSearchService } from '@/services/nodeSearchService'
import { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const EXAMPLE_NODE_DEFS: ComfyNodeDefImpl[] = (
  [
    {
      input: {
        required: {
          ckpt_name: [['model1.safetensors', 'model2.ckpt'], {}]
        }
      },
      output: ['MODEL', 'CLIP', 'VAE'],
      output_is_list: [false, false, false],
      output_name: ['MODEL', 'CLIP', 'VAE'],
      name: 'CheckpointLoaderSimple',
      display_name: 'Load Checkpoint',
      description: '',
      python_module: 'nodes',
      category: 'loaders',
      output_node: false
    },
    {
      input: {
        required: {
          samples: ['LATENT'],
          batch_index: [
            'INT',
            {
              default: 0,
              min: 0,
              max: 63
            }
          ],
          length: [
            'INT',
            {
              default: 1,
              min: 1,
              max: 64
            }
          ]
        }
      },
      output: ['LATENT'],
      output_is_list: [false],
      output_name: ['LATENT'],
      name: 'LatentFromBatch',
      display_name: 'Latent From Batch',
      description: '',
      python_module: 'nodes',
      category: 'latent/batch',
      output_node: false
    }
  ] as ComfyNodeDef[]
).map((nodeDef: ComfyNodeDef) => {
  const def = new ComfyNodeDefImpl(nodeDef)
  def['postProcessSearchScores'] = (s) => s
  return def
})

const NODE_DEFS_WITH_SEARCH_ALIASES: ComfyNodeDefImpl[] = (
  [
    {
      input: { required: {} },
      output: ['MODEL'],
      output_is_list: [false],
      output_name: ['MODEL'],
      name: 'CheckpointLoaderSimple',
      display_name: 'Load Checkpoint',
      description: '',
      python_module: 'nodes',
      category: 'loaders',
      output_node: false,
      search_aliases: ['ckpt', 'model loader', 'checkpoint']
    },
    {
      input: { required: {} },
      output: ['IMAGE'],
      output_is_list: [false],
      output_name: ['IMAGE'],
      name: 'LoadImage',
      display_name: 'Load Image',
      description: '',
      python_module: 'nodes',
      category: 'loaders',
      output_node: false,
      search_aliases: ['img', 'picture']
    },
    {
      input: { required: {} },
      output: ['LATENT'],
      output_is_list: [false],
      output_name: ['LATENT'],
      name: 'VAEEncode',
      display_name: 'VAE Encode',
      description: '',
      python_module: 'nodes',
      category: 'latent',
      output_node: false
      // No search_aliases
    }
  ] as ComfyNodeDef[]
).map((nodeDef: ComfyNodeDef) => {
  const def = new ComfyNodeDefImpl(nodeDef)
  def['postProcessSearchScores'] = (s) => s
  return def
})

describe('nodeSearchService', () => {
  it('searches with input filter', () => {
    const service = new NodeSearchService(EXAMPLE_NODE_DEFS)
    const inputFilter = service.inputTypeFilter
    expect(
      service.searchNode('L', [{ filterDef: inputFilter, value: 'LATENT' }])
    ).toHaveLength(1)
    // Wildcard should match all.
    expect(
      service.searchNode('L', [{ filterDef: inputFilter, value: '*' }])
    ).toHaveLength(2)
    expect(service.searchNode('L')).toHaveLength(2)
  })

  describe('search_aliases', () => {
    it('finds nodes by search_aliases', () => {
      const service = new NodeSearchService(NODE_DEFS_WITH_SEARCH_ALIASES)
      // Search by alias
      const ckptResults = service.searchNode('ckpt')
      expect(ckptResults).toHaveLength(1)
      expect(ckptResults[0].name).toBe('CheckpointLoaderSimple')
    })

    it('finds nodes by partial alias match', () => {
      const service = new NodeSearchService(NODE_DEFS_WITH_SEARCH_ALIASES)
      // Search by partial alias "model" should match "model loader" alias
      const modelResults = service.searchNode('model')
      expect(modelResults.length).toBeGreaterThanOrEqual(1)
      expect(
        modelResults.some((r) => r.name === 'CheckpointLoaderSimple')
      ).toBe(true)
    })

    it('finds nodes by display_name when no alias matches', () => {
      const service = new NodeSearchService(NODE_DEFS_WITH_SEARCH_ALIASES)
      // "VAE" should match by display_name since there are no aliases
      const vaeResults = service.searchNode('VAE')
      expect(vaeResults).toHaveLength(1)
      expect(vaeResults[0].name).toBe('VAEEncode')
    })

    it('finds nodes by both alias and display_name', () => {
      const service = new NodeSearchService(NODE_DEFS_WITH_SEARCH_ALIASES)
      // "img" should match LoadImage by alias
      const imgResults = service.searchNode('img')
      expect(imgResults).toHaveLength(1)
      expect(imgResults[0].name).toBe('LoadImage')

      // "Load" should match both checkpoint and image loaders by display_name
      const loadResults = service.searchNode('Load')
      expect(loadResults.length).toBeGreaterThanOrEqual(2)
    })

    it('handles nodes without search_aliases', () => {
      const service = new NodeSearchService(NODE_DEFS_WITH_SEARCH_ALIASES)
      // Ensure nodes without aliases are still searchable by name/display_name
      const encodeResults = service.searchNode('Encode')
      expect(encodeResults).toHaveLength(1)
      expect(encodeResults[0].name).toBe('VAEEncode')
    })
  })
})
