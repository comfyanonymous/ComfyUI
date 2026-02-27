import { describe, expect, it } from 'vitest'

import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import {
  getOrderedInputSpecs,
  sortWidgetValuesByInputOrder
} from '@/workbench/utils/nodeDefOrderingUtil'

describe('nodeDefOrderingUtil', () => {
  describe('getOrderedInputSpecs', () => {
    it('should maintain order when no input_order is specified', () => {
      const nodeDef: ComfyNodeDef = {
        name: 'TestNode',
        display_name: 'Test Node',
        category: 'test',
        python_module: 'test',
        description: 'Test node',
        output_node: false,
        input: {
          required: {
            image: ['IMAGE', {}],
            seed: ['INT', { default: 0 }],
            steps: ['INT', { default: 20 }]
          }
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(nodeDef)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      // Should maintain Object.values order when no input_order
      const names = result.map((spec) => spec.name)
      expect(names).toEqual(['image', 'seed', 'steps'])
    })

    it('should sort inputs according to input_order', () => {
      const nodeDef: ComfyNodeDef = {
        name: 'TestNode',
        display_name: 'Test Node',
        category: 'test',
        python_module: 'test',
        description: 'Test node',
        output_node: false,
        input: {
          required: {
            image: ['IMAGE', {}],
            seed: ['INT', { default: 0 }],
            steps: ['INT', { default: 20 }]
          }
        },
        input_order: {
          required: ['steps', 'seed', 'image']
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(nodeDef)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      const names = result.map((spec) => spec.name)
      expect(names).toEqual(['steps', 'seed', 'image'])
    })

    it('should handle missing inputs in input_order gracefully', () => {
      const nodeDef: ComfyNodeDef = {
        name: 'TestNode',
        display_name: 'Test Node',
        category: 'test',
        python_module: 'test',
        description: 'Test node',
        output_node: false,
        input: {
          required: {
            image: ['IMAGE', {}],
            seed: ['INT', { default: 0 }],
            steps: ['INT', { default: 20 }]
          }
        },
        input_order: {
          required: ['steps', 'nonexistent', 'seed']
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(nodeDef)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      // Should skip nonexistent and include image at the end
      const names = result.map((spec) => spec.name)
      expect(names).toEqual(['steps', 'seed', 'image'])
    })

    it('should handle inputs not in input_order', () => {
      const nodeDef: ComfyNodeDef = {
        name: 'TestNode',
        display_name: 'Test Node',
        category: 'test',
        python_module: 'test',
        description: 'Test node',
        output_node: false,
        input: {
          required: {
            image: ['IMAGE', {}],
            seed: ['INT', { default: 0 }],
            steps: ['INT', { default: 20 }],
            cfg: ['FLOAT', { default: 7.0 }]
          }
        },
        input_order: {
          required: ['steps', 'seed']
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(nodeDef)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      // Should have ordered ones first, then remaining
      const names = result.map((spec) => spec.name)
      expect(names).toEqual(['steps', 'seed', 'image', 'cfg'])
    })

    it('should handle both required and optional inputs', () => {
      const nodeDef: ComfyNodeDef = {
        name: 'TestNode',
        display_name: 'Test Node',
        category: 'test',
        python_module: 'test',
        description: 'Test node',
        output_node: false,
        input: {
          required: {
            image: ['IMAGE', {}],
            seed: ['INT', { default: 0 }]
          },
          optional: {
            mask: ['MASK', {}],
            strength: ['FLOAT', { default: 1.0 }]
          }
        },
        input_order: {
          required: ['seed', 'image'],
          optional: ['strength', 'mask']
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(nodeDef)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      const names = result.map((spec) => spec.name)
      const optionalFlags = result.map((spec) => spec.isOptional)

      expect(names).toEqual(['seed', 'image', 'strength', 'mask'])
      expect(optionalFlags).toEqual([false, false, true, true])
    })

    it('should work with real KSampler node example', () => {
      // Simulating different backend orderings
      const kSamplerDefBackendA: ComfyNodeDef = {
        name: 'KSampler',
        display_name: 'KSampler',
        category: 'sampling',
        python_module: 'nodes',
        description: 'KSampler node',
        output_node: false,
        input: {
          required: {
            // Alphabetical order from backend A
            cfg: ['FLOAT', { default: 8, min: 0, max: 100 }],
            denoise: ['FLOAT', { default: 1, min: 0, max: 1 }],
            latent_image: ['LATENT', {}],
            model: ['MODEL', {}],
            negative: ['CONDITIONING', {}],
            positive: ['CONDITIONING', {}],
            sampler_name: [['euler', 'euler_cfg_pp'], {}],
            scheduler: [['simple', 'sgm_uniform'], {}],
            seed: ['INT', { default: 0, min: 0, max: Number.MAX_SAFE_INTEGER }],
            steps: ['INT', { default: 20, min: 1, max: 10000 }]
          }
        },
        input_order: {
          required: [
            'model',
            'seed',
            'steps',
            'cfg',
            'sampler_name',
            'scheduler',
            'positive',
            'negative',
            'latent_image',
            'denoise'
          ]
        }
      }

      const nodeDefImpl = new ComfyNodeDefImpl(kSamplerDefBackendA)
      const result = getOrderedInputSpecs(nodeDefImpl, nodeDefImpl.inputs)

      const names = result.map((spec) => spec.name)
      // Should follow input_order, not alphabetical
      expect(names).toEqual([
        'model',
        'seed',
        'steps',
        'cfg',
        'sampler_name',
        'scheduler',
        'positive',
        'negative',
        'latent_image',
        'denoise'
      ])
    })
  })

  describe('sortWidgetValuesByInputOrder', () => {
    it('should reorder widget values to match input_order', () => {
      const widgetValues = [0, 'model_ref', 5, 1]
      const currentWidgetOrder = ['momentum', 'model', 'norm_threshold', 'eta']
      const correctOrder = ['model', 'eta', 'norm_threshold', 'momentum']

      const result = sortWidgetValuesByInputOrder(
        widgetValues,
        currentWidgetOrder,
        correctOrder
      )

      expect(result).toEqual(['model_ref', 1, 5, 0])
    })

    it('should handle missing widgets in input_order', () => {
      const widgetValues = [1, 2, 3, 4]
      const currentWidgetOrder = ['a', 'b', 'c', 'd']
      const inputOrder = ['b', 'd'] // Only partial order

      const result = sortWidgetValuesByInputOrder(
        widgetValues,
        currentWidgetOrder,
        inputOrder
      )

      // b=2, d=4, then a=1, c=3
      expect(result).toEqual([2, 4, 1, 3])
    })

    it('should handle extra widget values', () => {
      const widgetValues = [1, 2, 3, 4, 5] // More values than names
      const currentWidgetOrder = ['a', 'b', 'c']
      const inputOrder = ['c', 'a', 'b']

      const result = sortWidgetValuesByInputOrder(
        widgetValues,
        currentWidgetOrder,
        inputOrder
      )

      // c=3, a=1, b=2, then extras 4, 5
      expect(result).toEqual([3, 1, 2, 4, 5])
    })

    it('should return unchanged when no input_order', () => {
      const widgetValues = [1, 2, 3]
      const currentWidgetOrder = ['a', 'b', 'c']
      const inputOrder: string[] = []

      const result = sortWidgetValuesByInputOrder(
        widgetValues,
        currentWidgetOrder,
        inputOrder
      )

      expect(result).toEqual([1, 2, 3])
    })
  })
})
