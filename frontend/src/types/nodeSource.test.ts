import { describe, expect, it } from 'vitest'

import {
  NodeSourceType,
  getEssentialsCategory,
  getNodeSource
} from '@/types/nodeSource'

describe('getNodeSource', () => {
  it('should return UNKNOWN_NODE_SOURCE when python_module is undefined', () => {
    const result = getNodeSource(undefined)
    expect(result).toEqual({
      type: NodeSourceType.Unknown,
      className: 'comfy-unknown',
      displayText: 'Unknown',
      badgeText: '?'
    })
  })

  it('should identify core nodes from nodes module', () => {
    const result = getNodeSource('nodes.some_module')
    expect(result).toEqual({
      type: NodeSourceType.Core,
      className: 'comfy-core',
      displayText: 'Comfy Core',
      badgeText: '🦊'
    })
  })

  it('should identify core nodes from comfy_extras module', () => {
    const result = getNodeSource('comfy_extras.some_module')
    expect(result).toEqual({
      type: NodeSourceType.Core,
      className: 'comfy-core',
      displayText: 'Comfy Core',
      badgeText: '🦊'
    })
  })

  it('should identify core nodes from comfy_api_nodes module', () => {
    const result = getNodeSource('comfy_api_nodes.some_module')
    expect(result).toEqual({
      type: NodeSourceType.Core,
      className: 'comfy-core',
      displayText: 'Comfy Core',
      badgeText: '🦊'
    })
  })

  it('should identify custom nodes and format their names', () => {
    const result = getNodeSource('custom_nodes.ComfyUI-Example')
    expect(result).toEqual({
      type: NodeSourceType.CustomNodes,
      className: 'comfy-custom-nodes',
      displayText: 'Example',
      badgeText: 'Example'
    })
  })

  it('should identify custom nodes with version and format their names', () => {
    const result = getNodeSource('custom_nodes.ComfyUI-Example@1.0.0')
    expect(result).toEqual({
      type: NodeSourceType.CustomNodes,
      className: 'comfy-custom-nodes',
      displayText: 'Example',
      badgeText: 'Example'
    })
  })

  it('should return UNKNOWN_NODE_SOURCE for unrecognized modules', () => {
    const result = getNodeSource('unknown_module.something')
    expect(result).toEqual({
      type: NodeSourceType.Unknown,
      className: 'comfy-unknown',
      displayText: 'Unknown',
      badgeText: '?'
    })
  })

  describe('essentials nodes', () => {
    it('should identify essentials nodes when essentials_category is set', () => {
      const result = getNodeSource('nodes.some_module', 'Image')
      expect(result.type).toBe(NodeSourceType.Essentials)
      expect(result.className).toBe('comfy-essentials')
    })

    it('should identify essentials nodes from custom_nodes module', () => {
      const result = getNodeSource(
        'custom_nodes.ComfyUI-Example@1.0.0',
        'Video',
        'SomeNode'
      )
      expect(result.type).toBe(NodeSourceType.Essentials)
      expect(result.className).toBe('comfy-essentials')
      expect(result.displayText).toBe('Example')
    })

    it('should not identify nodes without essentials_category as essentials', () => {
      // Use a node name not in the mock list
      const result = getNodeSource(
        'nodes.some_module',
        undefined,
        'UnknownNode'
      )
      expect(result.type).toBe(NodeSourceType.Core)
    })

    it('should identify nodes from mock list as essentials', () => {
      const result = getNodeSource('nodes.some_module', undefined, 'LoadImage')
      expect(result.type).toBe(NodeSourceType.Essentials)
    })

    it('should normalize title-cased backend categories to canonical form', () => {
      const result = getNodeSource(
        'nodes.some_module',
        'Image Generation',
        'SomeNode'
      )
      expect(result.type).toBe(NodeSourceType.Essentials)
      expect(getEssentialsCategory('SomeNode', 'Image Generation')).toBe(
        'image generation'
      )
    })
  })

  describe('getEssentialsCategory', () => {
    it('should normalize title-cased essentials_category to canonical form', () => {
      expect(getEssentialsCategory('SomeNode', 'Image Generation')).toBe(
        'image generation'
      )
      expect(getEssentialsCategory('SomeNode', '3d')).toBe('3D')
      expect(getEssentialsCategory('SomeNode', '3D')).toBe('3D')
    })

    it('should fall back to ESSENTIALS_CATEGORY_MAP when no category provided', () => {
      expect(getEssentialsCategory('LoadImage')).toBe('basics')
    })

    it('should return undefined for unknown node without category', () => {
      expect(getEssentialsCategory('UnknownNode')).toBeUndefined()
    })
  })

  describe('blueprint nodes', () => {
    it('should identify blueprint nodes', () => {
      const result = getNodeSource('blueprint.my_blueprint')
      expect(result).toEqual({
        type: NodeSourceType.Blueprint,
        className: 'blueprint',
        displayText: 'Blueprint',
        badgeText: 'bp'
      })
    })
  })
})
