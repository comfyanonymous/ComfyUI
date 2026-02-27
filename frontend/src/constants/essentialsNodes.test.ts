import { describe, expect, it } from 'vitest'

import {
  ESSENTIALS_CATEGORIES,
  ESSENTIALS_CATEGORY_CANONICAL,
  ESSENTIALS_CATEGORY_MAP,
  ESSENTIALS_NODES,
  TOOLKIT_BLUEPRINT_MODULES,
  TOOLKIT_NOVEL_NODE_NAMES
} from './essentialsNodes'

describe('essentialsNodes', () => {
  it('has no duplicate node names across categories', () => {
    const seen = new Map<string, string>()
    for (const [category, nodes] of Object.entries(ESSENTIALS_NODES)) {
      for (const node of nodes) {
        expect(
          seen.has(node),
          `"${node}" duplicated in "${category}" and "${seen.get(node)}"`
        ).toBe(false)
        seen.set(node, category)
      }
    }
  })

  it('ESSENTIALS_CATEGORY_MAP covers every node in ESSENTIALS_NODES', () => {
    for (const [category, nodes] of Object.entries(ESSENTIALS_NODES)) {
      for (const node of nodes) {
        expect(ESSENTIALS_CATEGORY_MAP[node]).toBe(category)
      }
    }
  })

  it('TOOLKIT_NOVEL_NODE_NAMES excludes basics nodes', () => {
    for (const basicNode of ESSENTIALS_NODES.basics) {
      expect(TOOLKIT_NOVEL_NODE_NAMES.has(basicNode)).toBe(false)
    }
  })

  it('TOOLKIT_NOVEL_NODE_NAMES excludes SubgraphBlueprint-prefixed nodes', () => {
    for (const name of TOOLKIT_NOVEL_NODE_NAMES) {
      expect(name.startsWith('SubgraphBlueprint.')).toBe(false)
    }
  })

  it('ESSENTIALS_NODES keys match ESSENTIALS_CATEGORIES', () => {
    const nodeKeys = Object.keys(ESSENTIALS_NODES)
    expect(nodeKeys).toEqual([...ESSENTIALS_CATEGORIES])
  })

  it('TOOLKIT_BLUEPRINT_MODULES contains comfy_essentials', () => {
    expect(TOOLKIT_BLUEPRINT_MODULES.has('comfy_essentials')).toBe(true)
  })

  it('ESSENTIALS_CATEGORY_CANONICAL maps every category case-insensitively', () => {
    for (const category of ESSENTIALS_CATEGORIES) {
      expect(ESSENTIALS_CATEGORY_CANONICAL.get(category.toLowerCase())).toBe(
        category
      )
    }
  })
})
