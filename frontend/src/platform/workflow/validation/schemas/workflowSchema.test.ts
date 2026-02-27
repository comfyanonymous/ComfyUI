import fs from 'fs'
import { describe, expect, it } from 'vitest'

import {
  buildSubgraphExecutionPaths,
  validateComfyWorkflow
} from '@/platform/workflow/validation/schemas/workflowSchema'
import type { ComfyNode } from '@/platform/workflow/validation/schemas/workflowSchema'
import { defaultGraph } from '@/scripts/defaultGraph'

const WORKFLOW_DIR = 'src/platform/workflow/validation/schemas/__fixtures__'

describe('parseComfyWorkflow', () => {
  it('parses valid workflow', async () => {
    for await (const file of fs.readdirSync(WORKFLOW_DIR)) {
      if (file.endsWith('.json')) {
        const data = fs.readFileSync(`${WORKFLOW_DIR}/${file}`, 'utf-8')
        await expect(
          validateComfyWorkflow(JSON.parse(data))
        ).resolves.not.toBeNull()
      }
    }
  })

  it('workflow.nodes', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))
    workflow.nodes = undefined
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    workflow.nodes = null
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    workflow.nodes = []
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
  })

  it('workflow.version', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))
    workflow.version = undefined
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    workflow.version = '1.0.1' // Invalid format (string)
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    // 2018-2024 schema: 0.4
    workflow.version = 0.4
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
  })

  it('workflow.extra', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))
    workflow.extra = undefined
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    workflow.extra = null
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    workflow.extra = {}
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    workflow.extra = { foo: 'bar' } // Should accept extra fields.
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
  })

  it('workflow.nodes.pos', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))
    workflow.nodes[0].pos = [1, 2, 3]
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    workflow.nodes[0].pos = [1, 2]
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    // Should automatically transform the legacy format object to array.
    workflow.nodes[0].pos = { '0': 3, '1': 4 }
    let validatedWorkflow = await validateComfyWorkflow(workflow)
    // @ts-expect-error fixme ts strict error
    expect(validatedWorkflow.nodes[0].pos).toEqual([3, 4])

    workflow.nodes[0].pos = { 0: 3, 1: 4 }
    validatedWorkflow = await validateComfyWorkflow(workflow)
    // @ts-expect-error fixme ts strict error
    expect(validatedWorkflow.nodes[0].pos).toEqual([3, 4])

    // Should accept the legacy bugged format object.
    // https://github.com/Comfy-Org/ComfyUI_frontend/issues/710
    workflow.nodes[0].pos = {
      '0': 600,
      '1': 340,
      '2': 0,
      '3': 0,
      '4': 0,
      '5': 0,
      '6': 0,
      '7': 0,
      '8': 0,
      '9': 0
    }
    validatedWorkflow = await validateComfyWorkflow(workflow)
    // @ts-expect-error fixme ts strict error
    expect(validatedWorkflow.nodes[0].pos).toEqual([600, 340])
  })

  it('workflow.nodes.widget_values', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))
    workflow.nodes[0].widgets_values = ['foo', 'bar']
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    workflow.nodes[0].widgets_values = 'foo'
    await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()

    workflow.nodes[0].widgets_values = undefined
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()

    // The object format of widgets_values is used by VHS nodes to perform
    // dynamic widgets display.
    workflow.nodes[0].widgets_values = { foo: 'bar' }
    const validatedWorkflow = await validateComfyWorkflow(workflow)
    // @ts-expect-error fixme ts strict error
    expect(validatedWorkflow.nodes[0].widgets_values).toEqual({ foo: 'bar' })
  })

  it('workflow.links', async () => {
    const workflow = JSON.parse(JSON.stringify(defaultGraph))

    workflow.links = [
      [
        1, // Link id
        '100:1', // Node id of source node
        '12', // Output slot# of source node
        '100:2', // Node id of destination node
        15, // Input slot# of destination node
        'INT' // Data type
      ]
    ]
    await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
  })

  describe('workflow.nodes.properties.aux_id', () => {
    const validAuxIds = [
      'valid/valid',
      'valid-username-with-dash/valid_github-repo-name-with-underscore'
    ]
    it.each(validAuxIds)('valid aux_id: %s', async (aux_id) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.aux_id = aux_id
      await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
    })
    const invalidAuxIds = [
      'invalid spaces in username/repo',
      'invalid-chars-name-$/repo',
      'github-name/invalid spaces in repo',
      'not-both-names-with-slash'
    ]
    it.each(invalidAuxIds)('invalid aux_id: %s', async (aux_id) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.aux_id = aux_id
      await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()
    })
  })

  describe('workflow.nodes.properties.cnr_id', () => {
    const validCnrIds = ['valid', 'valid-with-dash', 'valid_with_underscores']
    it.each(validCnrIds)('valid cnr_id: %s', async (cnr_id) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.cnr_id = cnr_id
      await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
    })

    const invalidCnrIds = ['invalid cnr-id', 'invalid^cnr-id', 'invalid cnr id']
    it.each(invalidCnrIds)('invalid cnr_id: %s', async (cnr_id) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.cnr_id = cnr_id
      await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()
    })
  })

  describe('workflow.nodes.properties.ver', () => {
    const validVersionStrings = [
      // Semver
      '0.1.0',
      '0.1.0-alpha',
      '0.1.0-alpha.1',
      '1.3.321',
      // Git hash
      '080e6d4af809a46852d1c4b7ed85f06e8a3a72be',
      // Special case
      'unknown',
      // Git describe
      'v0.3.9-7-g1419dee',
      'v0.3.9-7-g1419dee-dirty'
    ]
    it.each(validVersionStrings)('valid version: %s', async (ver) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.ver = ver
      await expect(validateComfyWorkflow(workflow)).resolves.not.toBeNull()
    })

    const invalidVersionStrings = [
      // Semver
      '0.1-alpha',
      '0. 1.0',
      '0.0.0.0',
      // Git hash
      '080e6d4af809a46852d1c4b7ed85f06e8a3a72be-invalid'
    ]
    it.each(invalidVersionStrings)('invalid version: %s', async (ver) => {
      const workflow = JSON.parse(JSON.stringify(defaultGraph))
      workflow.nodes[0].properties.ver = ver
      await expect(validateComfyWorkflow(workflow)).resolves.toBeNull()
    })
  })
})

function node(id: number, type: string): ComfyNode {
  return { id, type } as ComfyNode
}

function subgraphDef(id: string, nodes: ComfyNode[]) {
  return { id, name: id, nodes, inputNode: {}, outputNode: {} }
}

describe('buildSubgraphExecutionPaths', () => {
  it('returns empty map when there are no subgraph definitions', () => {
    expect(buildSubgraphExecutionPaths([node(5, 'SomeNode')], [])).toEqual(
      new Map()
    )
  })

  it('returns empty map when no root node matches a subgraph type', () => {
    const def = subgraphDef('def-A', [])
    expect(
      buildSubgraphExecutionPaths([node(5, 'UnrelatedNode')], [def])
    ).toEqual(new Map())
  })

  it('maps a single subgraph instance to its execution path', () => {
    const def = subgraphDef('def-A', [])
    const result = buildSubgraphExecutionPaths([node(5, 'def-A')], [def])
    expect(result.get('def-A')).toEqual(['5'])
  })

  it('collects multiple instances of the same subgraph type', () => {
    const def = subgraphDef('def-A', [])
    const result = buildSubgraphExecutionPaths(
      [node(5, 'def-A'), node(10, 'def-A')],
      [def]
    )
    expect(result.get('def-A')).toEqual(['5', '10'])
  })

  it('builds nested execution paths for subgraphs within subgraphs', () => {
    const innerDef = subgraphDef('def-B', [])
    const outerDef = subgraphDef('def-A', [node(70, 'def-B')])
    const result = buildSubgraphExecutionPaths(
      [node(5, 'def-A')],
      [outerDef, innerDef]
    )
    expect(result.get('def-A')).toEqual(['5'])
    expect(result.get('def-B')).toEqual(['5:70'])
  })

  it('does not recurse infinitely on self-referential subgraph definitions', () => {
    const cyclicDef = subgraphDef('def-A', [node(70, 'def-A')])
    expect(() =>
      buildSubgraphExecutionPaths([node(5, 'def-A')], [cyclicDef])
    ).not.toThrow()
  })

  it('does not recurse infinitely on mutually cyclic subgraph definitions', () => {
    const defA = subgraphDef('def-A', [node(70, 'def-B')])
    const defB = subgraphDef('def-B', [node(80, 'def-A')])
    expect(() =>
      buildSubgraphExecutionPaths([node(5, 'def-A')], [defA, defB])
    ).not.toThrow()
  })
})
