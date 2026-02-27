import { clamp } from 'es-toolkit/compat'
import { describe, expect } from 'vitest'

import {
  LiteGraphGlobal,
  LGraphCanvas,
  LiteGraph,
  LGraph
} from '@/lib/litegraph/src/litegraph'

import { LGraph as DirectLGraph } from '@/lib/litegraph/src/LGraph'

import { test } from './__fixtures__/testExtensions'

describe('Litegraph module', () => {
  test('contains a global export', ({ expect }) => {
    expect(LiteGraph).toBeInstanceOf(LiteGraphGlobal)
    expect(LiteGraph.LGraphCanvas).toBe(LGraphCanvas)
  })

  test('has the same structure', ({ expect }) => {
    const lgGlobal = new LiteGraphGlobal()
    expect(lgGlobal).toMatchSnapshot('minLGraph')
  })

  test('clamps values', () => {
    expect(clamp(-1.124, 13, 24)).toStrictEqual(13)
    expect(clamp(Infinity, 18, 29)).toStrictEqual(29)
  })
})

describe('Import order dependency', () => {
  test('Imports reference the same types', ({ expect }) => {
    // Both imports should reference the same LGraph class
    expect(LiteGraph.LGraph).toBe(DirectLGraph)
    expect(LiteGraph.LGraph).toBe(LGraph)
  })
})
