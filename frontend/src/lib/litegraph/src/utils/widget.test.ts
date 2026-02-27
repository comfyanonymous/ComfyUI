import { describe, expect, test } from 'vitest'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IWidgetOptions } from '@/lib/litegraph/src/litegraph'
import {
  getWidgetStep,
  resolveNodeRootGraphId
} from '@/lib/litegraph/src/litegraph'

describe('getWidgetStep', () => {
  test('should return step2 when available', () => {
    const options: IWidgetOptions<unknown> = {
      step2: 0.5,
      step: 20
    }

    expect(getWidgetStep(options)).toBe(0.5)
  })

  test('should calculate from step when step2 is not available', () => {
    const options: IWidgetOptions<unknown> = {
      step: 20
    }

    expect(getWidgetStep(options)).toBe(2) // 20 * 0.1 = 2
  })

  test('should use default step value of 10 when neither step2 nor step is provided', () => {
    const options: IWidgetOptions<unknown> = {}

    expect(getWidgetStep(options)).toBe(1) // 10 * 0.1 = 1
  })
  // Zero value is not allowed for step, fallback to 1.
  test('should handle zero values correctly', () => {
    const optionsWithZeroStep2: IWidgetOptions<unknown> = {
      step2: 0,
      step: 20
    }

    expect(getWidgetStep(optionsWithZeroStep2)).toBe(2)

    const optionsWithZeroStep: IWidgetOptions<unknown> = {
      step: 0
    }

    expect(getWidgetStep(optionsWithZeroStep)).toBe(1)
  })
})

type GraphIdNode = Pick<LGraphNode, 'graph'>

describe('resolveNodeRootGraphId', () => {
  test('returns node rootGraph id when node belongs to a graph', () => {
    const node = {
      graph: {
        rootGraph: {
          id: 'subgraph-root-id'
        }
      }
    } as GraphIdNode

    expect(resolveNodeRootGraphId(node)).toBe('subgraph-root-id')
  })

  test('returns fallback graph id when node graph is missing', () => {
    const node = {
      graph: null
    } as GraphIdNode

    expect(resolveNodeRootGraphId(node, 'app-root-id')).toBe('app-root-id')
  })
})
