import { LGraphGroup } from '@/lib/litegraph/src/LGraphGroup'
import { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { Positionable } from '@/lib/litegraph/src/interfaces'
import { describe, expect, it, beforeEach } from 'vitest'
import { flatAndCategorizeSelectedItems, searchWidgets } from './shared'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

describe('searchWidgets', () => {
  const createWidget = (
    name: string,
    type: string,
    value?: string,
    label?: string
  ): { widget: IBaseWidget } => ({
    widget: {
      name,
      type,
      value,
      label
    } as IBaseWidget
  })

  it('should return all widgets when query is empty', () => {
    const widgets = [
      createWidget('width', 'number', '100'),
      createWidget('height', 'number', '200')
    ]
    const result = searchWidgets(widgets, '')
    expect(result).toEqual(widgets)
  })

  it('should filter widgets by name, label, type, or value', () => {
    const widgets = [
      createWidget('width', 'number', '100', 'Size Control'),
      createWidget('height', 'slider', '200', 'Image Height'),
      createWidget('quality', 'text', 'high', 'Quality')
    ]

    expect(searchWidgets(widgets, 'width')).toHaveLength(1)
    expect(searchWidgets(widgets, 'slider')).toHaveLength(1)
    expect(searchWidgets(widgets, 'image')).toHaveLength(1)
  })

  it('should support fuzzy matching (e.g., "high" matches both "height" and value "high")', () => {
    const widgets = [
      createWidget('width', 'number', '100', 'Size Control'),
      createWidget('height', 'slider', '200', 'Image Height'),
      createWidget('quality', 'text', 'high', 'Quality')
    ]

    const results = searchWidgets(widgets, 'high')
    expect(results).toHaveLength(2)
    expect(results.some((r) => r.widget.name === 'height')).toBe(true)
    expect(results.some((r) => r.widget.name === 'quality')).toBe(true)
  })

  it('should handle multiple search words', () => {
    const widgets = [
      createWidget('width', 'number', '100', 'Image Width'),
      createWidget('height', 'number', '200', 'Image Height')
    ]
    const result = searchWidgets(widgets, 'image width')
    expect(result).toHaveLength(1)
    expect(result[0].widget.name).toBe('width')
  })

  it('should be case insensitive', () => {
    const widgets = [createWidget('Width', 'Number', '100', 'Image Width')]
    const result = searchWidgets(widgets, 'IMAGE width')
    expect(result).toHaveLength(1)
  })
})

describe('flatAndCategorizeSelectedItems', () => {
  let testGroup1: LGraphGroup
  let testGroup2: LGraphGroup
  let testNode1: LGraphNode
  let testNode2: LGraphNode
  let testNode3: LGraphNode

  beforeEach(() => {
    testGroup1 = new LGraphGroup('Group 1', 1)
    testGroup2 = new LGraphGroup('Group 2', 2)
    testNode1 = new LGraphNode('Node 1')
    testNode2 = new LGraphNode('Node 2')
    testNode3 = new LGraphNode('Node 3')
  })

  it('should return empty arrays for empty input', () => {
    const result = flatAndCategorizeSelectedItems([])

    expect(result.all).toEqual([])
    expect(result.nodes).toEqual([])
    expect(result.groups).toEqual([])
    expect(result.others).toEqual([])
    expect(result.nodeToParentGroup.size).toBe(0)
  })

  it('should categorize nodes', () => {
    const result = flatAndCategorizeSelectedItems([testNode1])

    expect(result.all).toEqual([testNode1])
    expect(result.nodes).toEqual([testNode1])
    expect(result.groups).toEqual([])
    expect(result.others).toEqual([])
    expect(result.nodeToParentGroup.size).toBe(0)
  })

  it('should categorize single group without children', () => {
    const result = flatAndCategorizeSelectedItems([testGroup1])

    expect(result.all).toEqual([testGroup1])
    expect(result.nodes).toEqual([])
    expect(result.groups).toEqual([testGroup1])
    expect(result.others).toEqual([])
  })

  it('should flatten group with child nodes', () => {
    testGroup1._children.add(testNode1)
    testGroup1._children.add(testNode2)

    const result = flatAndCategorizeSelectedItems([testGroup1])

    expect(result.all).toEqual([testGroup1, testNode1, testNode2])
    expect(result.nodes).toEqual([testNode1, testNode2])
    expect(result.groups).toEqual([testGroup1])
    expect(result.nodeToParentGroup.get(testNode1)).toBe(testGroup1)
    expect(result.nodeToParentGroup.get(testNode2)).toBe(testGroup1)
  })

  it('should handle nested groups', () => {
    testGroup1._children.add(testGroup2)
    testGroup2._children.add(testNode1)

    const result = flatAndCategorizeSelectedItems([testGroup1])

    expect(result.all).toEqual([testGroup1, testGroup2, testNode1])
    expect(result.nodes).toEqual([testNode1])
    expect(result.groups).toEqual([testGroup1, testGroup2])
    expect(result.nodeToParentGroup.get(testNode1)).toBe(testGroup2)
  })

  it('should handle mixed selection of nodes and groups', () => {
    testGroup1._children.add(testNode2)

    const result = flatAndCategorizeSelectedItems([
      testNode1,
      testGroup1,
      testNode3
    ])

    expect(result.all).toContain(testNode1)
    expect(result.all).toContain(testNode2)
    expect(result.all).toContain(testNode3)
    expect(result.all).toContain(testGroup1)

    expect(result.nodes).toEqual([testNode1, testNode2, testNode3])
    expect(result.groups).toEqual([testGroup1])
    expect(result.nodeToParentGroup.get(testNode1)).toBeUndefined()
    expect(result.nodeToParentGroup.get(testNode2)).toBe(testGroup1)
    expect(result.nodeToParentGroup.get(testNode3)).toBeUndefined()
  })

  it('should remove duplicate items across group and direct selection', () => {
    testGroup1._children.add(testNode1)

    const result = flatAndCategorizeSelectedItems([testGroup1, testNode1])

    expect(result.all).toEqual([testGroup1, testNode1])
    expect(result.nodes).toEqual([testNode1])
    expect(result.groups).toEqual([testGroup1])
    expect(result.nodeToParentGroup.get(testNode1)).toBe(testGroup1)
  })

  it('should handle non-node/non-group items as others', () => {
    const unknownItem = { pos: [0, 0], size: [100, 100] } as Positionable

    const result = flatAndCategorizeSelectedItems([
      testNode1,
      unknownItem,
      testGroup1
    ])

    expect(result.nodes).toEqual([testNode1])
    expect(result.groups).toEqual([testGroup1])
    expect(result.others).toEqual([unknownItem])
    expect(result.all).not.toContain(unknownItem)
  })
})
