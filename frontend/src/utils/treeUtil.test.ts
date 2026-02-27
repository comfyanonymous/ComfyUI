import { describe, expect, it } from 'vitest'

import type { TreeNode } from '@/types/treeExplorerTypes'
import { buildTree, sortedTree } from '@/utils/treeUtil'

describe('buildTree', () => {
  it('should handle empty folder items correctly', () => {
    const items = [
      { path: 'a/b/c/' },
      { path: 'a/b/d.txt' },
      { path: 'a/e/' },
      { path: 'f.txt' }
    ]

    const tree = buildTree(items, (item) => item.path.split('/'))

    expect(tree).toEqual({
      key: 'root',
      label: 'root',
      children: [
        {
          key: 'root/a',
          label: 'a',
          leaf: false,
          children: [
            {
              key: 'root/a/b',
              label: 'b',
              leaf: false,
              children: [
                {
                  key: 'root/a/b/c',
                  label: 'c',
                  leaf: false,
                  children: [],
                  data: { path: 'a/b/c/' }
                },
                {
                  key: 'root/a/b/d.txt',
                  label: 'd.txt',
                  leaf: true,
                  children: [],
                  data: { path: 'a/b/d.txt' }
                }
              ]
            },
            {
              key: 'root/a/e',
              label: 'e',
              leaf: false,
              children: [],
              data: { path: 'a/e/' }
            }
          ]
        },
        {
          key: 'root/f.txt',
          label: 'f.txt',
          leaf: true,
          children: [],
          data: { path: 'f.txt' }
        }
      ]
    })
  })
})

describe('sortedTree', () => {
  const createNode = (label: string, leaf = false): TreeNode => ({
    key: label,
    label,
    leaf,
    children: []
  })

  it('should return a new node instance', () => {
    const node = createNode('root')
    const result = sortedTree(node)
    expect(result).not.toBe(node)
    expect(result).toEqual(node)
  })

  it('should sort children by label', () => {
    const node: TreeNode = {
      key: 'root',
      label: 'root',
      leaf: false,
      children: [createNode('c'), createNode('a'), createNode('b')]
    }

    const result = sortedTree(node)
    expect(result.children?.map((c) => c.label)).toEqual(['a', 'b', 'c'])
  })

  describe('with groupLeaf=true', () => {
    it('should group folders before files', () => {
      const node: TreeNode = {
        key: 'root',
        label: 'root',
        children: [
          createNode('file.txt', true),
          createNode('folder1'),
          createNode('another.txt', true),
          createNode('folder2')
        ]
      }

      const result = sortedTree(node, { groupLeaf: true })
      const labels = result.children?.map((c) => c.label)
      expect(labels).toEqual(['folder1', 'folder2', 'another.txt', 'file.txt'])
    })

    it('should sort recursively', () => {
      const node: TreeNode = {
        key: 'root',
        label: 'root',
        children: [
          {
            ...createNode('folder1'),
            children: [
              createNode('z.txt', true),
              createNode('subfolder2'),
              createNode('a.txt', true),
              createNode('subfolder1')
            ]
          }
        ]
      }

      const result = sortedTree(node, { groupLeaf: true })
      const folder = result.children?.[0]
      const subLabels = folder?.children?.map((c) => c.label)
      expect(subLabels).toEqual(['subfolder1', 'subfolder2', 'a.txt', 'z.txt'])
    })
  })

  it('should handle nodes without children', () => {
    const node: TreeNode = {
      key: 'leaf',
      label: 'leaf',
      leaf: true
    }

    const result = sortedTree(node)
    expect(result).toEqual(node)
  })
})
