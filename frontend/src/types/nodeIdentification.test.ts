import { describe, expect, it } from 'vitest'

import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  createNodeExecutionId,
  createNodeLocatorId,
  getAncestorExecutionIds,
  getParentExecutionIds,
  isNodeExecutionId,
  isNodeLocatorId,
  parseNodeExecutionId,
  parseNodeLocatorId
} from '@/types/nodeIdentification'
import type { NodeLocatorId } from '@/types/nodeIdentification'

describe('nodeIdentification', () => {
  describe('NodeLocatorId', () => {
    const validUuid = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    const validNodeId = '123'
    const validNodeLocatorId = `${validUuid}:${validNodeId}` as NodeLocatorId

    describe('isNodeLocatorId', () => {
      it('should return true for valid NodeLocatorId', () => {
        expect(isNodeLocatorId(validNodeLocatorId)).toBe(true)
        expect(isNodeLocatorId(`${validUuid}:456`)).toBe(true)
        expect(isNodeLocatorId(`${validUuid}:node_1`)).toBe(true)
        // Simple node IDs (root graph)
        expect(isNodeLocatorId('123')).toBe(true)
        expect(isNodeLocatorId('node_1')).toBe(true)
        expect(isNodeLocatorId('5')).toBe(true)
      })

      it('should return false for invalid formats', () => {
        expect(isNodeLocatorId('123:456')).toBe(false) // No UUID in first part
        expect(isNodeLocatorId('not-a-uuid:123')).toBe(false)
        expect(isNodeLocatorId('')).toBe(false) // Empty string
        expect(isNodeLocatorId(':123')).toBe(false) // Empty UUID
        expect(isNodeLocatorId(`${validUuid}:`)).toBe(false) // Empty node ID
        expect(isNodeLocatorId(`${validUuid}:123:456`)).toBe(false) // Too many parts
        expect(isNodeLocatorId(123)).toBe(false) // Not a string
        expect(isNodeLocatorId(null)).toBe(false)
        expect(isNodeLocatorId(undefined)).toBe(false)
      })

      it('should validate UUID format correctly', () => {
        // Valid UUID formats
        expect(
          isNodeLocatorId('00000000-0000-0000-0000-000000000000:123')
        ).toBe(true)
        expect(
          isNodeLocatorId('A1B2C3D4-E5F6-7890-ABCD-EF1234567890:123')
        ).toBe(true)

        // Invalid UUID formats
        expect(isNodeLocatorId('00000000-0000-0000-0000-00000000000:123')).toBe(
          false
        ) // Too short
        expect(
          isNodeLocatorId('00000000-0000-0000-0000-0000000000000:123')
        ).toBe(false) // Too long
        expect(
          isNodeLocatorId('00000000_0000_0000_0000_000000000000:123')
        ).toBe(false) // Wrong separator
        expect(
          isNodeLocatorId('g0000000-0000-0000-0000-000000000000:123')
        ).toBe(false) // Invalid hex
      })
    })

    describe('parseNodeLocatorId', () => {
      it('should parse valid NodeLocatorId', () => {
        const result = parseNodeLocatorId(validNodeLocatorId)
        expect(result).toEqual({
          subgraphUuid: validUuid,
          localNodeId: 123
        })
      })

      it('should handle string node IDs', () => {
        const stringNodeId = `${validUuid}:node_1`
        const result = parseNodeLocatorId(stringNodeId)
        expect(result).toEqual({
          subgraphUuid: validUuid,
          localNodeId: 'node_1'
        })
      })

      it('should handle simple node IDs (root graph)', () => {
        const result = parseNodeLocatorId('123')
        expect(result).toEqual({
          subgraphUuid: null,
          localNodeId: 123
        })

        const stringResult = parseNodeLocatorId('node_1')
        expect(stringResult).toEqual({
          subgraphUuid: null,
          localNodeId: 'node_1'
        })
      })

      it('should return null for invalid formats', () => {
        expect(parseNodeLocatorId('123:456')).toBeNull() // No UUID in first part
        expect(parseNodeLocatorId('')).toBeNull()
      })
    })

    describe('createNodeLocatorId', () => {
      it('should create NodeLocatorId from components', () => {
        const result = createNodeLocatorId(validUuid, 123)
        expect(result).toBe(validNodeLocatorId)
        expect(isNodeLocatorId(result)).toBe(true)
      })

      it('should handle string node IDs', () => {
        const result = createNodeLocatorId(validUuid, 'node_1')
        expect(result).toBe(`${validUuid}:node_1`)
        expect(isNodeLocatorId(result)).toBe(true)
      })
    })
  })

  describe('NodeExecutionId', () => {
    describe('isNodeExecutionId', () => {
      it('should return true for execution IDs', () => {
        expect(isNodeExecutionId('123:456')).toBe(true)
        expect(isNodeExecutionId('123:456:789')).toBe(true)
        expect(isNodeExecutionId('node_1:node_2')).toBe(true)
      })

      it('should return false for non-execution IDs', () => {
        expect(isNodeExecutionId('123')).toBe(false)
        expect(isNodeExecutionId('node_1')).toBe(false)
        expect(isNodeExecutionId('')).toBe(false)
        expect(isNodeExecutionId(123)).toBe(false)
        expect(isNodeExecutionId(null)).toBe(false)
        expect(isNodeExecutionId(undefined)).toBe(false)
      })
    })

    describe('parseNodeExecutionId', () => {
      it('should parse execution IDs correctly', () => {
        expect(parseNodeExecutionId('123:456')).toEqual([123, 456])
        expect(parseNodeExecutionId('123:456:789')).toEqual([123, 456, 789])
        expect(parseNodeExecutionId('node_1:node_2')).toEqual([
          'node_1',
          'node_2'
        ])
        expect(parseNodeExecutionId('123:node_2:456')).toEqual([
          123,
          'node_2',
          456
        ])
      })

      it('should return null for non-execution IDs', () => {
        expect(parseNodeExecutionId('123')).toBeNull()
        expect(parseNodeExecutionId('')).toBeNull()
      })
    })

    describe('createNodeExecutionId', () => {
      it('should create execution IDs from node arrays', () => {
        expect(createNodeExecutionId([123, 456])).toBe('123:456')
        expect(createNodeExecutionId([123, 456, 789])).toBe('123:456:789')
        expect(createNodeExecutionId(['node_1', 'node_2'])).toBe(
          'node_1:node_2'
        )
        expect(createNodeExecutionId([123, 'node_2', 456])).toBe(
          '123:node_2:456'
        )
      })

      it('should handle single node ID', () => {
        const result = createNodeExecutionId([123])
        expect(result).toBe('123')
        // Single node IDs are not execution IDs
        expect(isNodeExecutionId(result)).toBe(false)
      })

      it('should handle empty array', () => {
        expect(createNodeExecutionId([])).toBe('')
      })
    })
  })

  describe('Integration tests', () => {
    it('should round-trip NodeLocatorId correctly', () => {
      const uuid = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
      const nodeId: NodeId = 123

      const locatorId = createNodeLocatorId(uuid, nodeId)
      const parsed = parseNodeLocatorId(locatorId)

      expect(parsed).toBeTruthy()
      expect(parsed!.subgraphUuid).toBe(uuid)
      expect(parsed!.localNodeId).toBe(nodeId)
    })

    it('should round-trip NodeExecutionId correctly', () => {
      const nodeIds: NodeId[] = [123, 'node_2', 456]

      const executionId = createNodeExecutionId(nodeIds)
      const parsed = parseNodeExecutionId(executionId)

      expect(parsed).toEqual(nodeIds)
    })
  })

  describe('getAncestorExecutionIds', () => {
    it('returns only itself for a root node', () => {
      expect(getAncestorExecutionIds('65')).toEqual(['65'])
    })

    it('returns all ancestors including self for nested IDs', () => {
      expect(getAncestorExecutionIds('65:70')).toEqual(['65', '65:70'])
      expect(getAncestorExecutionIds('65:70:63')).toEqual([
        '65',
        '65:70',
        '65:70:63'
      ])
    })
  })

  describe('getParentExecutionIds', () => {
    it('returns empty for a root node', () => {
      expect(getParentExecutionIds('65')).toEqual([])
    })

    it('returns all ancestors excluding self for nested IDs', () => {
      expect(getParentExecutionIds('65:70')).toEqual(['65'])
      expect(getParentExecutionIds('65:70:63')).toEqual(['65', '65:70'])
    })
  })
})
