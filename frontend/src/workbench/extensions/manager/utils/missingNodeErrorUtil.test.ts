import { describe, expect, it } from 'vitest'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

import {
  getCnrIdFromProperties,
  getCnrIdFromNode
} from './missingNodeErrorUtil'

describe('getCnrIdFromProperties', () => {
  it('returns cnr_id when present', () => {
    expect(getCnrIdFromProperties({ cnr_id: 'my-pack' })).toBe('my-pack')
  })

  it('returns aux_id when cnr_id is absent', () => {
    expect(getCnrIdFromProperties({ aux_id: 'my-aux-pack' })).toBe(
      'my-aux-pack'
    )
  })

  it('prefers cnr_id over aux_id', () => {
    expect(
      getCnrIdFromProperties({ cnr_id: 'primary', aux_id: 'secondary' })
    ).toBe('primary')
  })

  it('returns undefined when neither is present', () => {
    expect(getCnrIdFromProperties({})).toBeUndefined()
  })

  it('returns undefined for null properties', () => {
    expect(getCnrIdFromProperties(null)).toBeUndefined()
  })

  it('returns undefined for undefined properties', () => {
    expect(getCnrIdFromProperties(undefined)).toBeUndefined()
  })

  it('returns undefined when cnr_id is not a string', () => {
    expect(getCnrIdFromProperties({ cnr_id: 123 })).toBeUndefined()
  })
})

describe('getCnrIdFromNode', () => {
  it('returns cnr_id from node properties', () => {
    const node = {
      properties: { cnr_id: 'node-pack' }
    } as unknown as LGraphNode
    expect(getCnrIdFromNode(node)).toBe('node-pack')
  })

  it('returns aux_id when cnr_id is absent', () => {
    const node = {
      properties: { aux_id: 'node-aux-pack' }
    } as unknown as LGraphNode
    expect(getCnrIdFromNode(node)).toBe('node-aux-pack')
  })

  it('prefers cnr_id over aux_id in node properties', () => {
    const node = {
      properties: { cnr_id: 'primary', aux_id: 'secondary' }
    } as unknown as LGraphNode
    expect(getCnrIdFromNode(node)).toBe('primary')
  })

  it('returns undefined when node has no cnr_id or aux_id', () => {
    const node = { properties: {} } as unknown as LGraphNode
    expect(getCnrIdFromNode(node)).toBeUndefined()
  })
})
