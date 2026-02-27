import { describe, expect, it } from 'vitest'

import {
  MAX_DRAFTS,
  createDraftCacheState,
  mostRecentDraftPath,
  moveDraft,
  removeDraft,
  touchEntry,
  upsertDraft
} from './draftCache'
import type { WorkflowDraftSnapshot } from './draftCache'

function createSnapshot(name: string): WorkflowDraftSnapshot {
  return {
    data: JSON.stringify({ name }),
    updatedAt: Date.now(),
    name,
    isTemporary: true
  }
}

describe('draftCache helpers', () => {
  it('touchEntry moves path to end', () => {
    expect(touchEntry(['a', 'b'], 'a')).toEqual(['b', 'a'])
    expect(touchEntry(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
  })

  it('upsertDraft stores snapshot and applies LRU', () => {
    let state = createDraftCacheState()
    for (let i = 0; i < MAX_DRAFTS; i++) {
      const path = `workflows/Draft${i}.json`
      state = upsertDraft(state, path, createSnapshot(String(i)))
    }

    expect(Object.keys(state.drafts).length).toBe(MAX_DRAFTS)

    state = upsertDraft(state, 'workflows/New.json', createSnapshot('new'))
    expect(Object.keys(state.drafts).length).toBe(MAX_DRAFTS)
    expect(state.drafts).not.toHaveProperty('workflows/Draft0.json')
    expect(state.order[state.order.length - 1]).toBe('workflows/New.json')
  })

  it('removeDraft clears entry and order', () => {
    const state = upsertDraft(
      createDraftCacheState(),
      'workflows/test.json',
      createSnapshot('test')
    )

    const nextState = removeDraft(state, 'workflows/test.json')
    expect(nextState.drafts).toEqual({})
    expect(nextState.order).toEqual([])
  })

  it('moveDraft renames entry and updates order', () => {
    const state = upsertDraft(
      createDraftCacheState(),
      'workflows/old.json',
      createSnapshot('old')
    )

    const nextState = moveDraft(
      state,
      'workflows/old.json',
      'workflows/new.json',
      'new'
    )
    expect(nextState.drafts).not.toHaveProperty('workflows/old.json')
    expect(nextState.drafts['workflows/new.json']?.name).toBe('new')
    expect(nextState.order).toEqual(['workflows/new.json'])
  })

  it('mostRecentDraftPath returns last entry', () => {
    const state = createDraftCacheState({}, ['a', 'b', 'c'])
    expect(mostRecentDraftPath(state.order)).toBe('c')
    expect(mostRecentDraftPath([])).toBeNull()
  })
})
