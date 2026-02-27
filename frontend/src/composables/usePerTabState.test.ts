import { ref } from 'vue'
import { describe, expect, it } from 'vitest'

import { usePerTabState } from './usePerTabState'

type TabId = 'a' | 'b' | 'c'

describe('usePerTabState', () => {
  function setup(initialTab: TabId = 'a') {
    const selectedTab = ref<TabId>(initialTab)
    const stateByTab = ref<Record<TabId, string[]>>({
      a: [],
      b: [],
      c: []
    })
    const state = usePerTabState(selectedTab, stateByTab)
    return { selectedTab, stateByTab, state }
  }

  it('should return state for the current tab', () => {
    const { selectedTab, stateByTab, state } = setup()

    stateByTab.value.a = ['key1', 'key2']
    stateByTab.value.b = ['key3']

    expect(state.value).toEqual(['key1', 'key2'])

    selectedTab.value = 'b'
    expect(state.value).toEqual(['key3'])
  })

  it('should set state only for the current tab', () => {
    const { stateByTab, state } = setup()

    state.value = ['new-key1', 'new-key2']

    expect(stateByTab.value.a).toEqual(['new-key1', 'new-key2'])
    expect(stateByTab.value.b).toEqual([])
    expect(stateByTab.value.c).toEqual([])
  })

  it('should preserve state when switching tabs', () => {
    const { selectedTab, stateByTab, state } = setup()

    state.value = ['a-key']
    selectedTab.value = 'b'
    state.value = ['b-key']
    selectedTab.value = 'c'
    state.value = ['c-key']

    expect(stateByTab.value.a).toEqual(['a-key'])
    expect(stateByTab.value.b).toEqual(['b-key'])
    expect(stateByTab.value.c).toEqual(['c-key'])

    selectedTab.value = 'a'
    expect(state.value).toEqual(['a-key'])
  })

  it('should not share state between tabs', () => {
    const { selectedTab, state } = setup()

    state.value = ['only-a']

    selectedTab.value = 'b'
    expect(state.value).toEqual([])

    selectedTab.value = 'c'
    expect(state.value).toEqual([])

    selectedTab.value = 'a'
    expect(state.value).toEqual(['only-a'])
  })
})
