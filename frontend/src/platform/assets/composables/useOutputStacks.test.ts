import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type * as OutputAssetUtil from '@/platform/assets/utils/outputAssetUtil'
import { useOutputStacks } from '@/platform/assets/composables/useOutputStacks'

const mocks = vi.hoisted(() => ({
  resolveOutputAssetItems: vi.fn()
}))

vi.mock('@/platform/assets/utils/outputAssetUtil', async (importOriginal) => {
  const actual = await importOriginal<typeof OutputAssetUtil>()
  return {
    ...actual,
    resolveOutputAssetItems: mocks.resolveOutputAssetItems
  }
})

type Deferred<T> = {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason?: unknown) => void
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolveFn, rejectFn) => {
    resolve = resolveFn
    reject = rejectFn
  })
  return { promise, resolve, reject }
}

function createAsset(overrides: Partial<AssetItem> = {}): AssetItem {
  return {
    id: 'asset-1',
    name: 'parent.png',
    tags: [],
    created_at: '2025-01-01T00:00:00.000Z',
    user_metadata: {
      jobId: 'job-1',
      nodeId: 'node-1',
      subfolder: 'outputs'
    },
    ...overrides
  }
}

describe('useOutputStacks', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('expands stacks and exposes children as selectable assets', async () => {
    const parent = createAsset({ id: 'parent', name: 'parent.png' })
    const childA = createAsset({
      id: 'child-a',
      name: 'child-a.png',
      user_metadata: undefined
    })
    const childB = createAsset({
      id: 'child-b',
      name: 'child-b.png',
      user_metadata: undefined
    })

    vi.mocked(mocks.resolveOutputAssetItems).mockResolvedValue([childA, childB])

    const { assetItems, isStackExpanded, selectableAssets, toggleStack } =
      useOutputStacks({ assets: ref([parent]) })

    await toggleStack(parent)

    expect(mocks.resolveOutputAssetItems).toHaveBeenCalledWith(
      expect.objectContaining({ jobId: 'job-1' }),
      {
        createdAt: parent.created_at,
        excludeOutputKey: 'node-1-outputs-parent.png'
      }
    )
    expect(isStackExpanded(parent)).toBe(true)
    expect(assetItems.value.map((item) => item.asset.id)).toEqual([
      parent.id,
      childA.id,
      childB.id
    ])
    expect(assetItems.value[1]).toMatchObject({
      asset: childA,
      isChild: true
    })
    expect(assetItems.value[2]).toMatchObject({
      asset: childB,
      isChild: true
    })
    expect(selectableAssets.value).toEqual([parent, childA, childB])
  })

  it('collapses an expanded stack when toggled again', async () => {
    const parent = createAsset({ id: 'parent', name: 'parent.png' })
    const child = createAsset({
      id: 'child',
      name: 'child.png',
      user_metadata: undefined
    })

    vi.mocked(mocks.resolveOutputAssetItems).mockResolvedValue([child])

    const { assetItems, isStackExpanded, toggleStack } = useOutputStacks({
      assets: ref([parent])
    })

    await toggleStack(parent)
    await toggleStack(parent)

    expect(isStackExpanded(parent)).toBe(false)
    expect(assetItems.value.map((item) => item.asset.id)).toEqual([parent.id])
  })

  it('ignores assets without stack metadata', async () => {
    const asset = createAsset({
      id: 'no-meta',
      name: 'no-meta.png',
      user_metadata: undefined
    })

    const { assetItems, isStackExpanded, toggleStack } = useOutputStacks({
      assets: ref([asset])
    })

    await toggleStack(asset)

    expect(mocks.resolveOutputAssetItems).not.toHaveBeenCalled()
    expect(isStackExpanded(asset)).toBe(false)
    expect(assetItems.value).toHaveLength(1)
    expect(assetItems.value[0].asset).toMatchObject(asset)
  })

  it('does not expand when no children are resolved', async () => {
    const parent = createAsset({ id: 'parent', name: 'parent.png' })

    vi.mocked(mocks.resolveOutputAssetItems).mockResolvedValue([])

    const { assetItems, isStackExpanded, toggleStack } = useOutputStacks({
      assets: ref([parent])
    })

    await toggleStack(parent)

    expect(isStackExpanded(parent)).toBe(false)
    expect(assetItems.value.map((item) => item.asset.id)).toEqual([parent.id])
  })

  it('does not expand when resolving children throws', async () => {
    const parent = createAsset({ id: 'parent', name: 'parent.png' })
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    vi.mocked(mocks.resolveOutputAssetItems).mockRejectedValue(
      new Error('resolve failed')
    )

    const { assetItems, isStackExpanded, toggleStack } = useOutputStacks({
      assets: ref([parent])
    })

    await toggleStack(parent)

    expect(isStackExpanded(parent)).toBe(false)
    expect(assetItems.value.map((item) => item.asset.id)).toEqual([parent.id])

    errorSpy.mockRestore()
  })

  it('guards against duplicate loads while a stack is resolving', async () => {
    const parent = createAsset({ id: 'parent', name: 'parent.png' })
    const child = createAsset({
      id: 'child',
      name: 'child.png',
      user_metadata: undefined
    })
    const deferred = createDeferred<AssetItem[]>()

    vi.mocked(mocks.resolveOutputAssetItems).mockReturnValue(deferred.promise)

    const { assetItems, toggleStack } = useOutputStacks({
      assets: ref([parent])
    })

    const firstToggle = toggleStack(parent)
    const secondToggle = toggleStack(parent)

    expect(mocks.resolveOutputAssetItems).toHaveBeenCalledTimes(1)

    deferred.resolve([child])

    await firstToggle
    await secondToggle

    expect(assetItems.value.map((item) => item.asset.id)).toEqual([
      parent.id,
      child.id
    ])
  })
})
