import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { OutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import type { ResultItemImpl } from '@/stores/queueStore'

import { resolveOutputAssetItems } from './outputAssetUtil'

const mocks = vi.hoisted(() => ({
  getJobDetail: vi.fn(),
  getPreviewableOutputsFromJobDetail: vi.fn()
}))

vi.mock('@/services/jobOutputCache', () => ({
  getJobDetail: mocks.getJobDetail,
  getPreviewableOutputsFromJobDetail: mocks.getPreviewableOutputsFromJobDetail
}))

type OutputOverrides = Partial<{
  filename: string
  subfolder: string
  nodeId: string
  url: string
}>

function createOutput(overrides: OutputOverrides = {}): ResultItemImpl {
  return {
    filename: 'file.png',
    subfolder: 'sub',
    nodeId: '1',
    url: 'https://example.com/file.png',
    ...overrides
  } as ResultItemImpl
}

describe('resolveOutputAssetItems', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('maps outputs and excludes a composite output key', async () => {
    const outputA = createOutput({
      filename: 'a.png',
      nodeId: '1',
      url: 'https://example.com/a.png'
    })
    const outputB = createOutput({
      filename: 'b.png',
      nodeId: '2',
      url: 'https://example.com/b.png'
    })
    const metadata: OutputAssetMetadata = {
      jobId: 'job-1',
      nodeId: '1',
      subfolder: 'sub',
      executionTimeInSeconds: 12.5,
      outputCount: 2,
      allOutputs: [outputA, outputB]
    }

    const results = await resolveOutputAssetItems(metadata, {
      createdAt: '2025-01-01T00:00:00.000Z',
      excludeOutputKey: '2-sub-b.png'
    })

    expect(mocks.getJobDetail).not.toHaveBeenCalled()
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(
      expect.objectContaining({
        id: 'job-1-1-sub-a.png',
        name: 'a.png',
        created_at: '2025-01-01T00:00:00.000Z',
        tags: ['output'],
        preview_url: 'https://example.com/a.png'
      })
    )
    expect(results[0].user_metadata).toEqual(
      expect.objectContaining({
        jobId: 'job-1',
        nodeId: '1',
        subfolder: 'sub',
        executionTimeInSeconds: 12.5
      })
    )
  })

  it('loads full outputs when metadata indicates more outputs', async () => {
    const previewOutput = createOutput({
      filename: 'preview.png',
      nodeId: '1',
      url: 'https://example.com/preview.png'
    })
    const fullOutput = createOutput({
      filename: 'full.png',
      nodeId: '2',
      url: 'https://example.com/full.png'
    })
    const metadata: OutputAssetMetadata = {
      jobId: 'job-2',
      nodeId: '1',
      subfolder: 'sub',
      outputCount: 3,
      allOutputs: [previewOutput]
    }
    const jobDetail = { id: 'job-1' }

    mocks.getJobDetail.mockResolvedValue(jobDetail)
    mocks.getPreviewableOutputsFromJobDetail.mockReturnValue([
      fullOutput,
      previewOutput
    ])

    const results = await resolveOutputAssetItems(metadata)

    expect(mocks.getJobDetail).toHaveBeenCalledWith('job-2')
    expect(mocks.getPreviewableOutputsFromJobDetail).toHaveBeenCalledWith(
      jobDetail
    )
    expect(results.map((asset) => asset.name)).toEqual([
      'full.png',
      'preview.png'
    ])
  })

  it('keeps root outputs with empty subfolders', async () => {
    const output = createOutput({
      filename: 'root.png',
      nodeId: '1',
      subfolder: '',
      url: 'https://example.com/root.png'
    })
    const metadata: OutputAssetMetadata = {
      jobId: 'job-root',
      nodeId: '1',
      subfolder: '',
      outputCount: 1,
      allOutputs: [output]
    }

    const results = await resolveOutputAssetItems(metadata)

    expect(mocks.getJobDetail).not.toHaveBeenCalled()
    expect(results).toHaveLength(1)
    const [asset] = results
    if (!asset) {
      throw new Error('Expected a root output asset')
    }
    expect(asset.id).toBe('job-root-1--root.png')
    if (!asset.user_metadata) {
      throw new Error('Expected output metadata')
    }
    expect(asset.user_metadata.subfolder).toBe('')
  })
})
