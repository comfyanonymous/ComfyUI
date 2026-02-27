import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTelemetry } from '@/platform/telemetry'

vi.mock('@/platform/distribution/types', () => ({
  isCloud: false
}))

describe('useTelemetry', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should return null when not in cloud distribution', () => {
    const provider = useTelemetry()

    // Should return null for OSS builds
    expect(provider).toBeNull()
  })

  it('should return null consistently for OSS builds', () => {
    const provider1 = useTelemetry()
    const provider2 = useTelemetry()

    // Both should be null for OSS builds
    expect(provider1).toBeNull()
    expect(provider2).toBeNull()
  })
})
