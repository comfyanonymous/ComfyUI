import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '@/scripts/api'

vi.mock('axios')

describe('getFolderPaths', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('returns legacy API response when available', async () => {
    const mockResponse = { checkpoints: ['/test/checkpoints'] }
    vi.mocked(axios.get).mockResolvedValueOnce({ data: mockResponse })

    const result = await api.getFolderPaths()

    expect(result).toEqual(mockResponse)
  })

  it('returns empty object when legacy API unavailable (dynamic discovery)', async () => {
    vi.mocked(axios.get).mockRejectedValueOnce(new Error())

    const result = await api.getFolderPaths()

    // With dynamic discovery, we don't pre-generate directories when API is unavailable
    expect(result).toEqual({})
  })
})
