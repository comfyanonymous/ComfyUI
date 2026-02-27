import type { NodeReplacementResponse } from './types'

import { api } from '@/scripts/api'

export async function fetchNodeReplacements(): Promise<NodeReplacementResponse> {
  const response = await api.fetchApi('/node_replacements')
  if (!response.ok) {
    throw new Error(
      `Failed to fetch node replacements: ${response.status} ${response.statusText}`
    )
  }
  return response.json()
}
