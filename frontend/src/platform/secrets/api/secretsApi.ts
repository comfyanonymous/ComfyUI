import { api } from '@/scripts/api'

import type {
  SecretCreateRequest,
  SecretErrorCode,
  SecretMetadata,
  SecretUpdateRequest
} from '../types'
import { SECRET_ERROR_CODES } from '../types'

interface ListSecretsResponse {
  data: SecretMetadata[]
}

interface ErrorResponse {
  message?: string
  code?: string
}

export class SecretsApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: SecretErrorCode
  ) {
    super(message)
    this.name = 'SecretsApiError'
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData: ErrorResponse = {}
    try {
      errorData = await response.json()
    } catch {
      // Response body is not JSON
    }
    const code = SECRET_ERROR_CODES.includes(
      errorData.code as (typeof SECRET_ERROR_CODES)[number]
    )
      ? (errorData.code as SecretErrorCode)
      : undefined
    throw new SecretsApiError(
      errorData.message ?? response.statusText,
      response.status,
      code
    )
  }
  return response.json()
}

export async function listSecrets(): Promise<SecretMetadata[]> {
  const response = await api.fetchApi('/secrets')
  const data = await handleResponse<ListSecretsResponse>(response)
  return data.data
}

export async function createSecret(
  payload: SecretCreateRequest
): Promise<SecretMetadata> {
  const response = await api.fetchApi('/secrets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  return handleResponse<SecretMetadata>(response)
}

export async function updateSecret(
  id: string,
  payload: SecretUpdateRequest
): Promise<SecretMetadata> {
  const response = await api.fetchApi(`/secrets/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  return handleResponse<SecretMetadata>(response)
}

export async function deleteSecret(id: string): Promise<void> {
  const response = await api.fetchApi(`/secrets/${id}`, {
    method: 'DELETE'
  })
  if (!response.ok) {
    await handleResponse<void>(response)
  }
}
