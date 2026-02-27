import { isPlainObject } from 'es-toolkit'
import { withTimeout } from 'es-toolkit/promise'

import type { CheckoutAttributionMetadata } from '../types'

type GaIdentity = {
  client_id?: string
  session_id?: string
  session_number?: string
}

const GA_IDENTITY_FIELDS = [
  'client_id',
  'session_id',
  'session_number'
] as const satisfies ReadonlyArray<GtagGetFieldName>
type GaIdentityField = GtagGetFieldName

const ATTRIBUTION_QUERY_KEYS = [
  'im_ref',
  'utm_source',
  'utm_medium',
  'utm_campaign',
  'utm_term',
  'utm_content',
  'gclid',
  'gbraid',
  'wbraid'
] as const
type AttributionQueryKey = (typeof ATTRIBUTION_QUERY_KEYS)[number]
const ATTRIBUTION_STORAGE_KEY = 'comfy_checkout_attribution'
const GENERATE_CLICK_ID_TIMEOUT_MS = 300
const GET_GA_IDENTITY_TIMEOUT_MS = 300

function readStoredAttribution(): Partial<Record<AttributionQueryKey, string>> {
  if (typeof window === 'undefined') return {}

  try {
    const stored = localStorage.getItem(ATTRIBUTION_STORAGE_KEY)
    if (!stored) return {}

    const parsed: unknown = JSON.parse(stored)
    if (!isPlainObject(parsed)) return {}

    const result: Partial<Record<AttributionQueryKey, string>> = {}

    for (const key of ATTRIBUTION_QUERY_KEYS) {
      const value = asNonEmptyString(parsed[key])
      if (value) {
        result[key] = value
      }
    }

    return result
  } catch {
    return {}
  }
}

function persistAttribution(
  payload: Partial<Record<AttributionQueryKey, string>>
): void {
  if (typeof window === 'undefined') return

  try {
    localStorage.setItem(ATTRIBUTION_STORAGE_KEY, JSON.stringify(payload))
  } catch {
    return
  }
}

function readAttributionFromUrl(
  search: string
): Partial<Record<AttributionQueryKey, string>> {
  const params = new URLSearchParams(search)

  const result: Partial<Record<AttributionQueryKey, string>> = {}

  for (const key of ATTRIBUTION_QUERY_KEYS) {
    const value = params.get(key)
    if (value) {
      result[key] = value
    }
  }

  return result
}

function hasAttributionChanges(
  existing: Partial<Record<AttributionQueryKey, string>>,
  incoming: Partial<Record<AttributionQueryKey, string>>
): boolean {
  for (const key of ATTRIBUTION_QUERY_KEYS) {
    const value = incoming[key]
    if (value !== undefined && existing[key] !== value) {
      return true
    }
  }

  return false
}

function asNonEmptyString(value: unknown): string | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value)
  }

  return typeof value === 'string' && value.length > 0 ? value : undefined
}

async function getGaIdentityField(
  measurementId: string,
  fieldName: GaIdentityField
): Promise<string | undefined> {
  if (typeof window === 'undefined' || typeof window.gtag !== 'function') {
    return undefined
  }
  const gtag = window.gtag

  return withTimeout(
    () =>
      new Promise<string | undefined>((resolve) => {
        gtag('get', measurementId, fieldName, (value) => {
          resolve(asNonEmptyString(value))
        })
      }),
    GET_GA_IDENTITY_TIMEOUT_MS
  ).catch(() => undefined)
}

async function getGaIdentity(): Promise<GaIdentity | undefined> {
  const measurementId = asNonEmptyString(window.__CONFIG__?.ga_measurement_id)
  if (!measurementId) {
    return undefined
  }

  const [clientId, sessionId, sessionNumber] = await Promise.all(
    GA_IDENTITY_FIELDS.map((fieldName) =>
      getGaIdentityField(measurementId, fieldName)
    )
  )

  if (!clientId && !sessionId && !sessionNumber) {
    return undefined
  }

  return {
    client_id: clientId,
    session_id: sessionId,
    session_number: sessionNumber
  }
}

async function getGeneratedClickId(): Promise<string | undefined> {
  if (typeof window === 'undefined') {
    return undefined
  }

  const impactQueue = window.ire
  if (typeof impactQueue !== 'function') {
    return undefined
  }

  try {
    return await withTimeout(
      () =>
        new Promise<string | undefined>((resolve, reject) => {
          try {
            impactQueue('generateClickId', (clickId: unknown) => {
              resolve(asNonEmptyString(clickId))
            })
          } catch (error) {
            reject(error)
          }
        }),
      GENERATE_CLICK_ID_TIMEOUT_MS
    )
  } catch {
    return undefined
  }
}

export function captureCheckoutAttributionFromSearch(search: string): void {
  const fromUrl = readAttributionFromUrl(search)
  const storedAttribution = readStoredAttribution()
  if (Object.keys(fromUrl).length === 0) return

  if (!hasAttributionChanges(storedAttribution, fromUrl)) return

  persistAttribution({
    ...storedAttribution,
    ...fromUrl
  })
}

export async function getCheckoutAttribution(): Promise<CheckoutAttributionMetadata> {
  if (typeof window === 'undefined') return {}

  const storedAttribution = readStoredAttribution()
  const fromUrl = readAttributionFromUrl(window.location.search)
  const generatedClickId = await getGeneratedClickId()
  const attribution: Partial<Record<AttributionQueryKey, string>> = {
    ...storedAttribution,
    ...fromUrl
  }

  if (generatedClickId) {
    attribution.im_ref = generatedClickId
  }

  if (hasAttributionChanges(storedAttribution, attribution)) {
    persistAttribution(attribution)
  }

  const gaIdentity = await getGaIdentity()

  return {
    ...attribution,
    ga_client_id: gaIdentity?.client_id,
    ga_session_id: gaIdentity?.session_id,
    ga_session_number: gaIdentity?.session_number
  }
}
