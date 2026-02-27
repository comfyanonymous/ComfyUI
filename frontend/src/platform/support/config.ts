import { isCloud, isNightly } from '@/platform/distribution/types'

/**
 * Zendesk ticket form field IDs.
 */
export const ZENDESK_FIELDS = {
  /** Distribution tag (cloud vs OSS) */
  DISTRIBUTION: 'tf_42243568391700',
  /** User email (anonymous requester) */
  ANONYMOUS_EMAIL: 'tf_anonymous_requester_email',
  /** User email (authenticated) */
  EMAIL: 'tf_40029135130388',
  /** User ID */
  USER_ID: 'tf_42515251051412'
} as const

/**
 * Gets the distribution identifier for Zendesk tracking.
 * Helps distinguish feedback from different build types.
 */
export function getDistribution(): 'ccloud' | 'oss-nightly' | 'oss' {
  if (isCloud) return 'ccloud'
  if (isNightly) return 'oss-nightly'
  return 'oss'
}

const SUPPORT_BASE_URL = 'https://support.comfy.org/hc/en-us/requests/new'

/**
 * Builds the support URL with optional user information for pre-filling.
 * Users without login information will still get a valid support URL without pre-fill.
 *
 * @param params - User information to pre-fill in the support form
 * @returns Complete Zendesk support URL with query parameters
 */
export function buildSupportUrl(params?: {
  userEmail?: string | null
  userId?: string | null
}): string {
  const searchParams = new URLSearchParams({
    [ZENDESK_FIELDS.DISTRIBUTION]: getDistribution()
  })

  if (params?.userEmail) {
    searchParams.append(ZENDESK_FIELDS.ANONYMOUS_EMAIL, params.userEmail)
    searchParams.append(ZENDESK_FIELDS.EMAIL, params.userEmail)
  }
  if (params?.userId) {
    searchParams.append(ZENDESK_FIELDS.USER_ID, params.userId)
  }

  return `${SUPPORT_BASE_URL}?${searchParams.toString()}`
}
