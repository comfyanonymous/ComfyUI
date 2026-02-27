import { useTelemetry } from '@/platform/telemetry'
import type { AuditLog } from '@/services/customerEventsService'

const STORAGE_KEY = 'pending_topup_timestamp'
const MAX_AGE_MS = 24 * 60 * 60 * 1000 // 24 hours

/**
 * Start tracking a credit top-up purchase.
 * Call this before opening the Stripe checkout window.
 */
export function startTopupTracking(): void {
  localStorage.setItem(STORAGE_KEY, Date.now().toString())
}

/**
 * Check if a pending top-up has completed by looking for a credit_added event
 * that occurred after the tracking started.
 *
 * @param events - Array of audit log events to check
 * @returns true if a completed top-up was detected and telemetry was sent
 */
export function checkForCompletedTopup(
  events: AuditLog[] | undefined | null
): boolean {
  const timestampStr = localStorage.getItem(STORAGE_KEY)
  if (!timestampStr) return false

  const timestamp = parseInt(timestampStr, 10)

  // Auto-cleanup if expired (older than 24 hours)
  if (Date.now() - timestamp > MAX_AGE_MS) {
    localStorage.removeItem(STORAGE_KEY)
    return false
  }

  if (!events || events.length === 0) return false

  // Find credit_added event that occurred after our timestamp
  const completedTopup = events.find(
    (e) =>
      e.event_type === 'credit_added' &&
      e.createdAt &&
      new Date(e.createdAt).getTime() > timestamp
  )

  if (completedTopup) {
    useTelemetry()?.trackApiCreditTopupSucceeded()
    localStorage.removeItem(STORAGE_KEY)
    return true
  }

  return false
}

/**
 * Clear any pending top-up tracking.
 * Useful for testing or manual cleanup.
 */
export function clearTopupTracking(): void {
  localStorage.removeItem(STORAGE_KEY)
}
