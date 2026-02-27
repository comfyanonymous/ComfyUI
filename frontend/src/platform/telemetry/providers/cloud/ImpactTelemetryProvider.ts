import { captureCheckoutAttributionFromSearch } from '@/platform/telemetry/utils/checkoutAttribution'
import { useApiKeyAuthStore } from '@/stores/apiKeyAuthStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

import type { PageViewMetadata, TelemetryProvider } from '../../types'

const IMPACT_SCRIPT_URL =
  'https://utt.impactcdn.com/A6951770-3747-434a-9ac7-4e582e67d91f1.js'
const IMPACT_QUEUE_NAME = 'ire'
const EMPTY_CUSTOMER_VALUE = ''

/**
 * Impact telemetry provider.
 * Initializes the Impact queue globals and loads the runtime script.
 */
export class ImpactTelemetryProvider implements TelemetryProvider {
  private initialized = false
  private stores: {
    apiKeyAuthStore: ReturnType<typeof useApiKeyAuthStore>
    firebaseAuthStore: ReturnType<typeof useFirebaseAuthStore>
  } | null = null

  constructor() {
    this.initialize()
  }

  trackPageView(_pageName: string, properties?: PageViewMetadata): void {
    const search = this.extractSearchFromPath(properties?.path)

    if (search) {
      captureCheckoutAttributionFromSearch(search)
    } else if (typeof window !== 'undefined') {
      captureCheckoutAttributionFromSearch(window.location.search)
    }

    void this.identifyCurrentUser()
  }

  private initialize(): void {
    if (typeof window === 'undefined' || this.initialized) return

    window.ire_o = IMPACT_QUEUE_NAME

    if (!window.ire) {
      const queueFn: NonNullable<Window['ire']> = (...args: unknown[]) => {
        ;(queueFn.a ??= []).push(args)
      }
      window.ire = queueFn
    }

    const existingScript = document.querySelector(
      `script[src="${IMPACT_SCRIPT_URL}"]`
    )
    if (existingScript) {
      this.initialized = true
      return
    }

    const script = document.createElement('script')
    script.async = true
    script.src = IMPACT_SCRIPT_URL

    document.head.insertBefore(script, document.head.firstChild)

    this.initialized = true
  }

  private extractSearchFromPath(path?: string): string {
    if (!path) return ''

    if (typeof window !== 'undefined') {
      try {
        const url = new URL(path, window.location.origin)
        return url.search
      } catch {
        // Fall through to manual parsing.
      }
    }

    const queryIndex = path.indexOf('?')
    return queryIndex >= 0 ? path.slice(queryIndex) : ''
  }

  private async identifyCurrentUser(): Promise<void> {
    if (typeof window === 'undefined') return

    const { customerId, customerEmail } = this.resolveCustomerIdentity()
    const normalizedEmail = customerEmail.trim().toLowerCase()
    // Impact's Identify spec requires customerEmail to be sent as a SHA1 hash.
    const hashedEmail = normalizedEmail
      ? await this.hashSha1(normalizedEmail)
      : EMPTY_CUSTOMER_VALUE

    window.ire?.('identify', {
      customerId,
      customerEmail: hashedEmail
    })
  }

  private resolveCustomerIdentity(): {
    customerId: string
    customerEmail: string
  } {
    const stores = this.resolveAuthStores()
    if (!stores) {
      return {
        customerId: EMPTY_CUSTOMER_VALUE,
        customerEmail: EMPTY_CUSTOMER_VALUE
      }
    }

    if (stores.firebaseAuthStore.currentUser) {
      return {
        customerId:
          stores.firebaseAuthStore.currentUser.uid ?? EMPTY_CUSTOMER_VALUE,
        customerEmail:
          stores.firebaseAuthStore.currentUser.email ?? EMPTY_CUSTOMER_VALUE
      }
    }

    if (stores.apiKeyAuthStore.isAuthenticated) {
      return {
        customerId:
          stores.apiKeyAuthStore.currentUser?.id ?? EMPTY_CUSTOMER_VALUE,
        customerEmail:
          stores.apiKeyAuthStore.currentUser?.email ?? EMPTY_CUSTOMER_VALUE
      }
    }

    return {
      customerId: EMPTY_CUSTOMER_VALUE,
      customerEmail: EMPTY_CUSTOMER_VALUE
    }
  }

  private resolveAuthStores(): {
    apiKeyAuthStore: ReturnType<typeof useApiKeyAuthStore>
    firebaseAuthStore: ReturnType<typeof useFirebaseAuthStore>
  } | null {
    if (this.stores) {
      return this.stores
    }

    try {
      const stores = {
        apiKeyAuthStore: useApiKeyAuthStore(),
        firebaseAuthStore: useFirebaseAuthStore()
      }
      this.stores = stores
      return stores
    } catch {
      return null
    }
  }

  private async hashSha1(value: string): Promise<string> {
    try {
      if (!globalThis.crypto?.subtle || typeof TextEncoder === 'undefined') {
        return EMPTY_CUSTOMER_VALUE
      }

      const digestBuffer = await crypto.subtle.digest(
        'SHA-1',
        new TextEncoder().encode(value)
      )

      return Array.from(new Uint8Array(digestBuffer))
        .map((byte) => byte.toString(16).padStart(2, '0'))
        .join('')
    } catch {
      return EMPTY_CUSTOMER_VALUE
    }
  }
}
