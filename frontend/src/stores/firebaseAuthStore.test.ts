import { FirebaseError } from 'firebase/app'
import type { User, UserCredential } from 'firebase/auth'
import * as firebaseAuth from 'firebase/auth'
import { setActivePinia } from 'pinia'
import type { Mock } from 'vitest'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as vuefire from 'vuefire'

import { useDialogService } from '@/services/dialogService'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import { createTestingPinia } from '@pinia/testing'

// Hoisted mocks for dynamic imports
const { mockDistributionTypes } = vi.hoisted(() => ({
  mockDistributionTypes: {
    isCloud: true,
    isDesktop: true
  }
}))

type MockUser = Omit<User, 'getIdToken'> & {
  getIdToken: Mock
}

type MockAuth = Record<string, unknown>

// Mock fetch
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

// Mock successful API responses
const mockCreateCustomerResponse = {
  ok: true,
  statusText: 'OK',
  json: () => Promise.resolve({ id: 'test-customer-id' })
}

const mockFetchBalanceResponse = {
  ok: true,
  json: () => Promise.resolve({ balance: 0 })
}

const mockAddCreditsResponse = {
  ok: true,
  statusText: 'OK'
}

const mockAccessBillingPortalResponse = {
  ok: true,
  statusText: 'OK',
  json: () =>
    Promise.resolve({ billing_portal_url: 'https://billing.stripe.com/test' })
}

vi.mock('vuefire', () => ({
  useFirebaseAuth: vi.fn()
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  }),
  createI18n: () => ({
    global: {
      t: (key: string) => key
    }
  })
}))

vi.mock('firebase/auth', async (importOriginal) => {
  const actual = await importOriginal<typeof firebaseAuth>()
  return {
    ...actual,
    signInWithEmailAndPassword: vi.fn(),
    createUserWithEmailAndPassword: vi.fn(),
    signOut: vi.fn(),
    onAuthStateChanged: vi.fn(),
    onIdTokenChanged: vi.fn(),
    signInWithPopup: vi.fn(),
    GoogleAuthProvider: class {
      addScope = vi.fn()
      setCustomParameters = vi.fn()
    },
    GithubAuthProvider: class {
      addScope = vi.fn()
      setCustomParameters = vi.fn()
    },
    setPersistence: vi.fn().mockResolvedValue(undefined)
  }
})

// Mock useToastStore
vi.mock('@/stores/toastStore', () => ({
  useToastStore: () => ({
    add: vi.fn()
  })
}))

// Mock useDialogService
vi.mock('@/services/dialogService')

// Mock apiKeyAuthStore
const mockApiKeyGetAuthHeader = vi.fn().mockReturnValue(null)
vi.mock('@/stores/apiKeyAuthStore', () => ({
  useApiKeyAuthStore: () => ({
    getAuthHeader: mockApiKeyGetAuthHeader,
    getApiKey: vi.fn(),
    currentUser: null,
    isAuthenticated: false,
    storeApiKey: vi.fn(),
    clearStoredApiKey: vi.fn()
  })
}))

describe('useFirebaseAuthStore', () => {
  let store: ReturnType<typeof useFirebaseAuthStore>
  let authStateCallback: (user: User | null) => void
  let idTokenCallback: (user: User | null) => void

  const mockAuth: MockAuth = {
    /* mock Auth object */
  }

  const mockUser: MockUser = {
    uid: 'test-user-id',
    email: 'test@example.com',
    getIdToken: vi.fn().mockResolvedValue('mock-id-token')
  } as Partial<User> as MockUser

  beforeEach(() => {
    vi.resetAllMocks()

    // Setup dialog service mock
    vi.mocked(useDialogService, { partial: true }).mockReturnValue({
      showErrorDialog: vi.fn()
    })

    // Mock useFirebaseAuth to return our mock auth object
    vi.mocked(vuefire.useFirebaseAuth).mockReturnValue(
      mockAuth as Partial<
        ReturnType<typeof vuefire.useFirebaseAuth>
      > as ReturnType<typeof vuefire.useFirebaseAuth>
    )

    // Mock onAuthStateChanged to capture the callback and simulate initial auth state
    vi.mocked(firebaseAuth.onAuthStateChanged).mockImplementation(
      (_, callback) => {
        authStateCallback = callback as (user: User | null) => void
        // Call the callback with our mock user
        ;(callback as (user: User | null) => void)(mockUser)
        // Return an unsubscribe function
        return vi.fn()
      }
    )

    // Mock fetch responses
    mockFetch.mockImplementation((url: string) => {
      if (url.endsWith('/customers')) {
        return Promise.resolve(mockCreateCustomerResponse)
      }
      if (url.endsWith('/customers/balance')) {
        return Promise.resolve(mockFetchBalanceResponse)
      }
      if (url.endsWith('/customers/credit')) {
        return Promise.resolve(mockAddCreditsResponse)
      }
      if (url.endsWith('/customers/billing')) {
        return Promise.resolve(mockAccessBillingPortalResponse)
      }
      return Promise.reject(new Error('Unexpected API call'))
    })

    // Initialize Pinia
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useFirebaseAuthStore()

    // Reset and set up getIdToken mock
    mockUser.getIdToken.mockReset()
    mockUser.getIdToken.mockResolvedValue('mock-id-token')

    // Default: no API key auth
    mockApiKeyGetAuthHeader.mockReturnValue(null)
  })

  describe('token refresh events', () => {
    beforeEach(async () => {
      vi.resetModules()
      vi.mock('@/platform/distribution/types', () => mockDistributionTypes)

      vi.mocked(firebaseAuth.onIdTokenChanged).mockImplementation(
        (_auth, callback) => {
          idTokenCallback = callback as (user: User | null) => void
          return vi.fn()
        }
      )

      vi.mocked(vuefire.useFirebaseAuth).mockReturnValue(
        mockAuth as Partial<
          ReturnType<typeof vuefire.useFirebaseAuth>
        > as ReturnType<typeof vuefire.useFirebaseAuth>
      )

      setActivePinia(createTestingPinia({ stubActions: false }))
      const storeModule = await import('@/stores/firebaseAuthStore')
      store = storeModule.useFirebaseAuthStore()
    })

    it("should not increment tokenRefreshTrigger on the user's first ID token event", () => {
      idTokenCallback?.(mockUser)
      expect(store.tokenRefreshTrigger).toBe(0)
    })

    it('should increment tokenRefreshTrigger on subsequent ID token events for the same user', () => {
      idTokenCallback?.(mockUser)
      idTokenCallback?.(mockUser)
      expect(store.tokenRefreshTrigger).toBe(1)
    })

    it('should not increment when ID token event is for a different user UID', () => {
      const otherUser = { uid: 'other-user-id' } as Partial<User> as User
      idTokenCallback?.(mockUser)
      idTokenCallback?.(otherUser)
      expect(store.tokenRefreshTrigger).toBe(0)
    })

    it('should increment after switching to a new UID and receiving a second event for that UID', () => {
      const otherUser = { uid: 'other-user-id' } as Partial<User> as User
      idTokenCallback?.(mockUser)
      idTokenCallback?.(otherUser)
      idTokenCallback?.(otherUser)
      expect(store.tokenRefreshTrigger).toBe(1)
    })
  })

  it('should initialize with the current user', () => {
    expect(store.currentUser).toEqual(mockUser)
    expect(store.isAuthenticated).toBe(true)
    expect(store.userEmail).toBe('test@example.com')
    expect(store.userId).toBe('test-user-id')
    expect(store.loading).toBe(false)
  })

  it('should set persistence to local storage on initialization', () => {
    expect(firebaseAuth.setPersistence).toHaveBeenCalledWith(
      mockAuth,
      firebaseAuth.browserLocalPersistence
    )
  })

  it('should properly clean up error state between operations', async () => {
    // First, cause an error
    const mockError = new Error('Invalid password')
    vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockRejectedValueOnce(
      mockError
    )

    try {
      await store.login('test@example.com', 'wrong-password')
    } catch (e) {
      // Error expected
    }

    // Now, succeed on next attempt
    vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockResolvedValueOnce({
      user: mockUser
    } as Partial<UserCredential> as UserCredential)

    await store.login('test@example.com', 'correct-password')
  })

  describe('login', () => {
    it('should login with valid credentials', async () => {
      const mockUserCredential = { user: mockUser }
      vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockResolvedValue(
        mockUserCredential as Partial<UserCredential> as UserCredential
      )

      const result = await store.login('test@example.com', 'password')

      expect(firebaseAuth.signInWithEmailAndPassword).toHaveBeenCalledWith(
        mockAuth,
        'test@example.com',
        'password'
      )
      expect(result).toEqual(mockUserCredential)
      expect(store.loading).toBe(false)
    })

    it('should handle login errors', async () => {
      const mockError = new Error('Invalid password')
      vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockRejectedValue(
        mockError
      )

      await expect(
        store.login('test@example.com', 'wrong-password')
      ).rejects.toThrow('Invalid password')

      expect(firebaseAuth.signInWithEmailAndPassword).toHaveBeenCalledWith(
        mockAuth,
        'test@example.com',
        'wrong-password'
      )
      expect(store.loading).toBe(false)
    })

    it('should handle concurrent login attempts correctly', async () => {
      // Set up multiple login promises
      const mockUserCredential = { user: mockUser }
      vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockResolvedValue(
        mockUserCredential as Partial<UserCredential> as UserCredential
      )

      const loginPromise1 = store.login('user1@example.com', 'password1')
      const loginPromise2 = store.login('user2@example.com', 'password2')

      // Resolve both promises
      await Promise.all([loginPromise1, loginPromise2])

      // Verify the loading state is reset
      expect(store.loading).toBe(false)
    })
  })

  describe('register', () => {
    it('should register a new user', async () => {
      const mockUserCredential = { user: mockUser }
      vi.mocked(firebaseAuth.createUserWithEmailAndPassword).mockResolvedValue(
        mockUserCredential as Partial<UserCredential> as UserCredential
      )

      const result = await store.register('new@example.com', 'password')

      expect(firebaseAuth.createUserWithEmailAndPassword).toHaveBeenCalledWith(
        mockAuth,
        'new@example.com',
        'password'
      )
      expect(result).toEqual(mockUserCredential)
      expect(store.loading).toBe(false)
    })

    it('should handle registration errors', async () => {
      const mockError = new Error('Email already in use')
      vi.mocked(firebaseAuth.createUserWithEmailAndPassword).mockRejectedValue(
        mockError
      )

      await expect(
        store.register('existing@example.com', 'password')
      ).rejects.toThrow('Email already in use')

      expect(firebaseAuth.createUserWithEmailAndPassword).toHaveBeenCalledWith(
        mockAuth,
        'existing@example.com',
        'password'
      )
      expect(store.loading).toBe(false)
    })
  })

  describe('logout', () => {
    it('should sign out the user', async () => {
      vi.mocked(firebaseAuth.signOut).mockResolvedValue(undefined)

      await store.logout()

      expect(firebaseAuth.signOut).toHaveBeenCalledWith(mockAuth)
    })

    it('should handle logout errors', async () => {
      const mockError = new Error('Network error')
      vi.mocked(firebaseAuth.signOut).mockRejectedValue(mockError)

      await expect(store.logout()).rejects.toThrow('Network error')

      expect(firebaseAuth.signOut).toHaveBeenCalledWith(mockAuth)
    })
  })

  describe('getIdToken', () => {
    it('should return the user ID token', async () => {
      // FIX 2: Reset the mock and set a specific return value
      mockUser.getIdToken.mockReset()
      mockUser.getIdToken.mockResolvedValue('mock-id-token')

      const token = await store.getIdToken()

      expect(mockUser.getIdToken).toHaveBeenCalled()
      expect(token).toBe('mock-id-token')
    })

    it('should return null when no user is logged in', async () => {
      // Simulate logged out state
      authStateCallback(null)

      const token = await store.getIdToken()

      expect(token).toBeUndefined()
    })

    it('should return null for token after login and logout sequence', async () => {
      // Setup mock for login
      const mockUserCredential = { user: mockUser }
      vi.mocked(firebaseAuth.signInWithEmailAndPassword).mockResolvedValue(
        mockUserCredential as Partial<UserCredential> as UserCredential
      )

      // Login
      await store.login('test@example.com', 'password')

      // Simulate successful auth state update after login
      authStateCallback(mockUser)

      // Verify we're logged in and can get a token
      mockUser.getIdToken.mockReset()
      mockUser.getIdToken.mockResolvedValue('mock-id-token')
      expect(await store.getIdToken()).toBe('mock-id-token')

      // Setup mock for logout
      vi.mocked(firebaseAuth.signOut).mockResolvedValue(undefined)

      // Logout
      await store.logout()

      // Simulate successful auth state update after logout
      authStateCallback(null)

      // Verify token is null after logout
      const tokenAfterLogout = await store.getIdToken()
      expect(tokenAfterLogout).toBeUndefined()
    })

    it('should handle network errors gracefully when offline (reproduces issue #4468)', async () => {
      // This test reproduces the issue where Firebase Auth makes network requests when offline
      // and fails without graceful error handling, causing toast error messages

      // Simulate a user with an expired token that requires network refresh
      mockUser.getIdToken.mockReset()

      // Mock network failure (auth/network-request-failed error from Firebase)
      const networkError = new FirebaseError(
        firebaseAuth.AuthErrorCodes.NETWORK_REQUEST_FAILED,
        'mock error'
      )

      mockUser.getIdToken.mockRejectedValue(networkError)

      const token = await store.getIdToken()
      expect(token).toBeUndefined() // Should return undefined instead of throwing
    })

    it('should show error dialog when getIdToken fails with non-network error', async () => {
      // This test verifies that non-network errors trigger the error dialog
      mockUser.getIdToken.mockReset()

      // Mock a non-network error using actual Firebase Auth error code
      const authError = new FirebaseError(
        firebaseAuth.AuthErrorCodes.USER_DISABLED,
        'User account is disabled.'
      )

      mockUser.getIdToken.mockRejectedValue(authError)

      // Should call the error dialog instead of throwing
      const token = await store.getIdToken()
      const dialogService = useDialogService()

      expect(dialogService.showErrorDialog).toHaveBeenCalledWith(authError, {
        title: 'errorDialog.defaultTitle',
        reportType: 'authenticationError'
      })
      expect(token).toBeUndefined()
    })
  })

  describe('getAuthHeader', () => {
    it('should handle network errors gracefully when getting Firebase token (reproduces issue #4468)', async () => {
      // This test reproduces the issue where getAuthHeader fails due to network errors
      // when Firebase Auth tries to refresh tokens offline

      // Setup user with network error on token refresh
      mockUser.getIdToken.mockReset()
      const networkError = new FirebaseError(
        firebaseAuth.AuthErrorCodes.NETWORK_REQUEST_FAILED,
        'mock error'
      )
      mockUser.getIdToken.mockRejectedValue(networkError)

      const authHeader = await store.getAuthHeader()
      expect(authHeader).toBeNull() // Should fallback gracefully
    })
  })

  describe('social authentication', () => {
    describe('loginWithGoogle', () => {
      it('should sign in with Google', async () => {
        const mockUserCredential = { user: mockUser }
        vi.mocked(firebaseAuth.signInWithPopup).mockResolvedValue(
          mockUserCredential as Partial<UserCredential> as UserCredential
        )

        const result = await store.loginWithGoogle()

        expect(firebaseAuth.signInWithPopup).toHaveBeenCalledWith(
          mockAuth,
          expect.any(firebaseAuth.GoogleAuthProvider)
        )
        expect(result).toEqual(mockUserCredential)
        expect(store.loading).toBe(false)
      })

      it('should handle Google sign in errors', async () => {
        const mockError = new Error('Google authentication failed')
        vi.mocked(firebaseAuth.signInWithPopup).mockRejectedValue(mockError)

        await expect(store.loginWithGoogle()).rejects.toThrow(
          'Google authentication failed'
        )

        expect(firebaseAuth.signInWithPopup).toHaveBeenCalledWith(
          mockAuth,
          expect.any(firebaseAuth.GoogleAuthProvider)
        )
        expect(store.loading).toBe(false)
      })
    })

    describe('loginWithGithub', () => {
      it('should sign in with Github', async () => {
        const mockUserCredential = { user: mockUser }
        vi.mocked(firebaseAuth.signInWithPopup).mockResolvedValue(
          mockUserCredential as Partial<UserCredential> as UserCredential
        )

        const result = await store.loginWithGithub()

        expect(firebaseAuth.signInWithPopup).toHaveBeenCalledWith(
          mockAuth,
          expect.any(firebaseAuth.GithubAuthProvider)
        )
        expect(result).toEqual(mockUserCredential)
        expect(store.loading).toBe(false)
      })

      it('should handle Github sign in errors', async () => {
        const mockError = new Error('Github authentication failed')
        vi.mocked(firebaseAuth.signInWithPopup).mockRejectedValue(mockError)

        await expect(store.loginWithGithub()).rejects.toThrow(
          'Github authentication failed'
        )

        expect(firebaseAuth.signInWithPopup).toHaveBeenCalledWith(
          mockAuth,
          expect.any(firebaseAuth.GithubAuthProvider)
        )
        expect(store.loading).toBe(false)
      })
    })

    it('should handle concurrent social login attempts correctly', async () => {
      const mockUserCredential = { user: mockUser }
      vi.mocked(firebaseAuth.signInWithPopup).mockResolvedValue(
        mockUserCredential as Partial<UserCredential> as UserCredential
      )

      const googleLoginPromise = store.loginWithGoogle()
      const githubLoginPromise = store.loginWithGithub()

      await Promise.all([googleLoginPromise, githubLoginPromise])

      expect(store.loading).toBe(false)
    })
  })

  describe('accessBillingPortal', () => {
    it('should call billing endpoint without body when no targetTier provided', async () => {
      const result = await store.accessBillingPortal()

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/customers/billing'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: 'Bearer mock-id-token',
            'Content-Type': 'application/json'
          })
        })
      )

      const callArgs = mockFetch.mock.calls.find((call) =>
        (call[0] as string).endsWith('/customers/billing')
      )
      expect(callArgs?.[1]).not.toHaveProperty('body')
      expect(result).toEqual({
        billing_portal_url: 'https://billing.stripe.com/test'
      })
    })

    it('should include target_tier in request body when targetTier provided', async () => {
      await store.accessBillingPortal('creator')

      const callArgs = mockFetch.mock.calls.find((call) =>
        (call[0] as string).endsWith('/customers/billing')
      )
      expect(callArgs?.[1]).toHaveProperty('body')
      expect(JSON.parse(callArgs?.[1]?.body as string)).toEqual({
        target_tier: 'creator'
      })
    })

    it('should handle different checkout tier formats', async () => {
      const tiers = [
        'standard',
        'creator',
        'pro',
        'standard-yearly',
        'creator-yearly',
        'pro-yearly'
      ] as const

      for (const tier of tiers) {
        mockFetch.mockClear()
        await store.accessBillingPortal(tier)

        const callArgs = mockFetch.mock.calls.find((call) =>
          (call[0] as string).endsWith('/customers/billing')
        )
        expect(JSON.parse(callArgs?.[1]?.body as string)).toEqual({
          target_tier: tier
        })
      }
    })

    it('should throw error when API returns error response', async () => {
      mockFetch.mockImplementationOnce(() =>
        Promise.resolve({
          ok: false,
          json: () => Promise.resolve({ message: 'Billing portal unavailable' })
        })
      )

      await expect(store.accessBillingPortal()).rejects.toThrow()
    })
  })

  describe('createCustomer', () => {
    it('should succeed with API key auth when no Firebase user is present', async () => {
      authStateCallback(null)
      mockApiKeyGetAuthHeader.mockReturnValue({ 'X-API-KEY': 'test-api-key' })

      const result = await store.createCustomer()

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/customers'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-API-KEY': 'test-api-key'
          })
        })
      )
      expect(result).toEqual({ id: 'test-customer-id' })
    })

    it('should use Firebase token when Firebase user is present', async () => {
      const result = await store.createCustomer()

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/customers'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: 'Bearer mock-id-token'
          })
        })
      )
      expect(result).toEqual({ id: 'test-customer-id' })
    })

    it('should throw when no auth method is available', async () => {
      authStateCallback(null)
      mockApiKeyGetAuthHeader.mockReturnValue(null)

      await expect(store.createCustomer()).rejects.toThrow()
    })
  })
})
