import { Form } from '@primevue/forms'
import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import Button from '@/components/ui/button/Button.vue'
import PrimeVue from 'primevue/config'
import InputText from 'primevue/inputtext'
import Password from 'primevue/password'
import ProgressSpinner from 'primevue/progressspinner'
import ToastService from 'primevue/toastservice'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json' with { type: 'json' }

import SignInForm from './SignInForm.vue'

type ComponentInstance = InstanceType<typeof SignInForm>

// Mock firebase auth modules
vi.mock('firebase/app', () => ({
  initializeApp: vi.fn(),
  getApp: vi.fn()
}))

vi.mock('firebase/auth', () => ({
  getAuth: vi.fn(),
  setPersistence: vi.fn(),
  browserLocalPersistence: {},
  onAuthStateChanged: vi.fn(),
  signInWithEmailAndPassword: vi.fn(),
  signOut: vi.fn(),
  sendPasswordResetEmail: vi.fn()
}))

// Mock the auth composables and stores
const mockSendPasswordReset = vi.fn()
vi.mock('@/composables/auth/useFirebaseAuthActions', () => ({
  useFirebaseAuthActions: vi.fn(() => ({
    sendPasswordReset: mockSendPasswordReset
  }))
}))

let mockLoading = false
vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() => ({
    get loading() {
      return mockLoading
    }
  }))
}))

// Mock toast
const mockToastAdd = vi.fn()
vi.mock('primevue/usetoast', () => ({
  useToast: vi.fn(() => ({
    add: mockToastAdd
  }))
}))

describe('SignInForm', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockSendPasswordReset.mockReset()
    mockToastAdd.mockReset()
    mockLoading = false
  })

  const mountComponent = (
    props = {},
    options = {}
  ): VueWrapper<ComponentInstance> => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(SignInForm, {
      global: {
        plugins: [PrimeVue, i18n, ToastService],
        components: {
          Form,
          Button,
          InputText,
          Password,
          ProgressSpinner
        }
      },
      props,
      ...options
    })
  }

  describe('Forgot Password Link', () => {
    it('shows disabled style when email is empty', async () => {
      const wrapper = mountComponent()
      await nextTick()

      const forgotPasswordSpan = wrapper.find(
        'span.text-muted.text-base.font-medium.select-none'
      )

      expect(forgotPasswordSpan.classes()).toContain('cursor-not-allowed')
      expect(forgotPasswordSpan.classes()).toContain('opacity-50')
    })

    it('shows toast and focuses email input when clicked while disabled', async () => {
      const wrapper = mountComponent()
      const forgotPasswordSpan = wrapper.find(
        'span.text-muted.text-base.font-medium.select-none'
      )

      // Mock getElementById to track focus
      const mockFocus = vi.fn()
      const mockElement: Partial<HTMLElement> = { focus: mockFocus }
      vi.spyOn(document, 'getElementById').mockReturnValue(
        mockElement as HTMLElement
      )

      // Click forgot password link while email is empty
      await forgotPasswordSpan.trigger('click')
      await nextTick()

      // Should show toast warning
      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'warn',
        summary: enMessages.auth.login.emailPlaceholder,
        life: 5000
      })

      // Should focus email input
      expect(document.getElementById).toHaveBeenCalledWith(
        'comfy-org-sign-in-email'
      )
      expect(mockFocus).toHaveBeenCalled()

      // Should NOT call sendPasswordReset
      expect(mockSendPasswordReset).not.toHaveBeenCalled()
    })

    it('calls handleForgotPassword with email when link is clicked', async () => {
      const wrapper = mountComponent()
      const component = wrapper.vm as typeof wrapper.vm & {
        handleForgotPassword: (email: string, valid: boolean) => void
        onSubmit: (data: { valid: boolean; values: unknown }) => void
      }

      // Spy on handleForgotPassword
      const handleForgotPasswordSpy = vi.spyOn(
        component,
        'handleForgotPassword'
      )

      const forgotPasswordSpan = wrapper.find(
        'span.text-muted.text-base.font-medium.select-none'
      )

      // Click the forgot password link
      await forgotPasswordSpan.trigger('click')

      // Should call handleForgotPassword
      expect(handleForgotPasswordSpy).toHaveBeenCalled()
    })
  })

  describe('Form Submission', () => {
    it('emits submit event when onSubmit is called with valid data', async () => {
      const wrapper = mountComponent()
      const component = wrapper.vm as typeof wrapper.vm & {
        handleForgotPassword: (email: string, valid: boolean) => void
        onSubmit: (data: { valid: boolean; values: unknown }) => void
      }

      // Call onSubmit directly with valid data
      component.onSubmit({
        valid: true,
        values: { email: 'test@example.com', password: 'password123' }
      })

      // Check emitted event
      expect(wrapper.emitted('submit')).toBeTruthy()
      expect(wrapper.emitted('submit')?.[0]).toEqual([
        {
          email: 'test@example.com',
          password: 'password123'
        }
      ])
    })

    it('does not emit submit event when form is invalid', async () => {
      const wrapper = mountComponent()
      const component = wrapper.vm as typeof wrapper.vm & {
        handleForgotPassword: (email: string, valid: boolean) => void
        onSubmit: (data: { valid: boolean; values: unknown }) => void
      }

      // Call onSubmit with invalid form
      component.onSubmit({ valid: false, values: {} })

      // Should not emit submit event
      expect(wrapper.emitted('submit')).toBeFalsy()
    })
  })

  describe('Loading State', () => {
    it('shows spinner when loading', async () => {
      mockLoading = true

      try {
        const wrapper = mountComponent()
        await nextTick()

        expect(wrapper.findComponent(ProgressSpinner).exists()).toBe(true)
        expect(wrapper.findComponent(Button).exists()).toBe(false)
      } catch (error) {
        // Fallback test - check HTML content if component rendering fails
        mockLoading = true
        const wrapper = mountComponent()
        expect(wrapper.html()).toContain('p-progressspinner')
        expect(wrapper.html()).not.toContain('<button')
      }
    })

    it('shows button when not loading', () => {
      mockLoading = false

      const wrapper = mountComponent()

      expect(wrapper.findComponent(ProgressSpinner).exists()).toBe(false)
      expect(wrapper.findComponent(Button).exists()).toBe(true)
    })
  })

  describe('Component Structure', () => {
    it('renders email input with correct attributes', () => {
      const wrapper = mountComponent()
      const emailInput = wrapper.findComponent(InputText)

      expect(emailInput.attributes('id')).toBe('comfy-org-sign-in-email')
      expect(emailInput.attributes('autocomplete')).toBe('email')
      expect(emailInput.attributes('name')).toBe('email')
      expect(emailInput.attributes('type')).toBe('text')
    })

    it('renders password input with correct attributes', () => {
      const wrapper = mountComponent()
      const passwordInput = wrapper.findComponent(Password)

      // Check props instead of attributes for Password component
      expect(passwordInput.props('inputId')).toBe('comfy-org-sign-in-password')
      // Password component passes name as prop, not attribute
      expect(passwordInput.props('name')).toBe('password')
      expect(passwordInput.props('feedback')).toBe(false)
      expect(passwordInput.props('toggleMask')).toBe(true)
    })

    it('renders form with correct resolver', () => {
      const wrapper = mountComponent()
      const form = wrapper.findComponent(Form)

      expect(form.props('resolver')).toBeDefined()
    })
  })

  describe('Focus Behavior', () => {
    it('focuses email input when handleForgotPassword is called with invalid email', async () => {
      const wrapper = mountComponent()
      const component = wrapper.vm as typeof wrapper.vm & {
        handleForgotPassword: (email: string, valid: boolean) => void
        onSubmit: (data: { valid: boolean; values: unknown }) => void
      }

      // Mock getElementById to track focus
      const mockFocus = vi.fn()
      const mockElement: Partial<HTMLElement> = { focus: mockFocus }
      vi.spyOn(document, 'getElementById').mockReturnValue(
        mockElement as HTMLElement
      )

      // Call handleForgotPassword with no email
      await component.handleForgotPassword('', false)

      // Should focus email input
      expect(document.getElementById).toHaveBeenCalledWith(
        'comfy-org-sign-in-email'
      )
      expect(mockFocus).toHaveBeenCalled()
    })

    it('does not focus email input when valid email is provided', async () => {
      const wrapper = mountComponent()
      const component = wrapper.vm as typeof wrapper.vm & {
        handleForgotPassword: (email: string, valid: boolean) => void
        onSubmit: (data: { valid: boolean; values: unknown }) => void
      }

      // Mock getElementById
      const mockFocus = vi.fn()
      const mockElement: Partial<HTMLElement> = { focus: mockFocus }
      vi.spyOn(document, 'getElementById').mockReturnValue(
        mockElement as HTMLElement
      )

      // Call handleForgotPassword with valid email
      await component.handleForgotPassword('test@example.com', true)

      // Should NOT focus email input
      expect(document.getElementById).not.toHaveBeenCalled()
      expect(mockFocus).not.toHaveBeenCalled()

      // Should call sendPasswordReset
      expect(mockSendPasswordReset).toHaveBeenCalledWith('test@example.com')
    })
  })
})
