import type { RouteRecordRaw } from 'vue-router'

export const cloudOnboardingRoutes: RouteRecordRaw[] = [
  {
    path: '/cloud',
    component: () =>
      import('@/platform/cloud/onboarding/components/CloudLayoutView.vue'),
    children: [
      {
        path: 'login',
        name: 'cloud-login',
        component: () =>
          import('@/platform/cloud/onboarding/CloudLoginView.vue'),
        beforeEnter: async (to, _from, next) => {
          // Only redirect if not explicitly switching accounts
          if (!to.query.switchAccount) {
            const { useCurrentUser } =
              await import('@/composables/auth/useCurrentUser')
            const { isLoggedIn } = useCurrentUser()

            if (isLoggedIn.value) {
              // User is already logged in, redirect to user-check
              // user-check will handle survey, or main page routing
              return next({ name: 'cloud-user-check' })
            }
          }
          next()
        }
      },
      {
        path: 'signup',
        name: 'cloud-signup',
        component: () =>
          import('@/platform/cloud/onboarding/CloudSignupView.vue')
      },
      {
        path: 'forgot-password',
        name: 'cloud-forgot-password',
        component: () =>
          import('@/platform/cloud/onboarding/CloudForgotPasswordView.vue')
      },
      {
        path: 'survey',
        name: 'cloud-survey',
        component: () =>
          import('@/platform/cloud/onboarding/CloudSurveyView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'user-check',
        name: 'cloud-user-check',
        component: () =>
          import('@/platform/cloud/onboarding/UserCheckView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'sorry-contact-support',
        name: 'cloud-sorry-contact-support',
        component: () =>
          import('@/platform/cloud/onboarding/CloudSorryContactSupportView.vue')
      },
      {
        path: 'auth-timeout',
        name: 'cloud-auth-timeout',
        component: () =>
          import('@/platform/cloud/onboarding/CloudAuthTimeoutView.vue'),
        props: true
      },
      {
        path: 'subscribe',
        name: 'cloud-subscribe',
        component: () =>
          import('@/platform/cloud/onboarding/CloudSubscriptionRedirectView.vue'),
        meta: { requiresAuth: true }
      }
    ]
  }
]
