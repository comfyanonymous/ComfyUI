import { definePreset } from '@primevue/themes'
import Aura from '@primevue/themes/aura'
import * as Sentry from '@sentry/vue'
import { initializeApp } from 'firebase/app'
import { createPinia } from 'pinia'
import 'primeicons/primeicons.css'
import PrimeVue from 'primevue/config'
import ConfirmationService from 'primevue/confirmationservice'
import ToastService from 'primevue/toastservice'
import Tooltip from 'primevue/tooltip'
import { createApp } from 'vue'
import { VueFire, VueFireAuth } from 'vuefire'

import { getFirebaseConfig } from '@/config/firebase'
import '@/lib/litegraph/public/css/litegraph.css'
import router from '@/router'
import { useBootstrapStore } from '@/stores/bootstrapStore'

import App from './App.vue'
// Intentionally relative import to ensure the CSS is loaded in the right order (after litegraph.css)
import './assets/css/style.css'
import { i18n } from './i18n'

/**
 * CRITICAL: Load remote config FIRST for cloud builds to ensure
 * window.__CONFIG__is available for all modules during initialization
 */
const isCloud = __DISTRIBUTION__ === 'cloud'

if (isCloud) {
  const { refreshRemoteConfig } =
    await import('@/platform/remoteConfig/refreshRemoteConfig')
  await refreshRemoteConfig({ useAuth: false })

  const { initTelemetry } = await import('@/platform/telemetry/initTelemetry')
  await initTelemetry()
}

const ComfyUIPreset = definePreset(Aura, {
  semantic: {
    // @ts-expect-error fixme ts strict error
    primary: Aura['primitive'].blue
  }
})

const firebaseApp = initializeApp(getFirebaseConfig())

const app = createApp(App)
const pinia = createPinia()

Sentry.init({
  app,
  dsn: __SENTRY_DSN__,
  enabled: __SENTRY_ENABLED__,
  release: __COMFYUI_FRONTEND_VERSION__,
  normalizeDepth: 8,
  tracesSampleRate: isCloud ? 1.0 : 0,
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 0,
  // Only set these for non-cloud builds
  ...(isCloud
    ? {}
    : {
        integrations: [],
        autoSessionTracking: false,
        defaultIntegrations: false
      })
})
app.directive('tooltip', Tooltip)
app
  .use(router)
  .use(PrimeVue, {
    theme: {
      preset: ComfyUIPreset,
      options: {
        prefix: 'p',
        cssLayer: {
          name: 'primevue',
          order: 'theme, base, primevue'
        },
        // This is a workaround for the issue with the dark mode selector
        // https://github.com/primefaces/primevue/issues/5515
        darkModeSelector: '.dark-theme, :root:has(.dark-theme)'
      }
    }
  })
  .use(ConfirmationService)
  .use(ToastService)
  .use(pinia)
  .use(i18n)
  .use(VueFire, {
    firebaseApp,
    modules: [VueFireAuth()]
  })

const bootstrapStore = useBootstrapStore(pinia)
void bootstrapStore.startStoreBootstrap()

app.mount('#vue-app')
