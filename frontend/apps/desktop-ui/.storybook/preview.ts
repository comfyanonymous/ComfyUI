import { definePreset } from '@primevue/themes'
import Aura from '@primevue/themes/aura'
import { setup } from '@storybook/vue3'
import type { Preview, StoryContext, StoryFn } from '@storybook/vue3-vite'
import { createPinia } from 'pinia'
import 'primeicons/primeicons.css'
import PrimeVue from 'primevue/config'
import ConfirmationService from 'primevue/confirmationservice'
import ToastService from 'primevue/toastservice'
import Tooltip from 'primevue/tooltip'

import '@/assets/css/style.css'
import { i18n } from '@/i18n'

const ComfyUIPreset = definePreset(Aura, {
  semantic: {
    // @ts-expect-error prime type quirk
    primary: Aura['primitive'].blue
  }
})

setup((app) => {
  app.directive('tooltip', Tooltip)

  const pinia = createPinia()

  app.use(pinia)
  app.use(i18n)
  app.use(PrimeVue, {
    theme: {
      preset: ComfyUIPreset,
      options: {
        prefix: 'p',
        cssLayer: { name: 'primevue', order: 'primevue, tailwind-utilities' },
        darkModeSelector: '.dark-theme, :root:has(.dark-theme)'
      }
    }
  })
  app.use(ConfirmationService)
  app.use(ToastService)
})

export const withTheme = (Story: StoryFn, context: StoryContext) => {
  const theme = context.globals.theme || 'light'
  if (theme === 'dark') {
    document.documentElement.classList.add('dark-theme')
    document.body.classList.add('dark-theme')
  } else {
    document.documentElement.classList.remove('dark-theme')
    document.body.classList.remove('dark-theme')
  }

  return Story(context.args, context)
}

const preview: Preview = {
  parameters: {
    controls: {
      matchers: { color: /(background|color)$/i, date: /Date$/i }
    },
    backgrounds: {
      default: 'light',
      values: [
        { name: 'light', value: '#ffffff' },
        { name: 'dark', value: '#0a0a0a' }
      ]
    }
  },
  globalTypes: {
    theme: {
      name: 'Theme',
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        icon: 'circlehollow',
        items: [
          { value: 'light', icon: 'sun', title: 'Light' },
          { value: 'dark', icon: 'moon', title: 'Dark' }
        ],
        showName: true,
        dynamicTitle: true
      }
    }
  },
  decorators: [withTheme]
}

export default preview
