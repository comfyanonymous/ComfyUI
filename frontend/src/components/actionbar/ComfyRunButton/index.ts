import { defineAsyncComponent } from 'vue'

import { isCloud } from '@/platform/distribution/types'

export default isCloud && window.__CONFIG__?.subscription_required
  ? defineAsyncComponent(() => import('./CloudRunButtonWrapper.vue'))
  : defineAsyncComponent(() => import('./ComfyQueueButton.vue'))
