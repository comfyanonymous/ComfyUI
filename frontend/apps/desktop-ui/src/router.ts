import {
  createRouter,
  createWebHashHistory,
  createWebHistory
} from 'vue-router'

import { isElectron } from '@/utils/envUtil'
import LayoutDefault from '@/views/layouts/LayoutDefault.vue'

const isFileProtocol = window.location.protocol === 'file:'
const basePath = isElectron() ? '/' : window.location.pathname

const router = createRouter({
  history: isFileProtocol ? createWebHashHistory() : createWebHistory(basePath),
  routes: [
    {
      path: '/',
      component: LayoutDefault,
      children: [
        {
          path: '',
          name: 'WelcomeView',
          component: () => import('@/views/WelcomeView.vue')
        },
        {
          path: 'welcome',
          name: 'WelcomeViewAlias',
          component: () => import('@/views/WelcomeView.vue')
        },
        {
          path: 'install',
          name: 'InstallView',
          component: () => import('@/views/InstallView.vue')
        },
        {
          path: 'download-git',
          name: 'DownloadGitView',
          component: () => import('@/views/DownloadGitView.vue')
        },
        {
          path: 'desktop-start',
          name: 'DesktopStartView',
          component: () => import('@/views/DesktopStartView.vue')
        },
        {
          path: 'desktop-update',
          name: 'DesktopUpdateView',
          component: () => import('@/views/DesktopUpdateView.vue')
        },
        {
          path: 'server-start',
          name: 'ServerStartView',
          component: () => import('@/views/ServerStartView.vue')
        },
        {
          path: 'manual-configuration',
          name: 'ManualConfigurationView',
          component: () => import('@/views/ManualConfigurationView.vue')
        },
        {
          path: 'metrics-consent',
          name: 'MetricsConsentView',
          component: () => import('@/views/MetricsConsentView.vue')
        },
        {
          path: 'maintenance',
          name: 'MaintenanceView',
          component: () => import('@/views/MaintenanceView.vue')
        },
        {
          path: 'not-supported',
          name: 'NotSupportedView',
          component: () => import('@/views/NotSupportedView.vue')
        },
        {
          path: 'desktop-dialog/:dialogId',
          name: 'DesktopDialogView',
          component: () => import('@/views/DesktopDialogView.vue')
        }
      ]
    }
  ],
  scrollBehavior(_to, _from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }

    return { top: 0 }
  }
})

export default router
