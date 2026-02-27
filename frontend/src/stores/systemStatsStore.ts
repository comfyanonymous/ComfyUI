import { useAsyncState } from '@vueuse/core'
import { defineStore } from 'pinia'

import { isCloud, isDesktop } from '@/platform/distribution/types'
import type { SystemStats } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'

export const useSystemStatsStore = defineStore('systemStats', () => {
  const fetchSystemStatsData = async () => {
    try {
      return await api.getSystemStats()
    } catch (err) {
      console.error('Error fetching system stats:', err)
      throw err
    }
  }

  const {
    state: systemStats,
    isLoading,
    error,
    isReady: isInitialized,
    execute: refetchSystemStats
  } = useAsyncState<SystemStats | null>(
    fetchSystemStatsData,
    null, // initial value
    {
      immediate: true
    }
  )

  function getFormFactor(): string {
    if (isCloud) {
      return 'cloud'
    }

    if (!systemStats.value?.system?.os) {
      return 'other'
    }

    const os = systemStats.value.system.os.toLowerCase()

    if (isDesktop) {
      if (os.includes('windows')) {
        return 'desktop-windows'
      }
      if (os.includes('darwin') || os.includes('mac')) {
        return 'desktop-mac'
      }
    } else {
      // Git/source installation
      if (os.includes('windows')) {
        return 'git-windows'
      }
      if (os.includes('darwin') || os.includes('mac')) {
        return 'git-mac'
      }
      if (os.includes('linux')) {
        return 'git-linux'
      }
    }

    return 'other'
  }

  return {
    systemStats,
    isLoading,
    error,
    isInitialized,
    refetchSystemStats,
    getFormFactor
  }
})
