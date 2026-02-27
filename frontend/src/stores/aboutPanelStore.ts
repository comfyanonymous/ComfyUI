import { defineStore } from 'pinia'
import { computed } from 'vue'

import { useExternalLink } from '@/composables/useExternalLink'
import { isCloud, isDesktop } from '@/platform/distribution/types'
import type { AboutPageBadge } from '@/types/comfy'
import { electronAPI } from '@/utils/envUtil'
import { formatCommitHash } from '@/utils/formatUtil'

import { useExtensionStore } from './extensionStore'
import { useSystemStatsStore } from './systemStatsStore'

export const useAboutPanelStore = defineStore('aboutPanel', () => {
  const frontendVersion = __COMFYUI_FRONTEND_VERSION__
  const extensionStore = useExtensionStore()
  const systemStatsStore = useSystemStatsStore()
  const { staticUrls } = useExternalLink()
  const coreVersion = computed(
    () => systemStatsStore?.systemStats?.system?.comfyui_version ?? ''
  )
  const templatesVersion = computed(
    () =>
      systemStatsStore?.systemStats?.system?.installed_templates_version ?? ''
  )
  const requiredTemplatesVersion = computed(
    () =>
      systemStatsStore?.systemStats?.system?.required_templates_version ?? ''
  )
  const isTemplatesOutdated = computed(
    () =>
      templatesVersion.value !== '' &&
      requiredTemplatesVersion.value !== '' &&
      templatesVersion.value !== requiredTemplatesVersion.value
  )

  const coreBadges = computed<AboutPageBadge[]>(() => [
    // In electron, the ComfyUI is packaged without the git repo,
    // so the python server's API doesn't have the version info.
    {
      label: `ComfyUI ${
        isDesktop
          ? 'v' + electronAPI().getComfyUIVersion()
          : formatCommitHash(coreVersion.value)
      }`,
      url: isCloud ? staticUrls.comfyOrg : staticUrls.github,
      icon: isCloud ? 'pi pi-cloud' : 'pi pi-github'
    },
    {
      label: `ComfyUI_frontend v${frontendVersion}`,
      url: staticUrls.githubFrontend,
      icon: 'pi pi-github'
    },
    ...(templatesVersion.value
      ? [
          {
            label: `Templates v${templatesVersion.value}`,
            url: 'https://pypi.org/project/comfyui-workflow-templates/',
            icon: 'pi pi-book',
            ...(isTemplatesOutdated.value
              ? { severity: 'danger' as const }
              : {})
          }
        ]
      : []),
    {
      label: 'Discord',
      url: staticUrls.discord,
      icon: 'pi pi-discord'
    },
    { label: 'ComfyOrg', url: staticUrls.comfyOrg, icon: 'pi pi-globe' }
  ])

  const allBadges = computed<AboutPageBadge[]>(() => [
    ...coreBadges.value,
    ...extensionStore.extensions.flatMap((e) => e.aboutPageBadges ?? [])
  ])

  return {
    badges: allBadges
  }
})
