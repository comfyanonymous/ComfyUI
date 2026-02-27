<template>
  <div>
    <ModelInfoField :label="t('g.description')">
      <MarkdownText
        v-if="nodePack.description"
        :text="nodePack.description"
        class="text-muted-foreground"
      />
      <span v-else class="text-muted-foreground italic">
        {{ t('manager.noDescription') }}
      </span>
    </ModelInfoField>
    <ModelInfoField v-if="nodePack.repository" :label="t('manager.repository')">
      <a
        :href="nodePack.repository"
        target="_blank"
        rel="noopener noreferrer"
        class="inline-flex items-center gap-1.5 text-muted-foreground no-underline transition-colors hover:text-foreground"
      >
        <i
          v-if="isGitHubLink(nodePack.repository)"
          class="pi pi-github text-base"
        />
        <span class="break-all">{{ nodePack.repository }}</span>
        <i class="icon-[lucide--external-link] size-4 shrink-0" />
      </a>
    </ModelInfoField>
    <ModelInfoField v-if="licenseInfo" :label="t('manager.license')">
      <a
        v-if="licenseInfo.isUrl"
        :href="licenseInfo.text"
        target="_blank"
        rel="noopener noreferrer"
        class="inline-flex items-center gap-1.5 text-muted-foreground no-underline transition-colors hover:text-foreground"
      >
        <span class="break-all">{{ licenseInfo.text }}</span>
        <i class="icon-[lucide--external-link] size-4 shrink-0" />
      </a>
      <span v-else class="text-muted-foreground break-all">
        {{ licenseInfo.text }}
      </span>
    </ModelInfoField>
    <ModelInfoField
      v-if="nodePack.latest_version?.dependencies?.length"
      :label="t('manager.dependencies')"
    >
      <div
        v-for="(dep, index) in nodePack.latest_version.dependencies"
        :key="index"
        class="break-words text-muted-foreground"
      >
        {{ dep }}
      </div>
    </ModelInfoField>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import ModelInfoField from '@/platform/assets/components/modelInfo/ModelInfoField.vue'
import type { components } from '@/types/comfyRegistryTypes'
import { isValidUrl } from '@/utils/formatUtil'
import MarkdownText from '@/workbench/extensions/manager/components/manager/infoPanel/MarkdownText.vue'

const { t } = useI18n()

const { nodePack } = defineProps<{
  nodePack: components['schemas']['Node']
}>()

const isGitHubLink = (url: string): boolean => url.includes('github.com')

const isLicenseFile = (filename: string): boolean => {
  // Match LICENSE, LICENSE.md, LICENSE.txt (case insensitive)
  const licensePattern = /^license(\.md|\.txt)?$/i
  return licensePattern.test(filename)
}

const extractBaseRepoUrl = (repoUrl: string): string => {
  const githubRepoPattern = /^(https?:\/\/github\.com\/[^/]+\/[^/]+)/i
  const match = repoUrl.match(githubRepoPattern)
  return match ? match[1] : repoUrl
}

const createLicenseUrl = (filename: string, repoUrl: string): string => {
  if (!repoUrl || !filename) return ''

  const licenseFile = isLicenseFile(filename) ? filename : 'LICENSE'
  const baseRepoUrl = extractBaseRepoUrl(repoUrl)
  return `${baseRepoUrl}/blob/main/${licenseFile}`
}

interface LicenseObject {
  file?: string
  text?: string
}

const parseLicenseObject = (
  licenseObj: LicenseObject
): { text: string; isUrl: boolean } => {
  const licenseFile = licenseObj.file || licenseObj.text

  if (
    typeof licenseFile === 'string' &&
    isLicenseFile(licenseFile) &&
    nodePack.repository
  ) {
    const url = createLicenseUrl(licenseFile, nodePack.repository)
    return {
      text: url,
      isUrl: !!url && isValidUrl(url)
    }
  } else if (licenseObj.text) {
    return {
      text: licenseObj.text,
      isUrl: false
    }
  } else if (typeof licenseFile === 'string') {
    // Return the license file name if repository is missing
    return {
      text: licenseFile,
      isUrl: false
    }
  }
  return {
    text: JSON.stringify(licenseObj),
    isUrl: false
  }
}

const formatLicense = (
  license: string
): { text: string; isUrl: boolean } | null => {
  // Treat "{}" JSON string as undefined
  if (license === '{}') return null

  try {
    const licenseObj = JSON.parse(license)
    // Handle empty object case
    if (Object.keys(licenseObj).length === 0) {
      return null
    }
    return parseLicenseObject(licenseObj)
  } catch (e) {
    if (isLicenseFile(license) && nodePack.repository) {
      const url = createLicenseUrl(license, nodePack.repository)
      return {
        text: url,
        isUrl: !!url && isValidUrl(url)
      }
    }
    return {
      text: license,
      isUrl: false
    }
  }
}

const licenseInfo = computed(() => {
  if (!nodePack.license) return null
  return formatLicense(nodePack.license)
})
</script>
