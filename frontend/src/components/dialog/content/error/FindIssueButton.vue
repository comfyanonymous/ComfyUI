<template>
  <Button variant="secondary" @click="openGitHubIssues">
    <i class="pi pi-github" />
    {{ $t('g.findIssues') }}
  </Button>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useTelemetry } from '@/platform/telemetry'

const props = defineProps<{
  errorMessage: string
  repoOwner: string
  repoName: string
}>()

const queryString = computed(() => props.errorMessage + ' is:issue')

/**
 * Open GitHub issues search and track telemetry.
 */
const openGitHubIssues = () => {
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'error_dialog_find_existing_issues_clicked'
  })
  const query = encodeURIComponent(queryString.value)
  const url = `https://github.com/${props.repoOwner}/${props.repoName}/issues?q=${query}`
  window.open(url, '_blank')
}
</script>
