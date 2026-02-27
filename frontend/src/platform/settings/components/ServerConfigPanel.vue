<template>
  <div class="server-config-panel flex flex-col gap-2">
    <Message v-if="modifiedConfigs.length > 0" severity="info" pt:text="w-full">
      <p>
        {{ $t('serverConfig.modifiedConfigs') }}
      </p>
      <ul>
        <li v-for="config in modifiedConfigs" :key="config.id">
          {{ config.name }}: {{ config.initialValue }} → {{ config.value }}
        </li>
      </ul>
      <div class="flex justify-end gap-2">
        <Button variant="secondary" @click="revertChanges">
          {{ $t('serverConfig.revertChanges') }}
        </Button>
        <Button variant="destructive" @click="restartApp">
          {{ $t('serverConfig.restart') }}
        </Button>
      </div>
    </Message>
    <Message v-if="commandLineArgs" severity="secondary" pt:text="w-full">
      <template #icon>
        <i class="icon-[lucide--terminal] text-xl font-bold" />
      </template>
      <div class="flex items-center justify-between">
        <p>{{ commandLineArgs }}</p>
        <Button
          size="icon"
          variant="muted-textonly"
          :aria-label="$t('g.copyToClipboard')"
          @click="copyCommandLineArgs"
        >
          <i class="pi pi-clipboard" />
        </Button>
      </div>
    </Message>
    <div
      v-for="([label, items], i) in Object.entries(serverConfigsByCategory)"
      :key="label"
    >
      <Divider v-if="i > 0" />
      <h3>{{ $t(`serverConfigCategories.${label}`, label) }}</h3>
      <div v-for="item in items" :key="item.name" class="mb-4">
        <FormItem
          :id="item.id"
          v-model:form-value="item.value"
          :item="translateItem(item)"
          :label-class="{
            'text-highlight': item.initialValue !== item.value
          }"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Divider from 'primevue/divider'
import Message from 'primevue/message'
import { onBeforeUnmount, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import FormItem from '@/components/common/FormItem.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import type { ServerConfig, ServerConfigValue } from '@/constants/serverConfig'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { FormItem as FormItemType } from '@/platform/settings/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useServerConfigStore } from '@/stores/serverConfigStore'
import { electronAPI } from '@/utils/envUtil'

const settingStore = useSettingStore()
const serverConfigStore = useServerConfigStore()
const toastStore = useToastStore()
const {
  serverConfigsByCategory,
  serverConfigValues,
  launchArgs,
  commandLineArgs,
  modifiedConfigs
} = storeToRefs(serverConfigStore)

let restartTriggered = false

const revertChanges = () => {
  serverConfigStore.revertChanges()
}

const restartApp = async () => {
  restartTriggered = true
  await electronAPI().restartApp()
}

watch(launchArgs, async (newVal) => {
  await settingStore.set('Comfy.Server.LaunchArgs', newVal)
})

watch(serverConfigValues, async (newVal) => {
  await settingStore.set('Comfy.Server.ServerConfigValues', newVal)
})

const { copyToClipboard } = useCopyToClipboard()
const copyCommandLineArgs = async () => {
  await copyToClipboard(commandLineArgs.value)
}

const { t } = useI18n()

onBeforeUnmount(() => {
  if (restartTriggered) {
    return
  }

  if (modifiedConfigs.value.length === 0) {
    return
  }

  toastStore.add({
    severity: 'warn',
    summary: t('serverConfig.restartRequiredToastSummary'),
    detail: t('serverConfig.restartRequiredToastDetail'),
    life: 10_000
  })
})

const translateItem = (item: ServerConfig<ServerConfigValue>): FormItemType => {
  return {
    ...item,
    name: t(`serverConfigItems.${item.id}.name`, item.name),
    tooltip: item.tooltip
      ? t(`serverConfigItems.${item.id}.tooltip`, item.tooltip)
      : undefined
  }
}
</script>
