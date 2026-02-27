<template>
  <div class="flex flex-col gap-4 text-neutral-400 text-sm">
    <div>
      <h3 class="text-lg font-medium text-neutral-100 mb-3 mt-0">
        {{ $t(`settings.${normalizedSettingId}.name`) }}
      </h3>
      <p class="my-1">
        {{ $t(`settings.${normalizedSettingId}.tooltip`) }}
      </p>
    </div>
    <UrlInput
      v-model="modelValue"
      :validate-url-fn="
        (mirror: string) =>
          checkMirrorReachable(mirror + (item.validationPathSuffix ?? ''))
      "
      @state-change="validationState = $event"
    />
    <div v-if="secondParagraph" class="mt-2">
      <a href="#" @click.prevent="showDialog = true">
        {{ $t('g.learnMore') }}
      </a>

      <Dialog
        v-model:visible="showDialog"
        modal
        dismissable-mask
        :header="$t(`settings.${normalizedSettingId}.urlFormatTitle`)"
        class="select-none max-w-3xl"
      >
        <div class="text-neutral-300">
          <p class="mt-1 whitespace-pre-wrap">{{ secondParagraph }}</p>
          <div class="mt-2 break-all">
            <span class="text-neutral-300 font-semibold">
              {{ EXAMPLE_URL_FIRST_PART }}
            </span>
            <span>{{ EXAMPLE_URL_SECOND_PART }}</span>
          </div>
          <Divider />
          <p>
            {{ $t(`settings.${normalizedSettingId}.fileUrlDescription`) }}
          </p>
          <span class="text-neutral-300 font-semibold">
            {{ FILE_URL_SCHEME }}
          </span>
          <span>
            {{ EXAMPLE_FILE_URL }}
          </span>
        </div>
      </Dialog>
    </div>
  </div>
</template>

<script setup lang="ts">
import { normalizeI18nKey } from '@comfyorg/shared-frontend-utils/formatUtil'
import Dialog from 'primevue/dialog'
import Divider from 'primevue/divider'
import { computed, onMounted, ref, watch } from 'vue'

import UrlInput from '@/components/common/UrlInput.vue'
import type { UVMirror } from '@/constants/uvMirrors'
import { st } from '@/i18n'
import { checkMirrorReachable } from '@/utils/electronMirrorCheck'
import { ValidationState } from '@/utils/validationUtil'

const FILE_URL_SCHEME = 'file://'
const EXAMPLE_FILE_URL = '/C:/MyPythonInstallers/'
const EXAMPLE_URL_FIRST_PART =
  'https://github.com/astral-sh/python-build-standalone/releases/download'
const EXAMPLE_URL_SECOND_PART =
  '/20250902/cpython-3.12.11+20250902-x86_64-pc-windows-msvc-install_only.tar.gz'

const { item } = defineProps<{
  item: UVMirror
}>()

const emit = defineEmits<{
  'state-change': [state: ValidationState]
}>()

const modelValue = defineModel<string>('modelValue', { required: true })
const validationState = ref<ValidationState>(ValidationState.IDLE)
const showDialog = ref(false)

const normalizedSettingId = computed(() => {
  return normalizeI18nKey(item.settingId)
})

const secondParagraph = computed(() =>
  st(`settings.${normalizedSettingId.value}.urlDescription`, '')
)

onMounted(() => {
  modelValue.value = item.mirror
})

watch(validationState, (newState) => {
  emit('state-change', newState)

  // Set fallback mirror if default mirror is invalid
  if (
    newState === ValidationState.INVALID &&
    modelValue.value === item.mirror
  ) {
    modelValue.value = item.fallbackMirror
  }
})
</script>
