<template>
  <div class="flex flex-col gap-2 flex-auto">
    <ComboboxRoot :ignore-filter="true" :open="false">
      <ComboboxAnchor
        :class="
          cn(
            'relative flex w-full cursor-text items-center',
            'rounded-lg bg-comfy-input text-comfy-input-foreground',
            showBorder &&
              'border border-solid border-border-default box-border',
            sizeClasses,
            className
          )
        "
      >
        <i
          v-if="!searchTerm"
          :class="cn('absolute left-4 pointer-events-none', icon, iconClass)"
        />
        <Button
          v-else
          class="absolute left-2"
          variant="textonly"
          size="icon"
          :aria-label="$t('g.clear')"
          @click="clearSearch"
        >
          <i class="icon-[lucide--x] size-4" />
        </Button>

        <ComboboxInput
          ref="inputRef"
          v-model="searchTerm"
          :class="
            cn(
              'size-full border-none bg-transparent text-sm outline-none',
              inputPadding
            )
          "
          :placeholder="placeholderText"
          :auto-focus="autofocus"
        />
      </ComboboxAnchor>
    </ComboboxRoot>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'
import { watchDebounced } from '@vueuse/core'
import { ComboboxAnchor, ComboboxInput, ComboboxRoot } from 'reka-ui'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'

const { t } = useI18n()

const {
  placeholder,
  icon = 'icon-[lucide--search]',
  debounceTime = 300,
  autofocus = false,
  showBorder = false,
  size = 'md',
  class: className
} = defineProps<{
  placeholder?: string
  icon?: string
  debounceTime?: number
  autofocus?: boolean
  showBorder?: boolean
  size?: 'md' | 'lg'
  class?: string
}>()

const emit = defineEmits<{
  search: [value: string]
}>()

const searchTerm = defineModel<string>({ required: true })

const inputRef = ref<InstanceType<typeof ComboboxInput> | null>(null)

defineExpose({
  focus: () => {
    inputRef.value?.$el?.focus()
  }
})

const isLarge = computed(() => size === 'lg')
const placeholderText = computed(
  () => placeholder ?? t('g.searchPlaceholder', { subject: '' })
)

const sizeClasses = computed(() => {
  if (showBorder) {
    return isLarge.value ? 'h-10 p-2' : 'h-8 p-2'
  }
  return isLarge.value ? 'h-12 px-4 py-2' : 'h-10 px-4 py-2'
})

const iconClass = computed(() => (isLarge.value ? 'size-5' : 'size-4'))
const inputPadding = computed(() => (isLarge.value ? 'pl-8' : 'pl-6'))

function clearSearch() {
  searchTerm.value = ''
}

watchDebounced(
  searchTerm,
  (value: string) => {
    emit('search', value)
  },
  { debounce: debounceTime }
)
</script>
