<template>
  <Dialog v-model:visible="visible" :header="$t('g.customizeFolder')">
    <div class="p-fluid">
      <div class="field icon-field">
        <label for="icon">{{ $t('g.icon') }}</label>
        <SelectButton
          v-model="selectedIcon"
          :options="iconOptions"
          option-label="name"
          data-key="value"
        >
          <template #option="slotProps">
            <i
              :class="['pi', slotProps.option.value, 'mr-2']"
              :style="{ color: finalColor }"
            />
          </template>
        </SelectButton>
      </div>
      <Divider />
      <div class="field color-field">
        <label for="color">{{ $t('g.color') }}</label>
        <ColorCustomizationSelector
          v-model="finalColor"
          :color-options="colorOptions"
        />
      </div>
    </div>
    <template #footer>
      <Button variant="textonly" @click="resetCustomization">
        <i class="pi pi-refresh" />
        {{ $t('g.reset') }}
      </Button>
      <Button autofocus @click="confirmCustomization">
        <i class="pi pi-check" />
        {{ $t('g.confirm') }}
      </Button>
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import Dialog from 'primevue/dialog'
import Divider from 'primevue/divider'
import SelectButton from 'primevue/selectbutton'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import ColorCustomizationSelector from '@/components/common/ColorCustomizationSelector.vue'
import Button from '@/components/ui/button/Button.vue'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'

const { t } = useI18n()

const props = defineProps<{
  modelValue: boolean
  initialIcon?: string
  initialColor?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', icon: string, color: string): void
}>()

const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const nodeBookmarkStore = useNodeBookmarkStore()

const iconOptions = [
  { name: t('icon.bookmark'), value: nodeBookmarkStore.defaultBookmarkIcon },
  { name: t('icon.folder'), value: 'pi-folder' },
  { name: t('icon.star'), value: 'pi-star' },
  { name: t('icon.heart'), value: 'pi-heart' },
  { name: t('icon.file'), value: 'pi-file' },
  { name: t('icon.inbox'), value: 'pi-inbox' },
  { name: t('icon.box'), value: 'pi-box' },
  { name: t('icon.briefcase'), value: 'pi-briefcase' }
]

const colorOptions = [
  { name: t('color.default'), value: nodeBookmarkStore.defaultBookmarkColor },
  { name: t('color.blue'), value: '#007bff' },
  { name: t('color.green'), value: '#28a745' },
  { name: t('color.red'), value: '#dc3545' },
  { name: t('color.pink'), value: '#e83e8c' },
  { name: t('color.yellow'), value: '#ffc107' }
]

const defaultIcon = iconOptions.find(
  (option) => option.value === nodeBookmarkStore.defaultBookmarkIcon
)

// @ts-expect-error fixme ts strict error
const selectedIcon = ref<{ name: string; value: string }>(defaultIcon)
const finalColor = ref(
  props.initialColor || nodeBookmarkStore.defaultBookmarkColor
)

const resetCustomization = () => {
  // @ts-expect-error fixme ts strict error
  selectedIcon.value =
    iconOptions.find((option) => option.value === props.initialIcon) ||
    defaultIcon
  finalColor.value =
    props.initialColor || nodeBookmarkStore.defaultBookmarkColor
}

const confirmCustomization = () => {
  emit('confirm', selectedIcon.value.value, finalColor.value)
  closeDialog()
}

const closeDialog = () => {
  visible.value = false
}

watch(
  () => props.modelValue,
  (newValue: boolean) => {
    if (newValue) {
      resetCustomization()
    }
  },
  { immediate: true }
)
</script>

<style scoped>
.p-selectbutton .p-button {
  padding: 0.5rem;
}

.p-selectbutton .p-button .pi {
  font-size: 1.5rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
</style>
