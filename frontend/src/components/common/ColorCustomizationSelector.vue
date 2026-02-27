<template>
  <div
    class="color-customization-selector-container flex flex-row items-center gap-2"
  >
    <SelectButton
      v-model="selectedColorOption"
      :options="colorOptionsWithCustom"
      option-label="name"
      data-key="value"
      :allow-empty="false"
    >
      <template #option="slotProps">
        <div
          v-if="slotProps.option.name !== '_custom'"
          :style="{
            width: '20px',
            height: '20px',
            backgroundColor: slotProps.option.value,
            borderRadius: '50%'
          }"
        />
        <i v-else class="pi pi-palette text-lg" />
      </template>
    </SelectButton>
    <ColorPicker
      v-if="selectedColorOption.name === '_custom'"
      v-model="customColorValue"
    />
  </div>
</template>

<script setup lang="ts">
import ColorPicker from 'primevue/colorpicker'
import SelectButton from 'primevue/selectbutton'
import { computed, onMounted, ref, watch } from 'vue'

const {
  modelValue,
  colorOptions,
  allowCustom = true
} = defineProps<{
  modelValue: string | null
  colorOptions: { name: Exclude<string, '_custom'>; value: string }[]
  allowCustom?: boolean
}>()

const customColorOption = { name: '_custom', value: '' }
const colorOptionsWithCustom = computed(() => [
  ...colorOptions,
  ...(allowCustom ? [customColorOption] : [])
])

const emit = defineEmits<{
  'update:modelValue': [value: string | null]
}>()

const selectedColorOption = ref(customColorOption)
const customColorValue = ref('')

// Initialize the component with the provided modelValue
onMounted(() => {
  if (modelValue) {
    const predefinedColor = colorOptions.find((opt) => opt.value === modelValue)
    if (predefinedColor) {
      selectedColorOption.value = predefinedColor
    } else {
      selectedColorOption.value = customColorOption
      customColorValue.value = modelValue.replace('#', '')
    }
  }
})

// Watch for changes in selection and emit updates
watch(selectedColorOption, (newOption, oldOption) => {
  if (newOption.name === '_custom') {
    // Inherit the color from previous selection
    customColorValue.value = oldOption.value.replace('#', '')
  } else {
    emit('update:modelValue', newOption.value)
  }
})

watch(customColorValue, (newValue) => {
  if (selectedColorOption.value.name === '_custom') {
    emit('update:modelValue', newValue ? `#${newValue}` : null)
  }
})
</script>
