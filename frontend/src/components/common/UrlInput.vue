<template>
  <IconField class="w-full">
    <InputText
      v-bind="$attrs"
      :model-value="internalValue"
      class="w-full"
      :invalid="validationState === ValidationState.INVALID"
      @update:model-value="handleInput"
      @blur="handleBlur"
    />
    <InputIcon
      :class="{
        'pi pi-spin pi-spinner text-neutral-400':
          validationState === ValidationState.LOADING,
        'pi pi-check cursor-pointer text-green-500':
          validationState === ValidationState.VALID,
        'pi pi-times cursor-pointer text-red-500':
          validationState === ValidationState.INVALID
      }"
      @click="validateUrl(props.modelValue)"
    />
  </IconField>
</template>

<script setup lang="ts">
import IconField from 'primevue/iconfield'
import InputIcon from 'primevue/inputicon'
import InputText from 'primevue/inputtext'
import { onMounted, ref, watch } from 'vue'

import { isValidUrl } from '@/utils/formatUtil'
import { checkUrlReachable } from '@/utils/networkUtil'
import { ValidationState } from '@/utils/validationUtil'

const props = defineProps<{
  modelValue: string
  validateUrlFn?: (url: string) => Promise<boolean>
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  'state-change': [state: ValidationState]
}>()

const validationState = ref<ValidationState>(ValidationState.IDLE)

const cleanInput = (value: string): string =>
  value ? value.replace(/\s+/g, '') : ''

// Add internal value state
const internalValue = ref(cleanInput(props.modelValue))

// Watch for external modelValue changes
watch(
  () => props.modelValue,
  async (newValue: string) => {
    internalValue.value = cleanInput(newValue)
    await validateUrl(newValue)
  }
)

watch(validationState, (newState) => {
  emit('state-change', newState)
})

// Validate on mount
onMounted(async () => {
  await validateUrl(props.modelValue)
})

const handleInput = (value: string | undefined) => {
  // Update internal value without emitting
  internalValue.value = cleanInput(value ?? '')
  // Reset validation state when user types
  validationState.value = ValidationState.IDLE
}

const handleBlur = async () => {
  const input = cleanInput(internalValue.value)

  let normalizedUrl = input
  try {
    const url = new URL(input)
    normalizedUrl = url.toString()
  } catch {
    // If URL parsing fails, just use the cleaned input
  }

  // Emit the update only on blur
  emit('update:modelValue', normalizedUrl)
}

// Default validation implementation
const defaultValidateUrl = async (url: string): Promise<boolean> => {
  if (!isValidUrl(url)) return false
  try {
    return await checkUrlReachable(url)
  } catch {
    return false
  }
}

const validateUrl = async (value: string) => {
  if (validationState.value === ValidationState.LOADING) return

  const url = cleanInput(value)

  // Reset state
  validationState.value = ValidationState.IDLE

  // Skip validation if empty
  if (!url) return

  validationState.value = ValidationState.LOADING
  try {
    const isValid = await (props.validateUrlFn ?? defaultValidateUrl)(url)
    validationState.value = isValid
      ? ValidationState.VALID
      : ValidationState.INVALID
  } catch {
    validationState.value = ValidationState.INVALID
  }
}

// Add inheritAttrs option to prevent attrs from being applied to root element
defineOptions({
  inheritAttrs: false
})
</script>
