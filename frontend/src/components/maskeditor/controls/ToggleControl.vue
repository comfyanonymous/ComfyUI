<template>
  <div class="flex flex-row gap-2.5 items-center min-h-6 relative">
    <span class="text-left text-xs font-sans text-[var(--descrip-text)]">{{
      label
    }}</span>
    <label class="maskEditor_sidePanelToggleContainer">
      <input
        type="checkbox"
        class="maskEditor_sidePanelToggleCheckbox"
        :checked="modelValue"
        @change="onChange"
      />
      <div class="maskEditor_sidePanelToggleSwitch"></div>
    </label>
  </div>
</template>

<script setup lang="ts">
interface Props {
  label: string
  modelValue: boolean
}

defineProps<Props>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

const onChange = (event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  emit('update:modelValue', checked)
}
</script>
