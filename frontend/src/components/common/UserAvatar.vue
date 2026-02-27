<template>
  <Avatar
    class="bg-interface-panel-selected-surface"
    :image="photoUrl ?? undefined"
    :icon="hasAvatar ? undefined : 'icon-[lucide--user]'"
    :pt:icon:class="{ 'size-4': !hasAvatar }"
    shape="circle"
    :aria-label="ariaLabel ?? $t('auth.login.userAvatar')"
    @error="handleImageError"
  />
</template>

<script setup lang="ts">
import Avatar from 'primevue/avatar'
import { computed, ref } from 'vue'

const { photoUrl, ariaLabel } = defineProps<{
  photoUrl?: string | null
  ariaLabel?: string
}>()

const imageError = ref(false)
const handleImageError = () => {
  imageError.value = true
}
const hasAvatar = computed(() => photoUrl && !imageError.value)
</script>
