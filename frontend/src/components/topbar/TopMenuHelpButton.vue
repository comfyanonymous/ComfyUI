<template>
  <Button
    class="comfy-help-center-btn relative text-base-foreground"
    variant="textonly"
    @click="toggleHelpCenter"
  >
    <div class="not-md:hidden">{{ $t('menu.helpAndFeedback') }}</div>
    <i class="icon-[lucide--circle-help] ml-0.5" />
    <span
      v-if="shouldShowRedDot"
      class="absolute top-[7px] right-[7px] size-1.5 rounded-full bg-[#ff3b30]"
    />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useHelpCenter } from '@/composables/useHelpCenter'

const { shouldShowRedDot, toggleHelpCenter } = useHelpCenter('topbar')
</script>
