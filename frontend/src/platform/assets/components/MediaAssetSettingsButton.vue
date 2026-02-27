<template>
  <div class="inline-flex items-center">
    <Popover>
      <template #button>
        <Button
          variant="secondary"
          size="icon"
          :aria-label="$t('sideToolbar.mediaAssets.viewSettings')"
        >
          <i class="icon-[lucide--settings-2]" />
        </Button>
      </template>
      <template #default="{ close }">
        <slot :close />
      </template>
    </Popover>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import Popover from '@/components/ui/Popover.vue'
</script>
