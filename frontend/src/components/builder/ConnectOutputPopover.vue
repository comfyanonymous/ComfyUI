<template>
  <PopoverRoot>
    <PopoverTrigger as-child>
      <slot />
    </PopoverTrigger>
    <PopoverContent
      side="bottom"
      :side-offset="8"
      :collision-padding="10"
      class="z-[1001] w-80 rounded-xl border border-border-default bg-base-background shadow-interface will-change-[transform,opacity] data-[state=open]:data-[side=bottom]:animate-slideUpAndFade"
    >
      <div class="flex h-12 items-center justify-between px-4">
        <h3 class="text-sm font-medium text-base-foreground">
          {{ t('builderToolbar.connectOutput') }}
        </h3>
        <PopoverClose
          :aria-label="t('g.close')"
          class="flex cursor-pointer appearance-none items-center justify-center border-none bg-transparent p-0 text-muted-foreground hover:text-base-foreground"
        >
          <i class="icon-[lucide--x] size-4" />
        </PopoverClose>
      </div>
      <div class="border-t border-border-default" />
      <p class="mt-3 px-4 text-xs text-muted-foreground leading-relaxed">
        {{ t('builderToolbar.connectOutputBody1') }}
      </p>
      <p
        v-if="!isSelectActive"
        class="mt-2 px-4 text-xs text-muted-foreground leading-relaxed"
      >
        {{ t('builderToolbar.connectOutputBody2') }}
      </p>
      <div class="mt-4 flex items-center justify-end gap-2 px-4 pb-4">
        <template v-if="isSelectActive">
          <PopoverClose as-child>
            <Button variant="secondary" size="md">
              {{ t('g.ok') }}
            </Button>
          </PopoverClose>
        </template>
        <template v-else>
          <PopoverClose as-child>
            <Button variant="muted-textonly" size="md">
              {{ t('g.cancel') }}
            </Button>
          </PopoverClose>
          <PopoverClose as-child>
            <Button variant="secondary" size="md" @click="emit('switch')">
              {{ t('builderToolbar.switchToSelect') }}
            </Button>
          </PopoverClose>
        </template>
      </div>
    </PopoverContent>
  </PopoverRoot>
</template>

<script setup lang="ts">
import {
  PopoverClose,
  PopoverContent,
  PopoverRoot,
  PopoverTrigger
} from 'reka-ui'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'

const { isSelectActive = false } = defineProps<{
  isSelectActive?: boolean
}>()

const { t } = useI18n()

const emit = defineEmits<{
  switch: []
}>()
</script>
