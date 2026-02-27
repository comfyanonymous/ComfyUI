<script setup lang="ts">
import NodeBadge from '@/renderer/extensions/vueNodes/components/NodeBadge.vue'
import type { NodeBadgeProps } from '@/renderer/extensions/vueNodes/components/NodeBadge.vue'

defineProps<{
  hasComfyBadge: boolean
  core: NodeBadgeProps[]
  extension: NodeBadgeProps[]
}>()
</script>
<template>
  <div
    v-if="hasComfyBadge || core.length || extension.length"
    class="flex h-5 w-full gap-2 px-2 text-muted-foreground"
  >
    <div
      v-if="hasComfyBadge"
      class="rounded-full bg-component-node-widget-background size-6 flex justify-center items-center"
    >
      <i class="icon-[comfy--comfy-c] size-3" />
    </div>
    <div
      v-if="core.length"
      class="rounded-full bg-component-node-widget-background h-6 flex justify-center items-center overflow-clip"
    >
      <template v-for="(badge, index) of core" :key="badge.text">
        <div
          v-if="index !== 0"
          class="border-muted-foreground border-r h-4 mr-0.5 pr-0.5"
        />
        <NodeBadge
          bg-color="transparent"
          v-bind="badge"
          class="h-6 first:pl-2 last:pr-2"
        />
      </template>
    </div>
    <div
      v-if="extension.length"
      class="rounded-full bg-component-node-widget-background h-6 flex justify-center items-center overflow-clip"
    >
      <template v-for="(badge, index) of extension" :key="badge.text">
        <div
          v-if="index !== 0"
          class="border-muted-foreground border-r h-4 mr-0.5 pr-0.5"
        />
        <NodeBadge
          bg-color="transparent"
          v-bind="badge"
          class="h-6 first:pl-2 last:pr-2"
        />
      </template>
    </div>
  </div>
</template>
