<template>
  <div
    class="flex size-8 items-center justify-center rounded-md text-base font-semibold text-white"
    :style="{
      background: gradient,
      textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)'
    }"
  >
    {{ letter }}
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const { workspaceName } = defineProps<{
  workspaceName: string
}>()

const letter = computed(() => workspaceName?.charAt(0)?.toUpperCase() ?? '?')

const gradient = computed(() => {
  const seed = letter.value.charCodeAt(0)

  function mulberry32(a: number) {
    return function () {
      let t = (a += 0x6d2b79f5)
      t = Math.imul(t ^ (t >>> 15), t | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }

  const rand = mulberry32(seed)

  const hue1 = Math.floor(rand() * 360)
  const hue2 = (hue1 + 40 + Math.floor(rand() * 80)) % 360
  const sat = 65 + Math.floor(rand() * 20)
  const light = 55 + Math.floor(rand() * 15)

  return `linear-gradient(135deg, hsl(${hue1}, ${sat}%, ${light}%), hsl(${hue2}, ${sat}%, ${light}%))`
})
</script>
