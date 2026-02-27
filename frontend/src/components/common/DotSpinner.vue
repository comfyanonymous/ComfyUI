<template>
  <div
    class="inline-flex items-center justify-center"
    :style="{ width: size + 'px', height: size + 'px' }"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      :width="size"
      :height="size"
      viewBox="0 0 14 14"
      fill="none"
      class="animate-spin"
      :style="{ animationDuration: duration }"
    >
      <g clip-path="url(#clip0_776_9582)">
        <!-- Top dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M7 2.21053C7.61042 2.21053 8.10526 1.71568 8.10526 1.10526C8.10526 0.494843 7.61042 0 7 0C6.38958 0 5.89474 0.494843 5.89474 1.10526C5.89474 1.71568 6.38958 2.21053 7 2.21053Z"
          :fill="color"
        />
        <!-- Left dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.25s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M2.21053 7C2.21053 7.61042 1.71568 8.10526 1.10526 8.10526C0.494843 8.10526 0 7.61042 0 7C0 6.38958 0.494843 5.89474 1.10526 5.89474C1.71568 5.89474 2.21053 6.38958 2.21053 7Z"
          :fill="color"
        />
        <!-- Right dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.5s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M14 7C14 7.61042 13.5052 8.10526 12.8947 8.10526C12.2843 8.10526 11.7895 7.61042 11.7895 7C11.7895 6.38958 12.2843 5.89474 12.8947 5.89474C13.5052 5.89474 14 6.38958 14 7Z"
          :fill="color"
        />
        <!-- Bottom dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.75s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M8.10526 12.8947C8.10526 13.5052 7.61041 14 6.99999 14C6.38957 14 5.89473 13.5052 5.89473 12.8947C5.89473 12.2843 6.38957 11.7895 6.99999 11.7895C7.61041 11.7895 8.10526 12.2843 8.10526 12.8947Z"
          :fill="color"
        />
        <!-- Top-left dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.125s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M2.05039 3.61349C2.48203 4.04513 3.18184 4.04513 3.61347 3.61349C4.0451 3.18186 4.0451 2.48205 3.61347 2.05042C3.18184 1.61878 2.48203 1.61878 2.05039 2.05042C1.61876 2.48205 1.61876 3.18186 2.05039 3.61349Z"
          :fill="color"
        />
        <!-- Bottom-right dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.625s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M11.9496 11.9496C11.518 12.3812 10.8182 12.3812 10.3865 11.9496C9.9549 11.5179 9.9549 10.8181 10.3865 10.3865C10.8182 9.95485 11.518 9.95485 11.9496 10.3865C12.3812 10.8181 12.3812 11.5179 11.9496 11.9496Z"
          :fill="color"
        />
        <!-- Bottom-left dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.875s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M2.05039 11.9496C2.48203 12.3812 3.18184 12.3812 3.61347 11.9496C4.0451 11.5179 4.0451 10.8181 3.61347 10.3865C3.18184 9.95485 2.48203 9.95485 2.05039 10.3865C1.61876 10.8181 1.61876 11.5179 2.05039 11.9496Z"
          :fill="color"
        />
        <!-- Top-right dot -->
        <path
          class="dot-animation"
          style="animation-delay: 0.375s"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M11.9496 3.61349C11.518 4.04513 10.8182 4.04513 10.3865 3.61349C9.9549 3.18186 9.9549 2.48205 10.3865 2.05042C10.8182 1.61878 11.518 1.61878 11.9496 2.05042C12.3812 2.48205 12.3812 3.18186 11.9496 3.61349Z"
          :fill="color"
        />
      </g>
      <defs>
        <clipPath id="clip0_776_9582">
          <rect width="14" height="14" fill="white" />
        </clipPath>
      </defs>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'

const { size = 24, duration = '2s' } = defineProps<{
  size?: number
  duration?: string
}>()

const colorPaletteStore = useColorPaletteStore()

const color = computed(() =>
  colorPaletteStore.completedActivePalette.light_theme ? '#2C2B30' : '#D4D4D4'
)
</script>

<style scoped>
.dot-animation {
  animation: dot-pulse 1s ease-in-out infinite;
}

@keyframes dot-pulse {
  0%,
  80%,
  100% {
    opacity: 0.3;
  }

  40% {
    opacity: 1;
  }
}
</style>
