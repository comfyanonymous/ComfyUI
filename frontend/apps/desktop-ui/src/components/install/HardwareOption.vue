<template>
  <div class="relative">
    <!-- Recommended Badge -->
    <button
      :class="
        cn(
          'hardware-option w-[170px] h-[190px] p-5 flex flex-col items-center rounded-3xl transition-all duration-200 bg-neutral-900/70 border-4',
          selected ? 'border-solid border-brand-yellow' : 'border-transparent'
        )
      "
      @click="$emit('click')"
    >
      <!-- Icon/Logo Area - Rounded square container -->
      <div
        class="icon-container w-[110px] h-[110px] shrink-0 rounded-2xl bg-neutral-800 flex items-center justify-center overflow-hidden"
      >
        <img
          v-if="imagePath"
          :src="imagePath"
          :alt="placeholderText"
          class="w-full h-full object-cover"
          style="object-position: 57% center"
          draggable="false"
        />
        <span v-else class="text-xl font-medium text-neutral-400">
          {{ placeholderText }}
        </span>
      </div>

      <!-- Text Content -->
      <div v-if="subtitle" class="text-center mt-4">
        <div class="text-sm text-neutral-500">{{ subtitle }}</div>
      </div>
    </button>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

interface Props {
  imagePath?: string
  placeholderText: string
  subtitle?: string
  selected?: boolean
}

defineProps<Props>()

defineEmits<{ click: [] }>()
</script>
