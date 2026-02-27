<template>
  <div>
    <div class="flex min-h-[638px] min-w-[320px] flex-col">
      <Skeleton width="100%" height="0.5rem" class="mb-8" />

      <div class="flex flex-1 flex-col p-0">
        <div class="flex min-h-full flex-1 flex-col justify-between">
          <div>
            <Skeleton width="70%" height="1.75rem" class="mb-8" />
            <div class="flex flex-col gap-6">
              <div v-for="i in 5" :key="i" class="flex items-center gap-3">
                <Skeleton width="1.25rem" height="1.25rem" shape="circle" />
                <Skeleton width="85%" height="0.875rem" />
              </div>
            </div>
          </div>

          <div class="flex justify-between pt-4">
            <span />
            <Skeleton width="100%" height="2.5rem" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Skeleton from 'primevue/skeleton'
</script>
