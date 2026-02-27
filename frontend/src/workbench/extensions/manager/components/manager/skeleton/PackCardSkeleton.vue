<template>
  <div
    class="flex h-full flex-col overflow-hidden rounded-lg shadow-sm"
    data-virtual-grid-item
  >
    <!-- Card header - flush with top, approximately 15% of height -->
    <div class="flex w-full items-center justify-between px-4 py-3">
      <div class="flex items-center">
        <div class="flex h-6 w-6 items-center justify-center">
          <Skeleton shape="circle" width="1.5rem" height="1.5rem" />
        </div>
        <Skeleton width="5rem" height="1rem" class="ml-2" />
      </div>
      <Skeleton width="4rem" height="1.75rem" border-radius="0.75rem" />
    </div>

    <!-- Card content with icon on left and text on right -->
    <div class="flex flex-1 p-4">
      <!-- Left icon - 64x64 -->
      <div class="mr-4 shrink-0">
        <Skeleton width="4rem" height="4rem" border-radius="0.5rem" />
      </div>

      <!-- Right content -->
      <div class="flex flex-1 flex-col overflow-hidden">
        <!-- Title -->
        <Skeleton width="80%" height="1rem" class="mb-2" />

        <!-- Description -->
        <div class="mb-3">
          <Skeleton width="100%" height="0.75rem" class="mb-1" />
          <Skeleton width="95%" height="0.75rem" class="mb-1" />
          <Skeleton width="90%" height="0.75rem" />
        </div>

        <!-- Tags/Badges -->
        <div class="flex gap-2">
          <Skeleton width="4rem" height="1.5rem" border-radius="0.75rem" />
          <Skeleton width="5rem" height="1.5rem" border-radius="0.75rem" />
        </div>
      </div>
    </div>

    <!-- Card footer - similar to header -->
    <div class="flex w-full items-center justify-between px-5 py-4">
      <Skeleton width="4rem" height="0.8rem" />
      <Skeleton width="6rem" height="0.8rem" />
    </div>
  </div>
</template>

<script setup lang="ts">
import Skeleton from 'primevue/skeleton'
</script>
