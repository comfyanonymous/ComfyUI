<template>
  <div class="flex h-full items-center justify-center p-8">
    <div class="max-w-[100vw] lg:w-96">
      <div class="rounded-lg bg-[#2d2e32] p-4">
        <Skeleton width="60%" height="1.125rem" class="mb-2" />
        <Skeleton width="90%" height="1rem" class="mb-2" />
        <Skeleton width="80%" height="1rem" />
      </div>

      <div class="mt-6 mb-8 flex flex-col gap-4">
        <Skeleton width="45%" height="1.5rem" class="my-0" />
        <div class="flex items-center">
          <Skeleton width="25%" height="1rem" class="mr-1" />
          <Skeleton width="20%" height="1rem" />
        </div>
      </div>

      <div class="mb-8">
        <Skeleton width="20%" height="1rem" class="mb-2" />
        <Skeleton width="100%" height="2.5rem" class="mb-4" />
        <Skeleton width="25%" height="1rem" class="mb-4" />
        <Skeleton width="100%" height="2.5rem" class="mb-6" />
        <Skeleton width="80%" height="1rem" class="mb-4" />
        <Skeleton width="100%" height="2.5rem" />
      </div>

      <div class="my-8 flex items-center">
        <div class="flex-1 border-t border-gray-300"></div>
        <Skeleton width="30%" height="1rem" class="mx-4" />
        <div class="flex-1 border-t border-gray-300"></div>
      </div>

      <div class="flex flex-col gap-6">
        <Skeleton width="100%" height="2.5rem" />
        <Skeleton width="100%" height="2.5rem" />
      </div>

      <div class="mt-5">
        <Skeleton width="70%" height="0.875rem" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Skeleton from 'primevue/skeleton'
</script>
