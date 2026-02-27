<template>
  <div>
    <Button
      v-tooltip="{ value: $t('g.moreWorkflows'), showDelay: 300 }"
      class="rounded-none h-full w-auto aspect-square"
      variant="muted-textonly"
      size="icon"
      :aria-label="$t('g.moreWorkflows')"
      @click="menu?.toggle($event)"
    >
      <i class="pi pi-ellipsis-h" />
    </Button>
    <Menu
      ref="menu"
      :model="menuItems"
      :popup="true"
      class="max-h-[40vh] overflow-auto"
    />
  </div>
</template>

<script setup lang="ts">
import Menu from 'primevue/menu'
import { computed, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'

const props = defineProps<{
  workflows: ComfyWorkflow[]
  activeWorkflow: ComfyWorkflow | null
}>()

const menu = ref<InstanceType<typeof Menu> | null>(null)
const workflowService = useWorkflowService()

const menuItems = computed(() =>
  props.workflows.map((workflow: ComfyWorkflow) => ({
    label: workflow.filename,
    icon:
      props.activeWorkflow?.key === workflow.key ? 'pi pi-check' : undefined,
    command: () => {
      void workflowService.openWorkflow(workflow)
    }
  }))
)
</script>
