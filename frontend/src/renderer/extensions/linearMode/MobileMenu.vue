<script setup lang="ts">
import {
  CollapsibleRoot,
  CollapsibleTrigger,
  CollapsibleContent
} from 'reka-ui'
import { useI18n } from 'vue-i18n'

import WorkflowsSidebarTab from '@/components/sidebar/tabs/WorkflowsSidebarTab.vue'
import Button from '@/components/ui/button/Button.vue'
import Popover from '@/components/ui/Popover.vue'
import { useWorkflowTemplateSelectorDialog } from '@/composables/useWorkflowTemplateSelectorDialog'
import { useCommandStore } from '@/stores/commandStore'

const { t } = useI18n()
</script>
<template>
  <CollapsibleRoot class="flex flex-col">
    <CollapsibleTrigger as-child>
      <Button variant="secondary" class="size-10 self-end m-4 mb-2">
        <i class="icon-[lucide--menu] size-8" />
      </Button>
    </CollapsibleTrigger>
    <CollapsibleContent class="flex gap-2 flex-col">
      <div class="w-full border-b-2 border-border-subtle" />
      <Popover>
        <template #button>
          <Button variant="secondary" size="lg" class="w-full">
            <i class="icon-[comfy--workflow]" />
            {{ t('Workflows') }}
          </Button>
        </template>
        <WorkflowsSidebarTab class="h-300 w-[80vw]" />
      </Popover>
      <Button
        variant="secondary"
        size="lg"
        class="w-full"
        @click="useWorkflowTemplateSelectorDialog().show('menu')"
      >
        <i class="icon-[comfy--template]" />
        {{ t('sideToolbar.templates') }}
      </Button>
      <Button
        variant="secondary"
        size="lg"
        class="w-full"
        @click="
          useCommandStore().execute('Comfy.ToggleLinear', {
            metadata: { source: 'button' }
          })
        "
      >
        <i class="icon-[lucide--log-out]" />
        {{ t('linearMode.graphMode') }}
      </Button>
      <div class="w-full border-b-2 border-border-subtle" />
    </CollapsibleContent>
  </CollapsibleRoot>
</template>
