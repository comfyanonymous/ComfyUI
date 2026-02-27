<script setup lang="ts">
import { remove } from 'es-toolkit'
import { computed, ref, toValue } from 'vue'
import type { MaybeRef } from 'vue'
import { useI18n } from 'vue-i18n'

import DraggableList from '@/components/common/DraggableList.vue'
import IoItem from '@/components/builder/IoItem.vue'
import PropertiesAccordionItem from '@/components/rightSidePanel/layout/PropertiesAccordionItem.vue'
import Button from '@/components/ui/button/Button.vue'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { INodeInputSlot } from '@/lib/litegraph/src/interfaces'
import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import { TitleMode } from '@/lib/litegraph/src/types/globalEnums'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { BaseWidget } from '@/lib/litegraph/src/widgets/BaseWidget'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCanvasInteractions } from '@/renderer/core/canvas/useCanvasInteractions'
import TransformPane from '@/renderer/core/layout/transform/TransformPane.vue'
import { app } from '@/scripts/app'
import { DOMWidgetImpl } from '@/scripts/domWidget'
import { useDialogService } from '@/services/dialogService'
import { useAppModeStore } from '@/stores/appModeStore'
import { cn } from '@/utils/tailwindUtil'

type BoundStyle = { top: string; left: string; width: string; height: string }

const appModeStore = useAppModeStore()
const canvasInteractions = useCanvasInteractions()
const canvasStore = useCanvasStore()
const settingStore = useSettingStore()
const workflowStore = useWorkflowStore()
const { t } = useI18n()
const canvas: LGraphCanvas = canvasStore.getCanvas()

const hoveringSelectable = ref(false)

workflowStore.activeWorkflow?.changeTracker?.reset()

const inputsWithState = computed(() =>
  appModeStore.selectedInputs.map(([nodeId, widgetName]) => {
    const node = app.rootGraph.getNodeById(nodeId)
    const widget = node?.widgets?.find((w) => w.name === widgetName)
    if (!node || !widget) return { nodeId, widgetName }

    const input = node.inputs.find((i) => i.widget?.name === widget.name)
    const rename = input && (() => renameWidget(widget, input))

    return {
      nodeId,
      widgetName,
      label: widget.label,
      subLabel: node.title,
      rename
    }
  })
)
const outputsWithState = computed<[NodeId, string][]>(() =>
  appModeStore.selectedOutputs.map((nodeId) => [
    nodeId,
    app.rootGraph.getNodeById(nodeId)?.title ?? String(nodeId)
  ])
)

async function renameWidget(widget: IBaseWidget, input: INodeInputSlot) {
  const newLabel = await useDialogService().prompt({
    title: t('g.rename'),
    message: t('g.enterNewNamePrompt'),
    defaultValue: widget.label,
    placeholder: widget.name
  })
  if (newLabel === null) return
  widget.label = newLabel || undefined
  input.label = newLabel || undefined
  widget.callback?.(widget.value)
  useCanvasStore().canvas?.setDirty(true)
}

function getHovered(
  e: MouseEvent
): undefined | [LGraphNode, undefined] | [LGraphNode, IBaseWidget] {
  const { graph } = canvas
  if (!canvas || !graph) return

  if (settingStore.get('Comfy.VueNodes.Enabled')) return undefined
  if (!e) return

  canvas.adjustMouseEvent(e)
  const node = graph.getNodeOnPos(e.canvasX, e.canvasY)
  if (!node) return

  const widget = node.getWidgetOnPos(e.canvasX, e.canvasY, false)

  if (widget || node.constructor.nodeData?.output_node) return [node, widget]
}

function getBounding(nodeId: NodeId, widgetName?: string) {
  if (settingStore.get('Comfy.VueNodes.Enabled')) return undefined
  const node = app.rootGraph.getNodeById(nodeId)
  if (!node) return

  const titleOffset =
    node.title_mode === TitleMode.NORMAL_TITLE ? LiteGraph.NODE_TITLE_HEIGHT : 0

  if (!widgetName)
    return {
      width: `${node.size[0]}px`,
      height: `${node.size[1] + titleOffset}px`,
      left: `${node.pos[0]}px`,
      top: `${node.pos[1] - titleOffset}px`
    }
  const widget = node.widgets?.find((w) => w.name === widgetName)
  if (!widget) return

  const margin = widget instanceof DOMWidgetImpl ? widget.margin : undefined
  const marginX = margin ?? BaseWidget.margin
  const height =
    (widget.computedHeight !== undefined
      ? widget.computedHeight - 4
      : LiteGraph.NODE_WIDGET_HEIGHT) - (margin ? 2 * margin - 4 : 0)
  return {
    width: `${node.size[0] - marginX * 2}px`,
    height: `${height}px`,
    left: `${node.pos[0] + marginX}px`,
    top: `${node.pos[1] + widget.y + (margin ?? 0)}px`
  }
}

function handleDown(e: MouseEvent) {
  const [node] = getHovered(e) ?? []
  if (!node || e.button > 0) canvasInteractions.forwardEventToCanvas(e)
}
function handleClick(e: MouseEvent) {
  const [node, widget] = getHovered(e) ?? []
  if (!node) return canvasInteractions.forwardEventToCanvas(e)

  if (!widget) {
    if (!node.constructor.nodeData?.output_node)
      return canvasInteractions.forwardEventToCanvas(e)
    const index = appModeStore.selectedOutputs.findIndex((id) => id === node.id)
    if (index === -1) appModeStore.selectedOutputs.push(node.id)
    else appModeStore.selectedOutputs.splice(index, 1)
    return
  }

  const index = appModeStore.selectedInputs.findIndex(
    ([nodeId, widgetName]) => node.id === nodeId && widget.name === widgetName
  )
  if (index === -1) appModeStore.selectedInputs.push([node.id, widget.name])
  else appModeStore.selectedInputs.splice(index, 1)
}

function nodeToDisplayTuple(
  n: LGraphNode
): [NodeId, MaybeRef<BoundStyle> | undefined, boolean] {
  return [
    n.id,
    getBounding(n.id),
    appModeStore.selectedOutputs.some((id) => n.id === id)
  ]
}

const renderedOutputs = computed(() => {
  void appModeStore.selectedOutputs.length
  return canvas
    .graph!.nodes.filter((n) => n.constructor.nodeData?.output_node)
    .map(nodeToDisplayTuple)
})
const renderedInputs = computed<[string, MaybeRef<BoundStyle> | undefined][]>(
  () =>
    appModeStore.selectedInputs.map(([nodeId, widgetName]) => [
      `${nodeId}: ${widgetName}`,
      getBounding(nodeId, widgetName)
    ])
)
</script>
<template>
  <div class="flex font-bold p-2 border-border-subtle border-b items-center">
    {{ t('linearMode.builder.title') }}
    <Button class="ml-auto" @click="appModeStore.exitBuilder">
      {{ t('linearMode.builder.exit') }}
    </Button>
  </div>
  <PropertiesAccordionItem
    :label="t('nodeHelpPage.inputs')"
    enable-empty-state
    :disabled="!appModeStore.selectedInputs.length"
    class="border-border-subtle border-b"
    :tooltip="`${t('linearMode.builder.inputsDesc')}\n${t('linearMode.builder.inputsExample')}`"
  >
    <template #label>
      <div class="flex gap-3">
        {{ t('nodeHelpPage.inputs') }}
        <i class="bg-muted-foreground icon-[lucide--circle-alert]" />
      </div>
    </template>
    <template #empty>
      <div
        class="w-full p-4 pt-2 text-muted-foreground"
        v-text="t('linearMode.builder.promptAddInputs')"
      />
    </template>
    <div
      class="w-full p-4 pt-2 text-muted-foreground"
      v-text="t('linearMode.builder.promptAddInputs')"
    />
    <DraggableList v-slot="{ dragClass }" v-model="appModeStore.selectedInputs">
      <IoItem
        v-for="{
          nodeId,
          widgetName,
          label,
          subLabel,
          rename
        } in inputsWithState"
        :key="`${nodeId}: ${widgetName}`"
        :class="cn(dragClass, 'bg-primary-background/30 p-2 my-2 rounded-lg')"
        :title="label ?? widgetName"
        :sub-title="subLabel"
        :rename
        :remove="
          () =>
            remove(
              appModeStore.selectedInputs,
              ([id, name]) => nodeId === id && widgetName === name
            )
        "
      />
    </DraggableList>
  </PropertiesAccordionItem>
  <PropertiesAccordionItem
    :label="t('nodeHelpPage.outputs')"
    enable-empty-state
    :disabled="!appModeStore.selectedOutputs.length"
    :tooltip="`${t('linearMode.builder.outputsDesc')}\n${t('linearMode.builder.outputsExample')}`"
  >
    <template #label>
      <div class="flex gap-3">
        {{ t('nodeHelpPage.outputs') }}
        <i class="bg-muted-foreground icon-[lucide--circle-alert]" />
      </div>
    </template>
    <template #empty>
      <div
        class="w-full p-4 pt-2 text-muted-foreground"
        v-text="t('linearMode.builder.promptAddOutputs')"
      />
    </template>
    <div
      class="w-full p-4 pt-2 text-muted-foreground"
      v-text="t('linearMode.builder.promptAddOutputs')"
    />
    <DraggableList
      v-slot="{ dragClass }"
      v-model="appModeStore.selectedOutputs"
    >
      <IoItem
        v-for="([key, title], index) in outputsWithState"
        :key
        :class="
          cn(
            dragClass,
            'bg-warning-background/40 p-2 my-2 rounded-lg',
            index === 0 && 'ring-warning-background ring-2'
          )
        "
        :title
        :sub-title="String(key)"
        :remove="() => remove(appModeStore.selectedOutputs, (k) => k === key)"
      />
    </DraggableList>
  </PropertiesAccordionItem>

  <Teleport to="body">
    <div
      :class="
        cn(
          'absolute w-full h-full pointer-events-auto',
          hoveringSelectable ? 'cursor-pointer' : 'cursor-grab'
        )
      "
      @pointerdown="handleDown"
      @pointermove="hoveringSelectable = !!getHovered($event)"
      @click="handleClick"
      @wheel="canvasInteractions.forwardEventToCanvas"
    >
      <TransformPane :canvas="canvasStore.getCanvas()">
        <div
          v-for="[key, style] in renderedInputs"
          :key
          :style="toValue(style)"
          class="fixed bg-primary-background/30 rounded-lg"
        />
        <div
          v-for="[key, style, isSelected] in renderedOutputs"
          :key
          :style="toValue(style)"
          :class="
            cn(
              'fixed ring-warning-background ring-5 rounded-2xl',
              !isSelected && 'ring-warning-background/50'
            )
          "
        >
          <div class="absolute top-0 right-0 size-8">
            <div
              v-if="isSelected"
              class="absolute -top-1/2 -right-1/2 size-full p-2 bg-warning-background rounded-lg"
            >
              <i class="icon-[lucide--check] bg-text-foreground size-full" />
            </div>
            <div
              v-else
              class="absolute -top-1/2 -right-1/2 size-full ring-warning-background/50 ring-4 ring-inset bg-component-node-background rounded-lg"
            />
          </div>
        </div>
      </TransformPane>
    </div>
  </Teleport>
</template>
