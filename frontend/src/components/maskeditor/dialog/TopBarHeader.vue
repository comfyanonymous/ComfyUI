<template>
  <div class="flex w-full items-center justify-between gap-3">
    <div class="flex items-center gap-3">
      <h3 class="m-0 text-lg font-semibold">{{ t('maskEditor.title') }}</h3>

      <div class="flex items-center gap-4">
        <button
          :class="iconButtonClass"
          :title="t('maskEditor.undo')"
          @click="onUndo"
        >
          <svg
            viewBox="0 0 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-current"
          >
            <path
              d="M8.77,12.18c-.25,0-.46-.2-.46-.46s.2-.46.46-.46c1.47,0,2.67-1.2,2.67-2.67,0-1.57-1.34-2.67-3.26-2.67h-3.98l1.43,1.43c.18.18.18.47,0,.64-.18.18-.47.18-.64,0l-2.21-2.21c-.18-.18-.18-.47,0-.64l2.21-2.21c.18-.18.47-.18.64,0,.18.18.18.47,0,.64l-1.43,1.43h3.98c2.45,0,4.17,1.47,4.17,3.58,0,1.97-1.61,3.58-3.58,3.58Z"
            />
          </svg>
        </button>

        <button
          :class="iconButtonClass"
          :title="t('maskEditor.redo')"
          @click="onRedo"
        >
          <svg
            viewBox="0 0 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-[var(--input-text)]"
          >
            <path
              class="cls-1"
              d="M6.23,12.18c-1.97,0-3.58-1.61-3.58-3.58,0-2.11,1.71-3.58,4.17-3.58h3.98l-1.43-1.43c-.18-.18-.18-.47,0-.64.18-.18.46-.18.64,0l2.21,2.21c.09.09.13.2.13.32s-.05.24-.13.32l-2.21,2.21c-.18.18-.47.18-.64,0-.18-.18-.18-.47,0-.64l1.43-1.43h-3.98c-1.92,0-3.26,1.1-3.26,2.67,0,1.47,1.2,2.67,2.67,2.67.25,0,.46.2.46.46s-.2.46-.46.46Z"
            />
          </svg>
        </button>

        <div class="h-5 border-l border-border" />

        <button
          :class="iconButtonClass"
          :title="t('maskEditor.rotateLeft')"
          @click="onRotateLeft"
        >
          <svg
            viewBox="-6 -7 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-[var(--input-text)]"
          >
            <path
              d="m2.25-2.625c.3452 0 .625.2798.625.625v5c0 .3452-.2798.625-.625.625h-5c-.3452 0-.625-.2798-.625-.625v-5c0-.3452.2798-.625.625-.625h5zm1.25.625v5c0 .6904-.5596 1.25-1.25 1.25h-5c-.6904 0-1.25-.5596-1.25-1.25v-5c0-.6904.5596-1.25 1.25-1.25h5c.6904 0 1.25.5596 1.25 1.25zm-.1673-2.3757-.4419.4419-1.5246-1.5246 1.5416-1.5417.442.4419-.7871.7872h.9373c1.3807 0 2.5 1.1193 2.5 2.5h-.625c0-1.0355-.8395-1.875-1.875-1.875h-.9375l.7702.7702z"
            />
          </svg>
        </button>

        <button
          :class="iconButtonClass"
          :title="t('maskEditor.rotateRight')"
          @click="onRotateRight"
        >
          <svg
            viewBox="-9 -7 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-[var(--input-text)]"
          >
            <g transform="scale(-1, 1)">
              <path
                d="m2.25-2.625c.3452 0 .625.2798.625.625v5c0 .3452-.2798.625-.625.625h-5c-.3452 0-.625-.2798-.625-.625v-5c0-.3452.2798-.625.625-.625h5zm1.25.625v5c0 .6904-.5596 1.25-1.25 1.25h-5c-.6904 0-1.25-.5596-1.25-1.25v-5c0-.6904.5596-1.25 1.25-1.25h5c.6904 0 1.25.5596 1.25 1.25zm-.1673-2.3757-.4419.4419-1.5246-1.5246 1.5416-1.5417.442.4419-.7871.7872h.9373c1.3807 0 2.5 1.1193 2.5 2.5h-.625c0-1.0355-.8395-1.875-1.875-1.875h-.9375l.7702.7702z"
              />
            </g>
          </svg>
        </button>

        <button
          :class="iconButtonClass"
          :title="t('maskEditor.mirrorHorizontal')"
          @click="onMirrorHorizontal"
        >
          <svg
            viewBox="0 0 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-[var(--input-text)]"
          >
            <path
              d="M7.5,1.5c-.28,0-.5.22-.5.5v11c0,.28.22.5.5.5s.5-.22.5-.5v-11c0-.28-.22-.5-.5-.5Z"
            />
            <path d="M3.5,4.5l-2,3,2,3v-6ZM11.5,4.5v6l2-3-2-3Z" />
          </svg>
        </button>

        <button
          :class="iconButtonClass"
          :title="t('maskEditor.mirrorVertical')"
          @click="onMirrorVertical"
        >
          <svg
            viewBox="0 0 15 15"
            class="h-6.25 w-6.25 pointer-events-none fill-[var(--input-text)]"
          >
            <path
              d="M2,7.5c0-.28.22-.5.5-.5h11c.28,0,.5.22.5.5s-.22.5-.5.5h-11c-.28,0-.5-.22-.5-.5Z"
            />
            <path d="M4.5,4.5l3-2,3,2h-6ZM4.5,10.5h6l-3,2-3-2Z" />
          </svg>
        </button>

        <div class="h-5 w-px bg-[var(--p-form-field-border-color)]" />

        <button :class="textButtonClass" @click="onInvert">
          {{ t('maskEditor.invert') }}
        </button>

        <button :class="textButtonClass" @click="onClear">
          {{ t('maskEditor.clear') }}
        </button>
      </div>
    </div>

    <div class="flex gap-3">
      <Button variant="primary" :disabled="!saveEnabled" @click="handleSave">
        <i class="pi pi-check" />
        {{ saveButtonText }}
      </Button>
      <Button variant="secondary" @click="handleCancel">
        <i class="pi pi-times" />
        {{ t('g.cancel') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useCanvasTools } from '@/composables/maskeditor/useCanvasTools'
import { useCanvasTransform } from '@/composables/maskeditor/useCanvasTransform'
import { useMaskEditorSaver } from '@/composables/maskeditor/useMaskEditorSaver'
import { useDialogStore } from '@/stores/dialogStore'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

const { t } = useI18n()
const store = useMaskEditorStore()
const dialogStore = useDialogStore()
const canvasTools = useCanvasTools()
const canvasTransform = useCanvasTransform()
const saver = useMaskEditorSaver()

const saveButtonText = ref(t('g.save'))
const saveEnabled = ref(true)

const iconButtonClass =
  'flex h-7.5 w-12.5 items-center justify-center rounded-[10px] border border-border-default pointer-events-auto transition-colors duration-100 bg-comfy-menu-bg hover:bg-secondary-background-hover'

const textButtonClass =
  'h-7.5 w-15 rounded-[10px] border border-border-default text-current font-sans pointer-events-auto transition-colors duration-100 bg-comfy-menu-bg hover:bg-secondary-background-hover'

const onUndo = () => {
  store.canvasHistory.undo()
}

const onRedo = () => {
  store.canvasHistory.redo()
}

const onRotateLeft = async () => {
  try {
    await canvasTransform.rotateCounterclockwise()
  } catch (error) {
    console.error('[TopBarHeader] Rotate left failed:', error)
  }
}

const onRotateRight = async () => {
  try {
    await canvasTransform.rotateClockwise()
  } catch (error) {
    console.error('[TopBarHeader] Rotate right failed:', error)
  }
}

const onMirrorHorizontal = async () => {
  try {
    await canvasTransform.mirrorHorizontal()
  } catch (error) {
    console.error('[TopBarHeader] Mirror horizontal failed:', error)
  }
}

const onMirrorVertical = async () => {
  try {
    await canvasTransform.mirrorVertical()
  } catch (error) {
    console.error('[TopBarHeader] Mirror vertical failed:', error)
  }
}

const onInvert = () => {
  canvasTools.invertMask()
}

const onClear = () => {
  canvasTools.clearMask()
  store.triggerClear()
}

const handleSave = async () => {
  saveButtonText.value = t('g.saving')
  saveEnabled.value = false

  try {
    store.brushVisible = false
    await saver.save()
    dialogStore.closeDialog()
  } catch (error) {
    console.error('[TopBarHeader] Save failed:', error)
    store.brushVisible = true
    saveButtonText.value = t('g.save')
    saveEnabled.value = true
  }
}

const handleCancel = () => {
  dialogStore.closeDialog({ key: 'global-mask-editor' })
}
</script>
