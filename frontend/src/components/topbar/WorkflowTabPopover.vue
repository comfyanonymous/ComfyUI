<template>
  <div
    ref="positionRef"
    class="absolute bottom-0 left-1/2 -translate-x-1/2"
  ></div>
  <Popover
    ref="popoverRef"
    append-to="body"
    :pt="{
      root: {
        class: 'workflow-popover-fade fit-content',
        'data-popover-id': id
      }
    }"
    @mouseenter="cancelHidePopover"
    @mouseleave="hidePopover"
  >
    <div class="workflow-preview-content">
      <div
        v-if="thumbnailUrl && !isActiveTab"
        class="workflow-preview-thumbnail relative"
      >
        <img
          :src="thumbnailUrl"
          class="block h-[200px] rounded-lg object-cover p-2"
          :style="{ width: `${POPOVER_WIDTH}px` }"
        />
      </div>
      <div class="workflow-preview-footer">
        <span class="workflow-preview-name">{{ workflowFilename }}</span>
      </div>
    </div>
  </Popover>
</template>

<script setup lang="ts">
import Popover from 'primevue/popover'
import { nextTick, ref, toRefs, useId } from 'vue'

const POPOVER_WIDTH = 250

interface Props {
  workflowFilename: string
  thumbnailUrl?: string
  isActiveTab: boolean
}

const props = defineProps<Props>()
const { thumbnailUrl, isActiveTab } = toRefs(props)
const popoverRef = ref<InstanceType<typeof Popover> | null>(null)
const positionRef = ref<HTMLElement | null>(null)
let hideTimeout: ReturnType<typeof setTimeout> | null = null
let showTimeout: ReturnType<typeof setTimeout> | null = null
const id = useId()

const showPopover = (event: Event) => {
  // Clear any existing timeouts
  if (hideTimeout) {
    clearTimeout(hideTimeout)
    hideTimeout = null
  }
  if (showTimeout) {
    clearTimeout(showTimeout)
    showTimeout = null
  }

  // Show popover after a short delay
  showTimeout = setTimeout(async () => {
    if (popoverRef.value && positionRef.value) {
      popoverRef.value.show(event, positionRef.value)
      await nextTick()
      // PrimeVue has a bug where when the tabs are scrolled, it positions the element incorrectly
      // Manually set the position to the middle of the tab and prevent it from going off the left/right edge
      const el = document.querySelector(
        `.workflow-popover-fade[data-popover-id="${id}"]`
      ) as HTMLElement
      if (el) {
        const middle = positionRef.value!.getBoundingClientRect().left
        const popoverWidth = el.getBoundingClientRect().width
        const halfWidth = popoverWidth / 2
        let pos = middle - halfWidth
        let shift = 0

        // Calculate shift when clamping is needed
        if (pos < 0) {
          shift = pos - 8 // Negative shift to move arrow left
          pos = 8
        } else if (pos + popoverWidth > window.innerWidth) {
          const newPos = window.innerWidth - popoverWidth - 16
          shift = pos - newPos // Positive shift to move arrow right
          pos = newPos
        }

        if (shift + halfWidth < 0) {
          shift = -halfWidth + 24
        }

        el.style.left = `${pos}px`
        el.style.setProperty('--shift', `${shift}px`)
      }
    }
  }, 200) // 200ms delay before showing
}

const cancelHidePopover = () => {
  // Temporarily disable this functionality until we need the popover to be interactive:
  /*
  if (hideTimeout) {
    clearTimeout(hideTimeout)
    hideTimeout = null
  }
  */
}

const hidePopover = () => {
  // Clear show timeout if mouse leaves before popover appears
  if (showTimeout) {
    clearTimeout(showTimeout)
    showTimeout = null
  }

  hideTimeout = setTimeout(() => {
    if (popoverRef.value) {
      popoverRef.value.hide()
    }
  }, 100) // Minimal delay to allow moving to popover
}

const togglePopover = (event: Event) => {
  if (popoverRef.value) {
    popoverRef.value.toggle(event)
  }
}

defineExpose({
  showPopover,
  hidePopover,
  togglePopover
})
</script>

<style scoped>
.workflow-preview-content {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: var(--radius-xl);
  max-width: var(--popover-width);
  background-color: var(--comfy-menu-bg);
  color: var(--fg-color);
}

.workflow-preview-thumbnail {
  position: relative;
  padding: calc(var(--spacing) * 2);
}

.workflow-preview-thumbnail img {
  box-shadow: var(--shadow-md);
  background-color: color-mix(in srgb, var(--comfy-menu-bg) 70%, black);
}

.dark-theme .workflow-preview-thumbnail img {
  box-shadow: var(--shadow-lg);
}

.workflow-preview-footer {
  padding-top: calc(var(--spacing) * 1);
  padding-right: calc(var(--spacing) * 3);
  padding-bottom: calc(var(--spacing) * 2);
  padding-left: calc(var(--spacing) * 3);
}

.workflow-preview-name {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--text-sm);
  line-height: var(--tw-leading, var(--text-sm--line-height));
  font-weight: var(--font-weight-medium);
  color: var(--fg-color);
}
</style>

<style>
.workflow-popover-fade {
  --p-popover-background: transparent;
  --p-popover-content-padding: 0;
  border-radius: var(--radius-xl);
  background-color: transparent;
  box-shadow: var(--shadow-lg);
  transition: opacity 0.15s ease-out !important;
}

.workflow-popover-fade.p-popover-flipped {
  transform: translateY(-100%);
}

.dark-theme .workflow-popover-fade {
  box-shadow: var(--shadow-2xl);
}

.workflow-popover-fade.p-popover::after,
.workflow-popover-fade.p-popover::before {
  --p-popover-border-color: var(--comfy-menu-secondary-bg);
  left: 50%;
  transform: translateX(calc(-50% + var(--shift)));
  margin-left: 0;
}
</style>
