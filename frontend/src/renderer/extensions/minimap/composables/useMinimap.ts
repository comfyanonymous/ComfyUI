import { useRafFn } from '@vueuse/core'
import { computed, nextTick, ref, shallowRef, watch } from 'vue'
import type { ShallowRef } from 'vue'

import type { LGraph } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

import type { MinimapCanvas, MinimapSettingsKey } from '../types'
import { useMinimapGraph } from './useMinimapGraph'
import { useMinimapInteraction } from './useMinimapInteraction'
import { useMinimapRenderer } from './useMinimapRenderer'
import { useMinimapSettings } from './useMinimapSettings'
import { useMinimapViewport } from './useMinimapViewport'

export function useMinimap({
  canvasRefMaybe,
  containerRefMaybe
}: {
  canvasRefMaybe?: Readonly<ShallowRef<HTMLCanvasElement | null>>
  containerRefMaybe?: Readonly<ShallowRef<HTMLDivElement | null>>
} = {}) {
  const canvasStore = useCanvasStore()
  const workflowStore = useWorkflowStore()
  const settingStore = useSettingStore()

  const minimapRef = ref<HTMLElement | null>(null)
  const canvasRef = canvasRefMaybe ?? shallowRef(null)
  const containerRef = containerRefMaybe ?? shallowRef(null)

  const visible = ref(true)
  const initialized = ref(false)

  const width = 250
  const height = 200

  const canvas = computed(() => canvasStore.canvas as MinimapCanvas | null)
  const graph = computed(() => {
    // If we're in a subgraph, use that; otherwise use the canvas graph
    const activeSubgraph = workflowStore.activeSubgraph
    return (activeSubgraph || canvas.value?.graph) as LGraph | null
  })

  // Settings
  const settings = useMinimapSettings()
  const {
    nodeColors,
    showLinks,
    showGroups,
    renderBypass,
    renderError,
    containerStyles,
    panelStyles
  } = settings

  const updateOption = async (key: MinimapSettingsKey, value: boolean) => {
    await settingStore.set(key, value)
    renderer.forceFullRedraw()
    renderer.updateMinimap(viewport.updateBounds, viewport.updateViewport)
  }

  // Viewport management
  const viewport = useMinimapViewport(canvas, graph, width, height)

  // Interaction handling
  const interaction = useMinimapInteraction(
    containerRef,
    viewport.bounds,
    viewport.scale,
    width,
    height,
    viewport.centerViewOn,
    canvas
  )

  // Graph event management
  const graphManager = useMinimapGraph(graph, () => {
    renderer.forceFullRedraw()
    renderer.updateMinimap(viewport.updateBounds, viewport.updateViewport)
  })

  // Rendering
  const renderer = useMinimapRenderer(
    canvasRef,
    graph,
    viewport.bounds,
    viewport.scale,
    graphManager.updateFlags,
    settings,
    width,
    height
  )

  // RAF loop for continuous updates
  const { pause: pauseChangeDetection, resume: resumeChangeDetection } =
    useRafFn(
      async () => {
        if (visible.value) {
          const hasChanges = await graphManager.checkForChanges()
          if (hasChanges) {
            renderer.updateMinimap(
              viewport.updateBounds,
              viewport.updateViewport
            )
          }
        }
      },
      { immediate: false }
    )

  const init = async () => {
    if (initialized.value) return

    visible.value = settingStore.get('Comfy.Minimap.Visible')

    if (canvas.value && graph.value) {
      graphManager.init()

      if (containerRef.value) {
        interaction.updateContainerRect()
      }
      viewport.updateCanvasDimensions()

      window.addEventListener('resize', interaction.updateContainerRect)
      window.addEventListener('scroll', interaction.updateContainerRect)
      window.addEventListener('resize', viewport.updateCanvasDimensions)

      renderer.forceFullRedraw()
      renderer.updateMinimap(viewport.updateBounds, viewport.updateViewport)
      viewport.updateViewport()

      if (visible.value) {
        resumeChangeDetection()
        viewport.startViewportSync()
      }
      initialized.value = true
    }
  }

  const destroy = () => {
    pauseChangeDetection()
    viewport.stopViewportSync()
    graphManager.destroy()

    window.removeEventListener('resize', interaction.updateContainerRect)
    window.removeEventListener('scroll', interaction.updateContainerRect)
    window.removeEventListener('resize', viewport.updateCanvasDimensions)

    initialized.value = false
  }

  watch(
    canvas,
    async (newCanvas, oldCanvas) => {
      if (oldCanvas) {
        graphManager.cleanupEventListeners()
        pauseChangeDetection()
        viewport.stopViewportSync()
        graphManager.destroy()
        window.removeEventListener('resize', interaction.updateContainerRect)
        window.removeEventListener('scroll', interaction.updateContainerRect)
        window.removeEventListener('resize', viewport.updateCanvasDimensions)
      }
      if (newCanvas && !initialized.value) {
        await init()
      }
    },
    { immediate: true, flush: 'post' }
  )

  // Watch for graph changes (e.g., when navigating to/from subgraphs)
  watch(graph, (newGraph, oldGraph) => {
    if (newGraph && newGraph !== oldGraph) {
      graphManager.cleanupEventListeners(oldGraph || undefined)
      graphManager.setupEventListeners()
      renderer.forceFullRedraw()
      renderer.updateMinimap(viewport.updateBounds, viewport.updateViewport)
    }
  })

  watch(visible, async (isVisible) => {
    if (isVisible) {
      if (containerRef.value) {
        interaction.updateContainerRect()
      }
      viewport.updateCanvasDimensions()

      renderer.forceFullRedraw()

      await nextTick()
      await nextTick()

      renderer.updateMinimap(viewport.updateBounds, viewport.updateViewport)
      viewport.updateViewport()
      resumeChangeDetection()
      viewport.startViewportSync()
    } else {
      pauseChangeDetection()
      viewport.stopViewportSync()
    }
  })

  const toggle = async () => {
    visible.value = !visible.value
    await settingStore.set('Comfy.Minimap.Visible', visible.value)
  }

  const setMinimapRef = (ref: HTMLElement | null) => {
    minimapRef.value = ref
  }

  // Dynamic viewport styles based on actual viewport transform
  const viewportStyles = computed(() => {
    const transform = viewport.viewportTransform.value
    return {
      transform: `translate(${transform.x}px, ${transform.y}px)`,
      width: `${transform.width}px`,
      height: `${transform.height}px`,
      border: `2px solid ${settings.isLightTheme.value ? '#E0E0E0' : '#FFF'}`,
      backgroundColor: `rgba(255, 255, 255, 0.2)`,
      willChange: 'transform',
      backfaceVisibility: 'hidden' as const,
      perspective: '1000px',
      pointerEvents: 'none' as const
    }
  })

  return {
    visible: computed(() => visible.value),
    initialized: computed(() => initialized.value),

    containerStyles,
    viewportStyles,
    panelStyles,
    width,
    height,

    nodeColors,
    showLinks,
    showGroups,
    renderBypass,
    renderError,

    init,
    destroy,
    toggle,
    renderMinimap: renderer.renderMinimap,
    handlePointerDown: interaction.handlePointerDown,
    handlePointerMove: interaction.handlePointerMove,
    handlePointerUp: interaction.handlePointerUp,
    handlePointerCancel: interaction.handlePointerCancel,
    handleWheel: interaction.handleWheel,
    setMinimapRef,
    updateOption
  }
}
