import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/litegraph'

export interface ImageRef {
  filename: string
  subfolder?: string
  type?: string
}

export interface ImageLayer {
  image: HTMLImageElement
  url: string
}

interface EditorInputData {
  baseLayer: ImageLayer
  maskLayer: ImageLayer
  paintLayer?: ImageLayer
  sourceRef: ImageRef
  nodeId: NodeId
}

export interface EditorOutputLayer {
  canvas: HTMLCanvasElement
  blob: Blob
  ref: ImageRef
}

export interface EditorOutputData {
  maskedImage: EditorOutputLayer
  paintLayer: EditorOutputLayer
  paintedImage: EditorOutputLayer
  paintedMaskedImage: EditorOutputLayer
}

export const useMaskEditorDataStore = defineStore('maskEditorData', () => {
  const inputData = ref<EditorInputData | null>(null)
  const outputData = ref<EditorOutputData | null>(null)
  const sourceNode = ref<LGraphNode | null>(null)

  const isLoading = ref(false)
  const loadError = ref<string | null>(null)

  const hasValidInput = computed(() => inputData.value !== null)

  const hasValidOutput = computed(() => outputData.value !== null)

  const isReady = computed(() => hasValidInput.value && !isLoading.value)

  const reset = () => {
    inputData.value = null
    outputData.value = null
    sourceNode.value = null
    isLoading.value = false
    loadError.value = null
  }

  const setLoading = (loading: boolean, error?: string) => {
    isLoading.value = loading
    if (error) {
      loadError.value = error
    }
  }

  return {
    inputData,
    outputData,
    sourceNode,
    isLoading,
    loadError,

    hasValidInput,
    hasValidOutput,
    isReady,

    reset,
    setLoading
  }
})
