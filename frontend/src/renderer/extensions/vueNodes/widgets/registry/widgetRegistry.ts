/**
 * Widget type registry and component mapping for Vue-based widgets
 */
import { defineAsyncComponent } from 'vue'
import type { Component } from 'vue'

import type { SafeWidgetData } from '@/composables/graph/useGraphNodeManager'

const WidgetButton = defineAsyncComponent(
  () => import('../components/WidgetButton.vue')
)
const WidgetInputText = defineAsyncComponent(
  () => import('../components/WidgetInputText.vue')
)
const WidgetInputNumber = defineAsyncComponent(
  () => import('../components/WidgetInputNumber.vue')
)
const WidgetToggleSwitch = defineAsyncComponent(
  () => import('../components/WidgetToggleSwitch.vue')
)
const WidgetSelect = defineAsyncComponent(
  () => import('../components/WidgetSelect.vue')
)
const WidgetColorPicker = defineAsyncComponent(
  () => import('../components/WidgetColorPicker.vue')
)
const WidgetTextarea = defineAsyncComponent(
  () => import('../components/WidgetTextarea.vue')
)
const WidgetChart = defineAsyncComponent(
  () => import('../components/WidgetChart.vue')
)
const WidgetImageCompare = defineAsyncComponent(
  () => import('../components/WidgetImageCompare.vue')
)
const WidgetGalleria = defineAsyncComponent(
  () => import('../components/WidgetGalleria.vue')
)
const WidgetMarkdown = defineAsyncComponent(
  () => import('../components/WidgetMarkdown.vue')
)
const WidgetLegacy = defineAsyncComponent(
  () => import('../components/WidgetLegacy.vue')
)
const WidgetRecordAudio = defineAsyncComponent(
  () => import('../components/WidgetRecordAudio.vue')
)
const AudioPreviewPlayer = defineAsyncComponent(
  () => import('../components/audio/AudioPreviewPlayer.vue')
)
const Load3D = defineAsyncComponent(
  () => import('@/components/load3d/Load3D.vue')
)
const WidgetImageCrop = defineAsyncComponent(
  () => import('@/components/imagecrop/WidgetImageCrop.vue')
)
const WidgetBoundingBox = defineAsyncComponent(
  () => import('@/components/boundingbox/WidgetBoundingBox.vue')
)
const WidgetCurve = defineAsyncComponent(
  () => import('@/components/curve/WidgetCurve.vue')
)
const WidgetPainter = defineAsyncComponent(
  () => import('@/components/painter/WidgetPainter.vue')
)

export const FOR_TESTING = {
  WidgetButton,
  WidgetColorPicker,
  WidgetInputNumber,
  WidgetInputText,
  WidgetMarkdown,
  WidgetSelect,
  WidgetTextarea,
  WidgetToggleSwitch
} as const

interface WidgetDefinition {
  component: Component
  aliases: string[]
  essential: boolean
}

const coreWidgetDefinitions: Array<[string, WidgetDefinition]> = [
  [
    'button',
    { component: WidgetButton, aliases: ['BUTTON'], essential: false }
  ],
  [
    'string',
    {
      component: WidgetInputText,
      aliases: ['STRING', 'text'],
      essential: false
    }
  ],
  ['int', { component: WidgetInputNumber, aliases: ['INT'], essential: true }],
  [
    'float',
    {
      component: WidgetInputNumber,
      aliases: ['FLOAT', 'number', 'slider', 'gradientslider'],
      essential: true
    }
  ],
  [
    'boolean',
    {
      component: WidgetToggleSwitch,
      aliases: ['BOOLEAN', 'toggle'],
      essential: true
    }
  ],
  [
    'combo',
    { component: WidgetSelect, aliases: ['COMBO', 'asset'], essential: true }
  ],
  [
    'color',
    { component: WidgetColorPicker, aliases: ['COLOR'], essential: false }
  ],
  [
    'textarea',
    {
      component: WidgetTextarea,
      aliases: ['TEXTAREA', 'multiline', 'customtext'],
      essential: false
    }
  ],
  ['chart', { component: WidgetChart, aliases: ['CHART'], essential: false }],
  [
    'imagecompare',
    {
      component: WidgetImageCompare,
      aliases: ['IMAGECOMPARE'],
      essential: false
    }
  ],
  [
    'galleria',
    { component: WidgetGalleria, aliases: ['GALLERIA'], essential: false }
  ],
  [
    'markdown',
    {
      component: WidgetMarkdown,
      aliases: ['MARKDOWN', 'progressText'],
      essential: false
    }
  ],
  ['legacy', { component: WidgetLegacy, aliases: [], essential: true }],
  [
    'audiorecord',
    {
      component: WidgetRecordAudio,
      aliases: ['AUDIO_RECORD', 'AUDIORECORD'],
      essential: false
    }
  ],
  [
    'audioUI',
    {
      component: AudioPreviewPlayer,
      aliases: ['AUDIOUI', 'AUDIO_UI'],
      essential: false
    }
  ],
  ['load3D', { component: Load3D, aliases: ['LOAD_3D'], essential: false }],
  [
    'imagecrop',
    {
      component: WidgetImageCrop,
      aliases: ['IMAGECROP'],
      essential: false
    }
  ],
  [
    'boundingbox',
    {
      component: WidgetBoundingBox,
      aliases: ['BOUNDING_BOX'],
      essential: false
    }
  ],
  [
    'curve',
    {
      component: WidgetCurve,
      aliases: ['CURVE'],
      essential: false
    }
  ],
  [
    'painter',
    {
      component: WidgetPainter,
      aliases: ['PAINTER'],
      essential: false
    }
  ]
]

// Build lookup maps
const widgets = new Map<string, WidgetDefinition>()
const aliasMap = new Map<string, string>()

for (const [type, def] of coreWidgetDefinitions) {
  widgets.set(type, def)
  for (const alias of def.aliases) {
    aliasMap.set(alias, type)
  }
}

// Utility functions
const getCanonicalType = (type: string): string => aliasMap.get(type) || type

export const getComponent = (type: string): Component | null => {
  const canonicalType = getCanonicalType(type)
  return widgets.get(canonicalType)?.component || null
}

export const isEssential = (type: string): boolean => {
  const canonicalType = getCanonicalType(type)
  return widgets.get(canonicalType)?.essential || false
}

export const shouldRenderAsVue = (widget: Partial<SafeWidgetData>): boolean => {
  return !widget.options?.canvasOnly && !!widget.type
}

const EXPANDING_TYPES = [
  'textarea',
  'markdown',
  'load3D',
  'curve',
  'painter'
] as const

export function shouldExpand(type: string): boolean {
  const canonicalType = getCanonicalType(type)
  return EXPANDING_TYPES.includes(
    canonicalType as (typeof EXPANDING_TYPES)[number]
  )
}
