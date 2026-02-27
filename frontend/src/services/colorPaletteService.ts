import { toRaw } from 'vue'
import { z } from 'zod'
import { fromZodError } from 'zod-validation-error'

import { downloadBlob } from '@/base/common/downloadUtil'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { LGraphCanvas, LiteGraph } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { paletteSchema, comfyBaseSchema } from '@/schemas/colorPaletteSchema'
import type { Colors, Palette } from '@/schemas/colorPaletteSchema'
import { app } from '@/scripts/app'
import { uploadFile } from '@/scripts/utils'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'

const THEME_PROPERTY_MAP = {
  NODE_BOX_OUTLINE_COLOR: 'component-node-border',
  NODE_DEFAULT_BGCOLOR: 'component-node-background',
  NODE_DEFAULT_BOXCOLOR: 'node-component-header-icon',
  NODE_DEFAULT_COLOR: 'node-component-header-surface',
  NODE_TITLE_COLOR: 'node-component-header',
  WIDGET_BGCOLOR: 'component-node-widget-background',
  WIDGET_TEXT_COLOR: 'component-node-foreground'
} as const satisfies Partial<Record<keyof Colors['litegraph_base'], string>>

export const useColorPaletteService = () => {
  const colorPaletteStore = useColorPaletteStore()
  const settingStore = useSettingStore()
  const nodeDefStore = useNodeDefStore()
  const { wrapWithErrorHandling, wrapWithErrorHandlingAsync } =
    useErrorHandling()

  /**
   * Validates the palette against the zod schema.
   *
   * @param data - The palette to validate.
   * @returns The validated palette.
   */
  const validateColorPalette = (data: unknown): Palette => {
    const result = paletteSchema.safeParse(data)
    if (result.success) return result.data

    const error = fromZodError(result.error)
    throw new Error(`Invalid color palette against zod schema:\n${error}`)
  }

  const persistCustomColorPalettes = async () => {
    await settingStore.set(
      'Comfy.CustomColorPalettes',
      colorPaletteStore.customPalettes
    )
  }

  /**
   * Deletes a custom color palette.
   *
   * @param colorPaletteId - The ID of the color palette to delete.
   */
  const deleteCustomColorPalette = async (colorPaletteId: string) => {
    colorPaletteStore.deleteCustomPalette(colorPaletteId)
    await persistCustomColorPalettes()
  }

  /**
   * Adds a custom color palette.
   *
   * @param colorPalette - The palette to add.
   */
  const addCustomColorPalette = async (colorPalette: Palette) => {
    validateColorPalette(colorPalette)
    colorPaletteStore.addCustomPalette(colorPalette)
    await persistCustomColorPalettes()
  }

  /**
   * Sets the colors of node slots and links.
   *
   * @param linkColorPalette - The palette to set.
   */
  const loadLinkColorPalette = (linkColorPalette: Colors['node_slot']) => {
    const types = Object.fromEntries(
      Array.from(nodeDefStore.nodeDataTypes).map((type) => [type, ''])
    )
    Object.assign(
      app.canvas.default_connection_color_byType,
      types,
      linkColorPalette
    )
    Object.assign(LGraphCanvas.link_type_colors, types, linkColorPalette)
  }

  function validThemeProp(
    propertyMaybe: unknown
  ): propertyMaybe is keyof typeof THEME_PROPERTY_MAP {
    return (
      (propertyMaybe as keyof typeof THEME_PROPERTY_MAP) in THEME_PROPERTY_MAP
    )
  }

  function loadLinkColorPaletteForVueNodes(
    linkColorPalette: Colors['node_slot']
  ) {
    if (!linkColorPalette) return
    const rootStyle = document.body?.style
    if (!rootStyle) return

    for (const dataType of nodeDefStore.nodeDataTypes) {
      const cssVar = `color-datatype-${dataType}`

      const valueMaybe = linkColorPalette[dataType as keyof Colors['node_slot']]
      if (valueMaybe) {
        rootStyle.setProperty(`--${cssVar}`, valueMaybe)
      } else {
        rootStyle.removeProperty(`--${cssVar}`)
      }
    }
  }

  function loadLitegraphForVueNodes(
    palette: Colors['litegraph_base'],
    colorPaletteId: string
  ) {
    if (!palette) return
    const rootStyle = document.body?.style
    if (!rootStyle) return

    for (const themeVar of Object.keys(THEME_PROPERTY_MAP)) {
      if (!validThemeProp(themeVar)) {
        continue
      }
      const cssVar = THEME_PROPERTY_MAP[themeVar]
      if (colorPaletteId === 'dark' || colorPaletteId === 'light') {
        rootStyle.removeProperty(`--${cssVar}`)
        continue
      }
      const valueMaybe = palette[themeVar]
      if (valueMaybe) {
        rootStyle.setProperty(`--${cssVar}`, valueMaybe)
      } else {
        rootStyle.removeProperty(`--${cssVar}`)
      }
    }
  }

  /**
   * Loads the LiteGraph color palette.
   *
   * @param liteGraphColorPalette - The palette to set.
   */
  const loadLiteGraphColorPalette = (palette: Colors['litegraph_base']) => {
    // Sets the colors of the LiteGraph objects
    app.canvas.node_title_color = palette.NODE_TITLE_COLOR
    app.canvas.default_link_color = palette.LINK_COLOR
    const backgroundImage = settingStore.get('Comfy.Canvas.BackgroundImage')
    if (backgroundImage) {
      app.canvas.clear_background_color = 'transparent'
    } else {
      app.canvas.background_image = palette.BACKGROUND_IMAGE
      app.canvas.clear_background_color = palette.CLEAR_BACKGROUND_COLOR
    }
    app.canvas._pattern = undefined

    if (typeof palette.NODE_DEFAULT_SHAPE === 'string')
      console.warn(
        `litegraph_base.NODE_DEFAULT_SHAPE only accepts [${[
          LiteGraph.BOX_SHAPE,
          LiteGraph.ROUND_SHAPE,
          LiteGraph.CARD_SHAPE
        ].join(', ')}] but got ${palette.NODE_DEFAULT_SHAPE}`
      )

    const default_shape =
      typeof palette.NODE_DEFAULT_SHAPE === 'string'
        ? LiteGraph.ROUND_SHAPE
        : palette.NODE_DEFAULT_SHAPE
    const sanitizedPalette: Partial<typeof LiteGraph> = {
      ...palette,
      NODE_DEFAULT_SHAPE: default_shape
    }

    Object.assign(LiteGraph, sanitizedPalette)
  }

  /**
   * Gets optional keys from a Zod schema object.
   *
   * @param schema - The Zod schema object to analyze.
   * @returns Array of optional key names.
   */
  const getOptionalKeys = (schema: z.ZodObject<z.ZodRawShape>) => {
    const optionalKeys: string[] = []
    const shape = schema.shape

    for (const [key, value] of Object.entries(shape)) {
      if (value instanceof z.ZodOptional || value instanceof z.ZodDefault) {
        optionalKeys.push(key)
      }
    }

    return optionalKeys
  }
  const optionalComfyBaseKeys = getOptionalKeys(comfyBaseSchema)

  /**
   * Loads the Comfy color palette.
   *
   * @param comfyColorPalette - The palette to set.
   */
  const loadComfyColorPalette = (comfyColorPalette: Colors['comfy_base']) => {
    if (!comfyColorPalette) return
    const rootStyle = document.documentElement.style
    for (const [key, value] of Object.entries(comfyColorPalette)) {
      rootStyle.setProperty('--' + key, value)
    }

    for (const optionalKey of optionalComfyBaseKeys) {
      if (!(optionalKey in comfyColorPalette)) {
        rootStyle.setProperty(
          '--' + optionalKey,
          `var(--palette-${optionalKey})`
        )
      }
    }

    const backgroundImage = settingStore.get('Comfy.Canvas.BackgroundImage')
    if (backgroundImage) {
      rootStyle.setProperty('--bg-img', `url('${backgroundImage}')`)
    } else {
      rootStyle.removeProperty('--bg-img')
    }
  }

  /**
   * Loads the color palette.
   *
   * @param colorPaletteId - The ID of the color palette to load.
   */
  const loadColorPalette = async (colorPaletteId: string) => {
    const colorPalette = colorPaletteStore.palettesLookup[colorPaletteId]
    if (!colorPalette) {
      throw new Error(`Color palette ${colorPaletteId} not found`)
    }

    const completedPalette = colorPaletteStore.completePalette(colorPalette)
    loadLinkColorPalette(completedPalette.colors.node_slot)
    loadLiteGraphColorPalette(completedPalette.colors.litegraph_base)
    loadLitegraphForVueNodes(
      completedPalette.colors.litegraph_base,
      colorPaletteId
    )
    loadLinkColorPaletteForVueNodes(completedPalette.colors.node_slot)
    loadComfyColorPalette(completedPalette.colors.comfy_base)
    app.canvas.setDirty(true, true)

    colorPaletteStore.activePaletteId = colorPaletteId
  }

  /**
   * Exports a color palette.
   *
   * @param colorPaletteId - The ID of the color palette to export.
   */
  const exportColorPalette = (colorPaletteId: string) => {
    const colorPalette = colorPaletteStore.palettesLookup[colorPaletteId]
    if (!colorPalette) {
      throw new Error(`Color palette ${colorPaletteId} not found`)
    }
    downloadBlob(
      colorPalette.id + '.json',
      new Blob([JSON.stringify(toRaw(colorPalette), null, 2)], {
        type: 'application/json'
      })
    )
  }

  /**
   * Imports a color palette.
   *
   * @returns The imported palette.
   */
  const importColorPalette = async () => {
    const file = await uploadFile('application/json')
    const text = await file.text()
    const palette = JSON.parse(text)
    await addCustomColorPalette(palette)
    return palette
  }

  return {
    getActiveColorPalette: () => colorPaletteStore.completedActivePalette,
    addCustomColorPalette: wrapWithErrorHandlingAsync(addCustomColorPalette),
    deleteCustomColorPalette: wrapWithErrorHandlingAsync(
      deleteCustomColorPalette
    ),
    loadColorPalette: wrapWithErrorHandlingAsync(loadColorPalette),
    exportColorPalette: wrapWithErrorHandling(exportColorPalette),
    importColorPalette: wrapWithErrorHandlingAsync(importColorPalette)
  }
}
