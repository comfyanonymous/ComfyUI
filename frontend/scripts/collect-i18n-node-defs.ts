import * as fs from 'fs'

import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'

import { comfyPageFixture as test } from '../browser_tests/fixtures/ComfyPage'
import { normalizeI18nKey } from '../packages/shared-frontend-utils/src/formatUtil'
import type { ComfyNodeDefImpl } from '../src/stores/nodeDefStore'

const localePath = './src/locales/en/main.json'
const nodeDefsPath = './src/locales/en/nodeDefs.json'

interface WidgetInfo {
  name?: string
  label?: string
}

interface WidgetLabels {
  [key: string]: Record<string, { name: string }>
}

test('collect-i18n-node-defs', async ({ comfyPage }) => {
  // Mock view route
  await comfyPage.page.route('**/view**', async (route) => {
    await route.fulfill({
      body: JSON.stringify({})
    })
  })

  // Note: Don't mock the object_info API endpoint - let it hit the actual backend

  const nodeDefs: ComfyNodeDefImpl[] = await comfyPage.page.evaluate(
    async () => {
      // @ts-expect-error - app is dynamically added to window
      const api = window['app'].api
      const rawNodeDefs = await api.getNodeDefs()
      const { ComfyNodeDefImpl } = await import('../src/stores/nodeDefStore')

      return (
        Object.values(rawNodeDefs)
          // Ignore DevTools nodes (used for internal testing)
          .filter((def: ComfyNodeDef) => !def.name.startsWith('DevTools'))
          .map((def: ComfyNodeDef) => new ComfyNodeDefImpl(def))
      )
    }
  )

  console.log(`Collected ${nodeDefs.length} node definitions`)

  const allDataTypesLocale = Object.fromEntries(
    nodeDefs
      .flatMap((nodeDef) => {
        const inputDataTypes = Object.values(nodeDef.inputs).map(
          (inputSpec) => inputSpec.type
        )
        const outputDataTypes = nodeDef.outputs.map(
          (outputSpec) => outputSpec.type
        )
        const allDataTypes = [...inputDataTypes, ...outputDataTypes].flatMap(
          (type: string) => type.split(',')
        )
        return allDataTypes.map((dataType) => [
          normalizeI18nKey(dataType),
          dataType
        ])
      })
      .sort((a, b) => a[0].localeCompare(b[0]))
  )

  async function extractWidgetLabels() {
    const nodeLabels: WidgetLabels = {}

    for (const nodeDef of nodeDefs) {
      const inputNames = Object.values(nodeDef.inputs).map(
        (input) => input.name
      )

      if (!inputNames.length) continue

      try {
        const widgetsMappings = await comfyPage.page.evaluate(
          (args) => {
            const [nodeName, displayName, inputNames] = args
            // @ts-expect-error - LiteGraph is dynamically added to window
            const node = window['LiteGraph'].createNode(nodeName, displayName)
            if (!node.widgets?.length) return {}
            return Object.fromEntries(
              node.widgets
                .filter(
                  (w: WidgetInfo) => w?.name && !inputNames.includes(w.name)
                )
                .map((w: WidgetInfo) => [w.name, w.label])
            )
          },
          [nodeDef.name, nodeDef.display_name, inputNames]
        )

        // Format runtime widgets
        const runtimeWidgets = Object.fromEntries(
          Object.entries(widgetsMappings)
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([key, value]) => [normalizeI18nKey(key), { name: value }])
        )

        if (Object.keys(runtimeWidgets).length > 0) {
          nodeLabels[nodeDef.name] = runtimeWidgets
        }
      } catch (error) {
        console.error(
          `Failed to extract widgets from ${nodeDef.name}: ${error}`
        )
      } finally {
        await comfyPage.nextFrame()
      }
    }

    return nodeLabels
  }

  const nodeDefLabels = await extractWidgetLabels()

  function extractInputs(nodeDef: ComfyNodeDefImpl) {
    const inputs = Object.fromEntries(
      Object.values(nodeDef.inputs).flatMap((input) => {
        const name = input.name
        const tooltip = input.tooltip

        if (name === undefined && tooltip === undefined) {
          return []
        }

        return [
          [
            normalizeI18nKey(input.name),
            {
              name,
              tooltip
            }
          ]
        ]
      })
    )
    return Object.keys(inputs).length > 0 ? inputs : undefined
  }

  function extractOutputs(nodeDef: ComfyNodeDefImpl) {
    const outputs = Object.fromEntries(
      nodeDef.outputs.flatMap((output, i) => {
        // Ignore data types if they are already translated in allDataTypesLocale.
        const name = output.name in allDataTypesLocale ? undefined : output.name
        const tooltip = output.tooltip

        if (name === undefined && tooltip === undefined) {
          return []
        }

        return [
          [
            i.toString(),
            {
              name,
              tooltip
            }
          ]
        ]
      })
    )
    return Object.keys(outputs).length > 0 ? outputs : undefined
  }

  const allNodeDefsLocale = Object.fromEntries(
    nodeDefs
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((nodeDef) => {
        const inputs = {
          ...extractInputs(nodeDef),
          ...(nodeDefLabels[nodeDef.name] ?? {})
        }

        return [
          normalizeI18nKey(nodeDef.name),
          {
            display_name: nodeDef.display_name ?? nodeDef.name,
            description: nodeDef.description || undefined,
            inputs: Object.keys(inputs).length > 0 ? inputs : undefined,
            outputs: extractOutputs(nodeDef)
          }
        ]
      })
  )

  const allNodeCategoriesLocale = Object.fromEntries(
    nodeDefs.flatMap((nodeDef) =>
      nodeDef.category
        .split('/')
        .map((category) => [normalizeI18nKey(category), category])
    )
  )

  const locale = JSON.parse(fs.readFileSync(localePath, 'utf-8'))
  fs.writeFileSync(
    localePath,
    JSON.stringify(
      {
        ...locale,
        dataTypes: allDataTypesLocale,
        nodeCategories: allNodeCategoriesLocale
      },
      null,
      2
    )
  )

  fs.writeFileSync(nodeDefsPath, JSON.stringify(allNodeDefsLocale, null, 2))
})
