import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { t } from '@/i18n'
import { SubgraphNode } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import type { LoadedComfyWorkflow } from '@/platform/workflow/management/stores/comfyWorkflow'
import { ComfyWorkflow } from '@/platform/workflow/management/stores/comfyWorkflow'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type {
  ComfyNode,
  ComfyWorkflowJSON,
  NodeId
} from '@/platform/workflow/validation/schemas/workflowSchema'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { NodeError } from '@/schemas/apiSchema'
import type {
  ComfyNodeDef as ComfyNodeDefV1,
  InputSpec
} from '@/schemas/nodeDefSchema'
import { isCloud, isDesktop } from '@/platform/distribution/types'
import { TemplateIncludeOnDistributionEnum } from '@/platform/workflow/templates/types/template'
import { api } from '@/scripts/api'
import type { GlobalSubgraphData } from '@/scripts/api'
import { useDialogService } from '@/services/dialogService'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'
import { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { UserFile } from '@/stores/userFileStore'

async function confirmOverwrite(name: string): Promise<boolean | null> {
  return await useDialogService().confirm({
    title: t('subgraphStore.overwriteBlueprintTitle'),
    type: 'overwriteBlueprint',
    message: t('subgraphStore.overwriteBlueprint'),
    itemList: [name]
  })
}

export const useSubgraphStore = defineStore('subgraph', () => {
  class SubgraphBlueprint extends ComfyWorkflow {
    static override readonly basePath = 'subgraphs/'
    override readonly tintCanvasBg = '#22227740'

    hasPromptedSave: boolean = false

    constructor(
      options: { path: string; modified: number; size: number },
      confirmFirstSave: boolean = false
    ) {
      super(options)
      this.hasPromptedSave = !confirmFirstSave
    }

    validateSubgraph() {
      if (!this.activeState?.definitions)
        throw new Error(
          'The root graph of a subgraph blueprint must consist of only a single subgraph node'
        )
      const { subgraphs } = this.activeState.definitions
      const { nodes } = this.activeState
      //Instanceof doesn't function as nodes are serialized
      function isSubgraphNode(node: ComfyNode) {
        return node && subgraphs.some((s) => s.id === node.type)
      }
      if (nodes.length == 1 && isSubgraphNode(nodes[0])) return
      const errors: Record<NodeId, NodeError> = {}
      //mark errors for all but first subgraph node
      let firstSubgraphFound = false
      for (let i = 0; i < nodes.length; i++) {
        if (!firstSubgraphFound && isSubgraphNode(nodes[i])) {
          firstSubgraphFound = true
          continue
        }
        errors[nodes[i].id] = {
          errors: [],
          class_type: nodes[i].type,
          dependent_outputs: []
        }
      }
      useExecutionErrorStore().lastNodeErrors = errors
      useCanvasStore().getCanvas().draw(true, true)
      throw new Error(
        'The root graph of a subgraph blueprint must consist of only a single subgraph node'
      )
    }

    override async save(): Promise<UserFile> {
      this.validateSubgraph()
      if (
        !this.hasPromptedSave &&
        useSettingStore().get('Comfy.Workflow.WarnBlueprintOverwrite')
      ) {
        if (!(await confirmOverwrite(this.filename))) return this
        this.hasPromptedSave = true
      }
      // Extract metadata from subgraph.extra to workflow.extra before saving
      this.extractMetadataToWorkflowExtra()
      const ret = await super.save()
      // Force reload to update initialState with saved metadata
      registerNodeDef(await this.load({ force: true }), {
        category: 'Subgraph Blueprints/User'
      })
      return ret
    }

    /**
     * Moves all properties (except workflowRendererVersion) from subgraph.extra
     * to workflow.extra, then removes from subgraph.extra to avoid duplication.
     */
    private extractMetadataToWorkflowExtra(): void {
      if (!this.activeState) return
      const subgraph = this.activeState.definitions?.subgraphs?.[0]
      if (!subgraph?.extra) return

      const sgExtra = subgraph.extra as Record<string, unknown>
      const workflowExtra = (this.activeState.extra ??= {}) as Record<
        string,
        unknown
      >

      for (const key of Object.keys(sgExtra)) {
        if (key === 'workflowRendererVersion') continue
        workflowExtra[key] = sgExtra[key]
        delete sgExtra[key]
      }
    }

    override async saveAs(path: string) {
      this.validateSubgraph()
      this.hasPromptedSave = true
      // Extract metadata from subgraph.extra to workflow.extra before saving
      this.extractMetadataToWorkflowExtra()
      const ret = await super.saveAs(path)
      // Force reload to update initialState with saved metadata
      registerNodeDef(await this.load({ force: true }), {
        category: 'Subgraph Blueprints/User'
      })
      return ret
    }
    override async load({ force = false }: { force?: boolean } = {}): Promise<
      this & LoadedComfyWorkflow
    > {
      if (!force && this.isLoaded) return await super.load({ force })
      const loaded = await super.load({ force })
      const st = loaded.activeState
      const sg = (st.definitions?.subgraphs ?? []).find(
        (sg) => sg.id == st.nodes[0].type
      )
      if (!sg)
        throw new Error(
          'Loaded subgraph blueprint does not contain valid subgraph'
        )
      sg.name = st.nodes[0].title = this.filename

      // Copy blueprint metadata from workflow extra to subgraph extra
      // so it's available when editing via canvas.subgraph.extra
      if (st.extra) {
        const sgExtra = (sg.extra ??= {}) as Record<string, unknown>
        for (const [key, value] of Object.entries(st.extra)) {
          if (key === 'workflowRendererVersion') continue
          sgExtra[key] = value
        }
      }

      return loaded
    }
    override async promptSave(): Promise<string | null> {
      return await useDialogService().prompt({
        title: t('subgraphStore.saveBlueprint'),
        message: t('subgraphStore.blueprintNamePrompt'),
        defaultValue: this.filename
      })
    }
    override unload(): void {
      //Skip unloading. Even if a workflow is closed after editing,
      //it must remain loaded in order to be added to the graph
    }
  }
  const subgraphCache: Record<string, LoadedComfyWorkflow> = {}
  const typePrefix = 'SubgraphBlueprint.'
  const subgraphDefCache = ref<Map<string, ComfyNodeDefImpl>>(new Map())
  const canvasStore = useCanvasStore()
  const subgraphBlueprints = computed(() => [
    ...subgraphDefCache.value.values()
  ])
  async function fetchSubgraphs() {
    async function loadBlueprint(options: {
      path: string
      modified: number
      size: number
    }): Promise<void> {
      options.path = SubgraphBlueprint.basePath + options.path
      const bp = await new SubgraphBlueprint(options, true).load()
      useWorkflowStore().attachWorkflow(bp)
      registerNodeDef(bp, { category: 'Subgraph Blueprints/User' })
    }
    async function loadInstalledBlueprints() {
      async function loadGlobalBlueprint([k, v]: [string, GlobalSubgraphData]) {
        const data = await v.data
        if (typeof data !== 'string' || data.trim().length === 0) {
          throw new Error(
            `Global blueprint '${v.name}' (${k}) returned empty content`
          )
        }
        const path = SubgraphBlueprint.basePath + v.name + '.json'
        const blueprint = new SubgraphBlueprint({
          path,
          modified: Date.now(),
          size: -1
        })
        blueprint.originalContent = blueprint.content = data
        blueprint.filename = v.name
        useWorkflowStore().attachWorkflow(blueprint)
        const loaded = await blueprint.load()
        const category = v.info.category
          ? `Subgraph Blueprints/${v.info.category}`
          : undefined
        registerNodeDef(
          loaded,
          {
            display_name: v.name,
            ...(category && { category }),
            ...(v.essentials_category && {
              essentials_category: v.essentials_category
            }),
            search_aliases: v.info.search_aliases,
            isGlobal: true
          },
          k
        )
      }
      const subgraphs = await api.getGlobalSubgraphs()
      const currentDistribution: TemplateIncludeOnDistributionEnum = isCloud
        ? TemplateIncludeOnDistributionEnum.Cloud
        : isDesktop
          ? TemplateIncludeOnDistributionEnum.Desktop
          : TemplateIncludeOnDistributionEnum.Local
      const filteredEntries = Object.entries(subgraphs).filter(([, v]) => {
        if (!isCloud && (v.info.requiresCustomNodes?.length ?? 0) > 0)
          return false
        if (
          (v.info.includeOnDistributions?.length ?? 0) > 0 &&
          !v.info.includeOnDistributions!.includes(currentDistribution)
        )
          return false
        return true
      })
      return Promise.allSettled(filteredEntries.map(loadGlobalBlueprint))
    }

    const userSubs = (
      await api.listUserDataFullInfo(SubgraphBlueprint.basePath)
    ).filter((f) => f.path.endsWith('.json'))
    const [globalResult, ...userResults] = await Promise.allSettled([
      loadInstalledBlueprints(),
      ...userSubs.map(loadBlueprint)
    ])
    const globalResults =
      globalResult.status === 'fulfilled' ? globalResult.value : []
    const settled = [...globalResults, ...userResults]

    const errors = settled.filter((i) => 'reason' in i).map((i) => i.reason)
    errors.forEach((e) => console.error('Failed to load subgraph blueprint', e))
    if (errors.length > 0) {
      useToastStore().add({
        severity: 'error',
        summary: t('subgraphStore.loadFailure'),
        detail: errors.length > 3 ? `x${errors.length}` : `${errors}`,
        life: 6000
      })
    }
  }
  function registerNodeDef(
    workflow: LoadedComfyWorkflow,
    overrides: Partial<ComfyNodeDefV1> = {},
    name: string = workflow.filename
  ) {
    const subgraphNode = workflow.changeTracker.initialState.nodes[0]
    if (!subgraphNode) throw new Error('Invalid Subgraph Blueprint')
    subgraphNode.inputs ??= []
    subgraphNode.outputs ??= []
    //NOTE: Types are cast to string. This is only used for input coloring on previews
    const inputs = Object.fromEntries(
      subgraphNode.inputs.map((i) => [
        i.name,
        [`${i.type}`, undefined] satisfies InputSpec
      ])
    )
    const workflowExtra = workflow.initialState.extra
    const description =
      workflowExtra?.BlueprintDescription ?? 'User generated subgraph blueprint'
    const search_aliases = workflowExtra?.BlueprintSearchAliases
    const subgraphDefCategory =
      workflow.initialState.definitions?.subgraphs?.[0]?.category
    const subgraphDefEssentialsCategory =
      workflow.initialState.definitions?.subgraphs?.[0]?.essentials_category
    const category = subgraphDefCategory
      ? `Subgraph Blueprints/${subgraphDefCategory}`
      : 'Subgraph Blueprints'
    const nodedefv1: ComfyNodeDefV1 = {
      input: { required: inputs },
      output: subgraphNode.outputs.map((o) => `${o.type}`),
      output_name: subgraphNode.outputs.map((o) => o.name),
      name: typePrefix + name,
      display_name: name,
      description,
      category,
      output_node: false,
      python_module: 'blueprint',
      search_aliases,
      essentials_category: subgraphDefEssentialsCategory,
      ...overrides
    }
    const nodeDefImpl = new ComfyNodeDefImpl(nodedefv1)
    subgraphDefCache.value.set(name, nodeDefImpl)
    subgraphCache[name] = workflow
  }
  async function publishSubgraph(providedName?: string) {
    const canvas = canvasStore.getCanvas()
    const subgraphNode = [...canvas.selectedItems][0]
    if (
      canvas.selectedItems.size !== 1 ||
      !(subgraphNode instanceof SubgraphNode)
    )
      throw new TypeError('Must have single SubgraphNode selected to publish')

    const { nodes = [], subgraphs = [] } = canvas._serializeItems([
      subgraphNode
    ])
    if (nodes.length != 1) {
      throw new TypeError('Must have single SubgraphNode selected to publish')
    }

    //create minimal workflow
    const workflowData = {
      revision: 0,
      last_node_id: subgraphNode.id,
      last_link_id: 0,
      nodes,
      links: [] as never[],
      version: 0.4,
      definitions: { subgraphs }
    }
    //prompt name
    const name =
      providedName ??
      (await useDialogService().prompt({
        title: t('subgraphStore.saveBlueprint'),
        message: t('subgraphStore.blueprintNamePrompt'),
        defaultValue: subgraphNode.title
      }))
    if (!name) return
    if (subgraphDefCache.value.has(name) && !(await confirmOverwrite(name)))
      //User has chosen not to overwrite.
      return

    //upload file
    const path = SubgraphBlueprint.basePath + name + '.json'
    const workflow = new SubgraphBlueprint({
      path,
      size: -1,
      modified: Date.now()
    })
    workflow.originalContent = JSON.stringify(workflowData)
    const loadedWorkflow = await workflow.load()
    //Mark non-temporary
    workflow.size = 1
    await workflow.save()
    //add to files list?
    useWorkflowStore().attachWorkflow(loadedWorkflow)
    useToastStore().add({
      severity: 'success',
      summary: t('subgraphStore.publishSuccess'),
      detail: t('subgraphStore.publishSuccessMessage'),
      life: 4000
    })
  }
  async function editBlueprint(nodeType: string) {
    const name = nodeType.slice(typePrefix.length)
    if (!(name in subgraphCache))
      //As loading is blocked on in startup, this can likely be changed to invalid type
      throw new Error('not yet loaded')
    useWorkflowStore().attachWorkflow(subgraphCache[name])
    await useWorkflowService().openWorkflow(subgraphCache[name])
    const canvas = useCanvasStore().getCanvas()
    if (canvas.graph && 'subgraph' in canvas.graph.nodes[0])
      canvas.setGraph(canvas.graph.nodes[0].subgraph)
  }
  function getBlueprint(nodeType: string): ComfyWorkflowJSON {
    const name = nodeType.slice(typePrefix.length)
    if (!(name in subgraphCache))
      //As loading is blocked on in startup, this can likely be changed to invalid type
      throw new Error('not yet loaded')
    return subgraphCache[name].changeTracker.initialState
  }
  async function deleteBlueprint(nodeType: string) {
    const name = nodeType.slice(typePrefix.length)
    if (!(name in subgraphCache)) throw new Error('not yet loaded')

    if (isGlobalBlueprint(name)) {
      useToastStore().add({
        severity: 'warn',
        summary: t('subgraphStore.cannotDeleteGlobal'),
        life: 4000
      })
      return
    }

    if (
      !(await useDialogService().confirm({
        title: t('subgraphStore.confirmDeleteTitle'),
        type: 'delete',
        message: t('subgraphStore.confirmDelete'),
        itemList: [name]
      }))
    )
      return

    await subgraphCache[name].delete()
    delete subgraphCache[name]
    subgraphDefCache.value.delete(name)
  }
  function isSubgraphBlueprint(
    workflow: unknown
  ): workflow is SubgraphBlueprint {
    return workflow instanceof SubgraphBlueprint
  }

  function isGlobalBlueprint(name: string): boolean {
    const nodeDef = subgraphDefCache.value.get(name)
    return nodeDef !== undefined && nodeDef.isGlobal === true
  }

  return {
    deleteBlueprint,
    editBlueprint,
    fetchSubgraphs,
    getBlueprint,
    isGlobalBlueprint,
    isSubgraphBlueprint,
    publishSubgraph,
    subgraphBlueprints,
    typePrefix
  }
})
