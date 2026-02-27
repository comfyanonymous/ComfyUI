import { markRaw } from 'vue'

import { t } from '@/i18n'
import type { ChangeTracker } from '@/scripts/changeTracker'
import type { AppMode } from '@/composables/useAppMode'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import { UserFile } from '@/stores/userFileStore'
import type {
  ComfyWorkflowJSON,
  ModelFile
} from '@/platform/workflow/validation/schemas/workflowSchema'
import type { MissingNodeType } from '@/types/comfy'

export interface LinearData {
  inputs: [NodeId, string][]
  outputs: NodeId[]
}

export interface PendingWarnings {
  missingNodeTypes?: MissingNodeType[]
  missingModels?: {
    missingModels: ModelFile[]
    paths: Record<string, string[]>
  }
}

type LinearModeTarget = { extra?: Record<string, unknown> | null } | null

export function syncLinearMode(
  workflow: ComfyWorkflow,
  targets: LinearModeTarget[],
  options?: { flushLinearData?: boolean }
): void {
  for (const target of targets) {
    if (!target) continue
    if (workflow.initialMode === 'app' || workflow.initialMode === 'graph') {
      const extra = (target.extra ??= {})
      extra.linearMode = workflow.initialMode === 'app'
    } else {
      delete target.extra?.linearMode
    }
    if (options?.flushLinearData && workflow.dirtyLinearData) {
      const extra = (target.extra ??= {})
      extra.linearData = workflow.dirtyLinearData
    }
  }
  if (options?.flushLinearData && workflow.dirtyLinearData) {
    workflow.dirtyLinearData = null
  }
}

export class ComfyWorkflow extends UserFile {
  static readonly basePath: string = 'workflows/'
  readonly tintCanvasBg?: string

  /**
   * The change tracker for the workflow. Non-reactive raw object.
   */
  changeTracker: ChangeTracker | null = null
  /**
   * Whether the workflow has been modified comparing to the initial state.
   */
  _isModified: boolean = false
  /**
   * Warnings deferred from load time, shown when the workflow is first focused.
   */
  pendingWarnings: PendingWarnings | null = null
  /**
   * Initial app mode derived from the serialized workflow (extra.linearMode).
   * - `undefined`: not yet resolved (first load hasn't happened)
   * - `null`: resolved, but no mode was set (never builder-saved)
   * - `AppMode`: resolved to a specific mode
   */
  initialMode: AppMode | null | undefined = undefined
  /**
   * Current app mode set by the user during the session.
   * Takes precedence over initialMode when present.
   */
  activeMode: AppMode | null = null
  /**
   * In-progress builder selections not yet persisted via save.
   * Preserved across tab switches, discarded on exitBuilder.
   */
  dirtyLinearData: LinearData | null = null

  /**
   * @param options The path, modified, and size of the workflow.
   * Note: path is the full path, including the 'workflows/' prefix.
   */
  constructor(options: { path: string; modified: number; size: number }) {
    super(options.path, options.modified, options.size)
  }

  override get key() {
    return this.path.substring(ComfyWorkflow.basePath.length)
  }

  get activeState(): ComfyWorkflowJSON | null {
    return this.changeTracker?.activeState ?? null
  }

  get initialState(): ComfyWorkflowJSON | null {
    return this.changeTracker?.initialState ?? null
  }

  override get isLoaded(): boolean {
    return this.changeTracker !== null
  }

  override get isModified(): boolean {
    return this._isModified
  }

  override set isModified(value: boolean) {
    this._isModified = value
  }

  /**
   * Load the workflow content from remote storage. Directly returns the loaded
   * workflow if the content is already loaded.
   *
   * @param force Whether to force loading the content even if it is already loaded.
   * @returns this
   */
  override async load({ force = false }: { force?: boolean } = {}): Promise<
    this & LoadedComfyWorkflow
  > {
    const { useWorkflowDraftStore } =
      await import('@/platform/workflow/persistence/stores/workflowDraftStore')
    const { useSettingStore } = await import('@/platform/settings/settingStore')
    const draftStore = useWorkflowDraftStore()
    const persistEnabled = useSettingStore().get('Comfy.Workflow.Persist')
    let draft =
      !force && persistEnabled ? draftStore.getDraft(this.path) : undefined
    let draftState: ComfyWorkflowJSON | null = null
    let draftContent: string | null = null

    if (draft) {
      if (draft.updatedAt < this.lastModified) {
        draftStore.removeDraft(this.path)
        draft = undefined
      }
    }

    if (draft) {
      try {
        draftState = JSON.parse(draft.data)
        draftContent = draft.data
      } catch (err) {
        console.warn('Failed to parse workflow draft, clearing it', err)
        draftStore.removeDraft(this.path)
      }
    }

    await super.load({ force })
    if (!force && this.isLoaded) return this as this & LoadedComfyWorkflow

    if (this.originalContent == null) {
      throw new Error(
        `[ASSERT] Workflow content should be loaded for '${this.path}'`
      )
    }
    if (this.originalContent.trim().length === 0) {
      throw new Error(`Workflow content is empty for '${this.path}'`)
    }

    const initialState = JSON.parse(this.originalContent)
    const { ChangeTracker } = await import('@/scripts/changeTracker')
    this.changeTracker = markRaw(new ChangeTracker(this, initialState))
    if (draftState && draftContent) {
      this.changeTracker.activeState = draftState
      this.content = draftContent
      this._isModified = true
      draftStore.markDraftUsed(this.path)
    }
    return this as this & LoadedComfyWorkflow
  }

  override unload(): void {
    this.changeTracker = null
    this.activeMode = null
    super.unload()
  }

  override async save() {
    const { useWorkflowDraftStore } =
      await import('@/platform/workflow/persistence/stores/workflowDraftStore')
    const draftStore = useWorkflowDraftStore()
    this.content = JSON.stringify(this.activeState)
    // Force save to ensure the content is updated in remote storage incase
    // the isModified state is screwed by changeTracker.
    const ret = await super.save({ force: true })
    this.changeTracker?.reset()
    this.isModified = false
    draftStore.removeDraft(this.path)
    return ret
  }

  /**
   * Save the workflow as a new file.
   * @param path The path to save the workflow to. Note: with 'workflows/' prefix.
   * @returns this
   */
  override async saveAs(path: string) {
    const { useWorkflowDraftStore } =
      await import('@/platform/workflow/persistence/stores/workflowDraftStore')
    const draftStore = useWorkflowDraftStore()
    this.content = JSON.stringify(this.activeState)
    const result = await super.saveAs(path)
    draftStore.removeDraft(path)
    return result
  }

  async promptSave(): Promise<string | null> {
    const { useDialogService } = await import('@/services/dialogService')
    return await useDialogService().prompt({
      title: t('workflowService.saveWorkflow'),
      message: t('workflowService.enterFilenamePrompt'),
      defaultValue: this.filename
    })
  }
}

export interface LoadedComfyWorkflow extends ComfyWorkflow {
  isLoaded: true
  originalContent: string
  content: string
  changeTracker: ChangeTracker
  initialState: ComfyWorkflowJSON
  activeState: ComfyWorkflowJSON
}
