import { useSettingStore } from '@/platform/settings/settingStore'
import { WORKFLOW_ACCEPT_STRING } from '@/platform/workflow/core/types/formats'
import { type StatusWsMessageStatus } from '@/schemas/apiSchema'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { isCloud } from '@/platform/distribution/types'
import { extractWorkflow } from '@/platform/remote/comfyui/jobs/fetchJobs'
import type { JobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import { useTelemetry } from '@/platform/telemetry'
import { useLitegraphService } from '@/services/litegraphService'
import { useCommandStore } from '@/stores/commandStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'

import { api } from './api'
import { ComfyApp, app } from './app'
import { ComfyDialog as _ComfyDialog } from './ui/dialog'
import { ComfySettingsDialog } from './ui/settings'
import { toggleSwitch } from './ui/toggleSwitch'

export const ComfyDialog = _ComfyDialog

type Position2D = {
  x: number
  y: number
}

type Props = {
  parent?: HTMLElement
  $?: (el: HTMLElement) => void
  dataset?: DOMStringMap
  style?: Partial<CSSStyleDeclaration>
  for?: string
  textContent?: string
  [key: string]: unknown
}

type Children = Element[] | Element | string | string[]

type ElementType<K extends string> = K extends keyof HTMLElementTagNameMap
  ? HTMLElementTagNameMap[K]
  : HTMLElement

export function $el<TTag extends string>(
  tag: TTag,
  propsOrChildren?: Children | Props,
  children?: Children
): ElementType<TTag> {
  const split = tag.split('.')
  const element = document.createElement(split.shift() as string)
  if (split.length > 0) {
    element.classList.add(...split)
  }

  if (propsOrChildren) {
    if (typeof propsOrChildren === 'string') {
      propsOrChildren = { textContent: propsOrChildren }
    } else if (propsOrChildren instanceof Element) {
      propsOrChildren = [propsOrChildren]
    }
    if (Array.isArray(propsOrChildren)) {
      element.append(...propsOrChildren)
    } else {
      const {
        parent,
        $: cb,
        dataset,
        style,
        ...rest
      } = propsOrChildren as Props

      if (rest.for) {
        element.setAttribute('for', rest.for)
      }

      if (style) {
        Object.assign(element.style, style)
      }

      if (dataset) {
        Object.assign(element.dataset, dataset)
      }

      Object.assign(element, rest)
      if (children) {
        element.append(...(Array.isArray(children) ? children : [children]))
      }

      if (parent) {
        parent.append(element)
      }

      if (cb) {
        cb(element)
      }
    }
  }
  return element as ElementType<TTag>
}

// @ts-expect-error fixme ts strict error
function dragElement(dragEl): () => void {
  var posDiffX = 0,
    posDiffY = 0,
    posStartX = 0,
    posStartY = 0,
    newPosX = 0,
    newPosY = 0
  if (dragEl.getElementsByClassName('drag-handle')[0]) {
    // if present, the handle is where you move the DIV from:
    dragEl.getElementsByClassName('drag-handle')[0].onmousedown = dragMouseDown
  } else {
    // otherwise, move the DIV from anywhere inside the DIV:
    dragEl.onmousedown = dragMouseDown
  }

  // When the element resizes (e.g. view queue) ensure it is still in the windows bounds
  new ResizeObserver(() => {
    ensureInBounds()
  }).observe(dragEl)

  function ensureInBounds() {
    try {
      newPosX = Math.min(
        document.body.clientWidth - dragEl.clientWidth,
        Math.max(0, dragEl.offsetLeft)
      )
      newPosY = Math.min(
        document.body.clientHeight - dragEl.clientHeight,
        Math.max(0, dragEl.offsetTop)
      )

      positionElement()
    } catch (exception) {
      // robust
    }
  }

  function positionElement() {
    if (dragEl.style.display === 'none') return

    const halfWidth = document.body.clientWidth / 2
    const anchorRight = newPosX + dragEl.clientWidth / 2 > halfWidth

    // set the element's new position:
    if (anchorRight) {
      dragEl.style.left = 'unset'
      dragEl.style.right =
        document.body.clientWidth - newPosX - dragEl.clientWidth + 'px'
    } else {
      dragEl.style.left = newPosX + 'px'
      dragEl.style.right = 'unset'
    }

    dragEl.style.top = newPosY + 'px'
    dragEl.style.bottom = 'unset'

    // @ts-expect-error fixme ts strict error
    if (savePos) {
      localStorage.setItem(
        'Comfy.MenuPosition',
        JSON.stringify({
          x: dragEl.offsetLeft,
          y: dragEl.offsetTop
        })
      )
    }
  }

  function restorePos() {
    let posString = localStorage.getItem('Comfy.MenuPosition')
    if (posString) {
      const pos = JSON.parse(posString) as Position2D
      newPosX = pos.x
      newPosY = pos.y
      positionElement()
      ensureInBounds()
    }
  }

  // @ts-expect-error fixme ts strict error
  let savePos = undefined
  restorePos()
  savePos = true

  // @ts-expect-error fixme ts strict error
  function dragMouseDown(e) {
    e = e || window.event
    e.preventDefault()
    // get the mouse cursor position at startup:
    posStartX = e.clientX
    posStartY = e.clientY
    document.onmouseup = closeDragElement
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag
  }

  // @ts-expect-error fixme ts strict error
  function elementDrag(e) {
    e = e || window.event
    e.preventDefault()

    dragEl.classList.add('comfy-menu-manual-pos')

    // calculate the new cursor position:
    posDiffX = e.clientX - posStartX
    posDiffY = e.clientY - posStartY
    posStartX = e.clientX
    posStartY = e.clientY

    newPosX = Math.min(
      document.body.clientWidth - dragEl.clientWidth,
      Math.max(0, dragEl.offsetLeft + posDiffX)
    )
    newPosY = Math.min(
      document.body.clientHeight - dragEl.clientHeight,
      Math.max(0, dragEl.offsetTop + posDiffY)
    )

    positionElement()
  }

  window.addEventListener('resize', () => {
    ensureInBounds()
  })

  function closeDragElement() {
    // stop moving when mouse button is released:
    document.onmouseup = null
    document.onmousemove = null
  }

  return restorePos
}

class ComfyList {
  private _type
  private _text
  private _reverse
  element: HTMLDivElement
  button?: HTMLButtonElement

  // @ts-expect-error fixme ts strict error
  constructor(text, type?, reverse?) {
    this._text = text
    this._type = type || text.toLowerCase()
    this._reverse = reverse || false
    this.element = $el('div.comfy-list') as HTMLDivElement
    this.element.style.display = 'none'
  }

  get visible() {
    return this.element.style.display !== 'none'
  }

  async load() {
    const items =
      this._type === 'history'
        ? { history: await api.getHistory() }
        : await api.getQueue()
    this.element.replaceChildren(
      ...Object.keys(items).flatMap((section) => [
        $el('h4', {
          textContent: section
        }),
        $el('div.comfy-list-items', [
          // @ts-expect-error fixme ts strict error
          ...(this._reverse ? items[section].reverse() : items[section]).map(
            (item: JobListItem) => {
              // Allow items to specify a custom remove action (e.g. for interrupt current prompt)
              const removeAction =
                section === 'Running'
                  ? {
                      name: 'Cancel',
                      cb: () => api.interrupt(item.id)
                    }
                  : {
                      name: 'Delete',
                      cb: () => api.deleteItem(this._type, item.id)
                    }
              return $el('div', { textContent: item.priority + ': ' }, [
                $el('button', {
                  textContent: 'Load',
                  onclick: async () => {
                    const job = await api.getJobDetail(item.id)
                    if (!job) return
                    const workflow = await extractWorkflow(job)
                    await app.loadGraphData(workflow, true, false)
                    if ('outputs' in job && job.outputs) {
                      app.nodeOutputs = {}
                      for (const [key, value] of Object.entries(job.outputs)) {
                        app.nodeOutputs[key] = value
                      }
                    }
                  }
                }),
                $el('button', {
                  textContent: removeAction.name,
                  onclick: async () => {
                    await removeAction.cb()
                    await this.update()
                  }
                })
              ])
            }
          )
        ])
      ]),
      $el('div.comfy-list-actions', [
        $el('button', {
          textContent: 'Clear ' + this._text,
          onclick: async () => {
            await api.clearItems(this._type)
            await this.load()
          }
        }),
        $el('button', { textContent: 'Refresh', onclick: () => this.load() })
      ])
    )
  }

  async update() {
    if (this.visible) {
      await this.load()
    }
  }

  async show() {
    this.element.style.display = 'block'
    // @ts-expect-error fixme ts strict error
    this.button.textContent = 'Close'

    await this.load()
  }

  hide() {
    this.element.style.display = 'none'
    // @ts-expect-error fixme ts strict error
    this.button.textContent = 'View ' + this._text
  }

  toggle() {
    if (this.visible) {
      this.hide()
      return false
    } else {
      this.show()
      return true
    }
  }
}

export class ComfyUI {
  app: ComfyApp
  dialog: _ComfyDialog
  settings: ComfySettingsDialog
  batchCount: number
  lastQueueSize: number
  queue: ComfyList
  history: ComfyList
  // @ts-expect-error fixme ts strict error
  autoQueueMode: string
  // @ts-expect-error fixme ts strict error
  graphHasChanged: boolean
  // @ts-expect-error fixme ts strict error
  autoQueueEnabled: boolean
  // @ts-expect-error fixme ts strict error
  menuContainer: HTMLDivElement
  // @ts-expect-error fixme ts strict error
  queueSize: Element
  // @ts-expect-error fixme ts strict error
  restoreMenuPosition: () => void
  // @ts-expect-error fixme ts strict error
  loadFile: () => void

  // @ts-expect-error fixme ts strict error
  constructor(app) {
    this.app = app
    this.dialog = new ComfyDialog()
    this.settings = new ComfySettingsDialog(app)

    this.batchCount = 1
    this.lastQueueSize = 0
    this.queue = new ComfyList('Queue')
    this.history = new ComfyList('History', 'history', true)

    api.addEventListener('status', () => {
      this.queue.update()
      this.history.update()
    })

    this.setup(document.body)
  }

  setup(containerElement: HTMLElement) {
    const fileInput = $el('input', {
      id: 'comfy-file-input',
      type: 'file',
      accept: WORKFLOW_ACCEPT_STRING,
      style: { display: 'none' },
      parent: document.body,
      onchange: async () => {
        const file = fileInput.files?.[0]
        if (file) {
          await app.handleFile(file, 'file_button')
          fileInput.value = ''
        }
      }
    })

    this.loadFile = () => fileInput.click()

    const autoQueueModeEl = toggleSwitch(
      'autoQueueMode',
      [
        {
          text: 'instant',
          tooltip: 'A new prompt will be queued as soon as the queue reaches 0'
        },
        {
          text: 'change',
          tooltip:
            'A new prompt will be queued when the queue is at 0 and the graph is/has changed'
        }
      ],
      {
        // @ts-expect-error fixme ts strict error
        onChange: (value) => {
          this.autoQueueMode = value.item.value
        }
      }
    )
    autoQueueModeEl.style.display = 'none'

    api.addEventListener('graphChanged', () => {
      if (this.autoQueueMode === 'change' && this.autoQueueEnabled === true) {
        if (this.lastQueueSize === 0) {
          this.graphHasChanged = false
          app.queuePrompt(0, this.batchCount)
        } else {
          this.graphHasChanged = true
        }
      }
    })

    this.menuContainer = $el(
      'div.comfy-menu.no-drag',
      { parent: containerElement },
      [
        $el(
          'div.drag-handle.comfy-menu-header',
          {
            style: {
              overflow: 'hidden',
              position: 'relative',
              width: '100%',
              cursor: 'default'
            }
          },
          [
            $el('span.drag-handle'),
            $el('span.comfy-menu-queue-size', {
              $: (q) => (this.queueSize = q)
            }),
            $el('div.comfy-menu-actions', [
              $el('button.comfy-settings-btn', {
                textContent: '⚙️',
                onclick: () => {
                  useSettingsDialog().show()
                }
              }),
              $el('button.comfy-close-menu-btn', {
                textContent: '\u00d7',
                onclick: () => {
                  useWorkspaceStore().focusMode = true
                }
              })
            ])
          ]
        ),
        $el('button.comfy-queue-btn', {
          id: 'queue-button',
          textContent: 'Queue Prompt',
          onclick: () => {
            if (isCloud) {
              useTelemetry()?.trackRunButton({ trigger_source: 'legacy_ui' })
              useTelemetry()?.trackWorkflowExecution()
            }
            app.queuePrompt(0, this.batchCount)
          }
        }),
        $el('div', {}, [
          $el('label', { innerHTML: 'Extra options' }, [
            $el('input', {
              type: 'checkbox',
              // @ts-expect-error fixme ts strict error
              onchange: (i) => {
                // @ts-expect-error fixme ts strict error
                document.getElementById('extraOptions').style.display = i
                  .srcElement.checked
                  ? 'block'
                  : 'none'
                this.batchCount = i.srcElement.checked
                  ? Number.parseInt(
                      (
                        document.getElementById(
                          'batchCountInputRange'
                        ) as HTMLInputElement
                      ).value
                    )
                  : 1
                ;(
                  document.getElementById(
                    'autoQueueCheckbox'
                  ) as HTMLInputElement
                ).checked = false
                this.autoQueueEnabled = false
              }
            })
          ])
        ]),
        $el(
          'div',
          { id: 'extraOptions', style: { width: '100%', display: 'none' } },
          [
            $el('div', [
              $el('label', { innerHTML: 'Batch count' }),
              $el('input', {
                id: 'batchCountInputNumber',
                type: 'number',
                value: this.batchCount,
                min: '1',
                style: { width: '35%', marginLeft: '0.4em' },
                // @ts-expect-error fixme ts strict error
                oninput: (i) => {
                  this.batchCount = i.target.value
                  /* Even though an <input> element with a type of range logically represents a number (since
              it's used for numeric input), the value it holds is still treated as a string in HTML and
              JavaScript. This behavior is consistent across all <input> elements regardless of their type
              (like text, number, or range), where the .value property is always a string. */
                  ;(
                    document.getElementById(
                      'batchCountInputRange'
                    ) as HTMLInputElement
                  ).value = this.batchCount.toString()
                }
              }),
              $el('input', {
                id: 'batchCountInputRange',
                type: 'range',
                min: '1',
                max: '100',
                value: this.batchCount,
                // @ts-expect-error fixme ts strict error
                oninput: (i) => {
                  this.batchCount = i.srcElement.value
                  // Note
                  ;(
                    document.getElementById(
                      'batchCountInputNumber'
                    ) as HTMLInputElement
                  ).value = i.srcElement.value
                }
              })
            ]),
            $el('div', [
              $el('label', {
                for: 'autoQueueCheckbox',
                innerHTML: 'Auto Queue'
              }),
              $el('input', {
                id: 'autoQueueCheckbox',
                type: 'checkbox',
                checked: false,
                title: 'Automatically queue prompt when the queue size hits 0',
                // @ts-expect-error fixme ts strict error
                onchange: (e) => {
                  this.autoQueueEnabled = e.target.checked
                  autoQueueModeEl.style.display = this.autoQueueEnabled
                    ? ''
                    : 'none'
                }
              }),
              autoQueueModeEl
            ])
          ]
        ),
        $el('div.comfy-menu-btns', [
          $el('button', {
            id: 'queue-front-button',
            textContent: 'Queue Front',
            onclick: () => {
              if (isCloud) {
                useTelemetry()?.trackRunButton({ trigger_source: 'legacy_ui' })
                useTelemetry()?.trackWorkflowExecution()
              }
              app.queuePrompt(-1, this.batchCount)
            }
          }),
          $el('button', {
            $: (b) => (this.queue.button = b as HTMLButtonElement),
            id: 'comfy-view-queue-button',
            textContent: 'View Queue',
            onclick: () => {
              this.history.hide()
              this.queue.toggle()
            }
          }),
          $el('button', {
            $: (b) => (this.history.button = b as HTMLButtonElement),
            id: 'comfy-view-history-button',
            textContent: 'View History',
            onclick: () => {
              this.queue.hide()
              this.history.toggle()
            }
          })
        ]),
        this.queue.element,
        this.history.element,
        $el('button', {
          id: 'comfy-save-button',
          textContent: 'Save',
          onclick: () => {
            useCommandStore().execute('Comfy.ExportWorkflow')
          }
        }),
        $el('button', {
          id: 'comfy-dev-save-api-button',
          textContent: 'Save (API Format)',
          style: { width: '100%', display: 'none' },
          onclick: () => {
            useCommandStore().execute('Comfy.ExportWorkflowAPI')
          }
        }),
        $el('button', {
          id: 'comfy-load-button',
          textContent: 'Load',
          onclick: () => fileInput.click()
        }),
        $el('button', {
          id: 'comfy-refresh-button',
          textContent: 'Refresh',
          onclick: () => app.refreshComboInNodes()
        }),
        $el('button', {
          id: 'comfy-clipspace-button',
          textContent: 'Clipspace',
          onclick: () => app.openClipspace()
        }),
        $el('button', {
          id: 'comfy-clear-button',
          textContent: 'Clear',
          onclick: () => {
            if (
              !useSettingStore().get('Comfy.ConfirmClear') ||
              confirm('Clear workflow?')
            ) {
              app.clean()
              useLitegraphService().resetView()
              api.dispatchCustomEvent('graphCleared')
            }
          }
        }),
        $el('button', {
          id: 'comfy-load-default-button',
          textContent: 'Load Default',
          onclick: async () => {
            if (
              !useSettingStore().get('Comfy.ConfirmClear') ||
              confirm('Load default workflow?')
            ) {
              useLitegraphService().resetView()
              await app.loadGraphData()
            }
          }
        }),
        $el('button', {
          id: 'comfy-reset-view-button',
          textContent: 'Reset View',
          onclick: async () => {
            useLitegraphService().resetView()
          }
        })
      ]
    ) as HTMLDivElement
    // Hide by default on construction so it does not interfere with other views.
    this.menuContainer.style.display = 'none'

    this.restoreMenuPosition = dragElement(this.menuContainer)

    // @ts-expect-error
    this.setStatus({ exec_info: { queue_remaining: 'X' } })
  }

  setStatus(status: StatusWsMessageStatus | null) {
    this.queueSize.textContent =
      'Queue size: ' + (status ? status.exec_info.queue_remaining : 'ERR')
    if (status) {
      if (
        this.lastQueueSize != 0 &&
        status.exec_info.queue_remaining == 0 &&
        this.autoQueueEnabled &&
        (this.autoQueueMode === 'instant' || this.graphHasChanged) &&
        !app.lastExecutionError
      ) {
        app.queuePrompt(0, this.batchCount)
        status.exec_info.queue_remaining += this.batchCount
        this.graphHasChanged = false
      }
      this.lastQueueSize = status.exec_info.queue_remaining
    }
  }
}
