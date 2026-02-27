import { $el } from '../../ui'
import { ComfyDialog } from '../dialog'

type DialogAction<T> = string | { value?: T; text: string }

export class ComfyAsyncDialog<
  T = string | null
> extends ComfyDialog<HTMLDialogElement> {
  private _resolve: (value: T | null) => void = () => {}

  constructor(actions?: Array<DialogAction<T>>) {
    super(
      'dialog.comfy-dialog.comfyui-dialog',
      actions?.map((opt) => {
        const action = typeof opt === 'string' ? { text: opt } : opt
        return $el('button.comfyui-button', {
          type: 'button',
          textContent: action.text,
          onclick: () => this.close((action.value ?? action.text) as T)
        }) as HTMLButtonElement
      })
    )
  }

  override show(html: string | HTMLElement | HTMLElement[]): Promise<T | null> {
    this.element.addEventListener('close', () => {
      this.close()
    })

    super.show(html)

    return new Promise((resolve) => {
      this._resolve = resolve
    })
  }

  showModal(html: string | HTMLElement | HTMLElement[]): Promise<T | null> {
    this.element.addEventListener('close', () => {
      this.close()
    })

    super.show(html)
    this.element.showModal()

    return new Promise((resolve) => {
      this._resolve = resolve
    })
  }

  override close(result: T | null = null) {
    this._resolve(result)
    this.element.close()
    super.close()
  }

  static async prompt<U = string>({
    title = null,
    message,
    actions
  }: {
    title: string | null
    message: string
    actions: Array<DialogAction<U>>
  }): Promise<U | null> {
    const dialog = new ComfyAsyncDialog<U>(actions)
    const content = [$el('span', message)]
    if (title) {
      content.unshift($el('h3', title))
    }
    const res = await dialog.showModal(content)
    dialog.element.remove()
    return res
  }
}
