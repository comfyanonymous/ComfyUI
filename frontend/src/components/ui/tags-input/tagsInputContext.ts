import type { InjectionKey, Ref } from 'vue'

export type FocusCallback = (() => void) | undefined

export const tagsInputFocusKey: InjectionKey<
  (callback: FocusCallback) => void
> = Symbol('tagsInputFocus')

export const tagsInputIsEditingKey: InjectionKey<Ref<boolean>> =
  Symbol('tagsInputIsEditing')
