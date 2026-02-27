import { applyTextReplacements as _applyTextReplacements } from '@/utils/searchAndReplace'

import { api } from './api'
import type { ComfyApp } from './app'
import { $el } from './ui'

export function clone<T>(obj: T): T {
  try {
    if (typeof structuredClone !== 'undefined') {
      return structuredClone(obj)
    }
  } catch (error) {
    // structuredClone is stricter than using JSON.parse/stringify so fallback to that
  }

  return JSON.parse(JSON.stringify(obj))
}

/**
 * @deprecated Use `applyTextReplacements` from `@/utils/searchAndReplace` instead
 * There are external callers to this function, so we need to keep it for now
 */
export function applyTextReplacements(app: ComfyApp, value: string): string {
  return _applyTextReplacements(app.rootGraph, value)
}

export async function addStylesheet(
  urlOrFile: string,
  relativeTo?: string
): Promise<void> {
  return new Promise((res, rej) => {
    let url
    if (urlOrFile.endsWith('.js')) {
      url = urlOrFile.substr(0, urlOrFile.length - 2) + 'css'
    } else {
      url = new URL(
        urlOrFile,
        relativeTo ?? `${window.location.protocol}//${window.location.host}`
      ).toString()
    }
    $el('link', {
      parent: document.head,
      rel: 'stylesheet',
      type: 'text/css',
      href: url,
      onload: res,
      onerror: rej
    })
  })
}

export { downloadBlob } from '@/base/common/downloadUtil'

export function uploadFile(accept: string) {
  return new Promise<File>((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return reject(new Error('No file selected'))
      resolve(file)
    }
    input.click()
  })
}

export function prop<T>(
  target: object,
  name: string,
  defaultValue: T,
  onChanged?: (
    currentValue: T,
    previousValue: T,
    target: object,
    name: string
  ) => void
): T {
  // @ts-expect-error fixme ts strict error
  let currentValue
  Object.defineProperty(target, name, {
    get() {
      // @ts-expect-error fixme ts strict error
      return currentValue
    },
    set(newValue) {
      // @ts-expect-error fixme ts strict error
      const prevValue = currentValue
      currentValue = newValue
      onChanged?.(currentValue, prevValue, target, name)
    }
  })
  return defaultValue
}

export function getStorageValue(id: string) {
  const clientId = api.clientId ?? api.initialClientId
  return (
    (clientId && sessionStorage.getItem(`${id}:${clientId}`)) ??
    localStorage.getItem(id)
  )
}

export function setStorageValue(id: string, value: string) {
  const clientId = api.clientId ?? api.initialClientId
  if (clientId) {
    sessionStorage.setItem(`${id}:${clientId}`, value)
  }
  localStorage.setItem(id, value)
}
