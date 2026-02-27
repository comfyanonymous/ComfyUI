// Within Vue component context, you can directly call useToast().add()
// instead of going through the store.
// The store is useful when you need to call it from outside the Vue component context.
import { defineStore } from 'pinia'
import type { ToastMessageOptions } from 'primevue/toast'
import { ref } from 'vue'

export const useToastStore = defineStore('toast', () => {
  const messagesToAdd = ref<ToastMessageOptions[]>([])
  const messagesToRemove = ref<ToastMessageOptions[]>([])
  const removeAllRequested = ref(false)

  function add(message: ToastMessageOptions) {
    messagesToAdd.value = [...messagesToAdd.value, message]
  }

  function remove(message: ToastMessageOptions) {
    messagesToRemove.value = [...messagesToRemove.value, message]
  }

  function removeAll() {
    removeAllRequested.value = true
  }

  function addAlert(message: string) {
    add({ severity: 'warn', summary: 'Alert', detail: message })
  }

  return {
    messagesToAdd,
    messagesToRemove,
    removeAllRequested,

    add,
    remove,
    removeAll,
    addAlert
  }
})
