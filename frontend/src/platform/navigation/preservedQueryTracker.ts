import type { Router } from 'vue-router'

import {
  capturePreservedQuery,
  hydratePreservedQuery
} from '@/platform/navigation/preservedQueryManager'

export const installPreservedQueryTracker = (
  router: Router,
  definitions: Array<{ namespace: string; keys: string[] }>
) => {
  const trackedDefinitions = definitions.map((definition) => ({
    ...definition
  }))

  router.beforeEach((to, _from, next) => {
    const queryKeys = new Set(Object.keys(to.query))

    trackedDefinitions.forEach(({ namespace, keys }) => {
      hydratePreservedQuery(namespace)
      const shouldCapture = keys.some((key) => queryKeys.has(key))
      if (shouldCapture) {
        capturePreservedQuery(namespace, to.query, keys)
      }
    })

    next()
  })
}
