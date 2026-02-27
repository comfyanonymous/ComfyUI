import { test as base } from '@playwright/test'

export const webSocketFixture = base.extend<{
  ws: { trigger(data: unknown, url?: string): Promise<void> }
}>({
  ws: [
    async ({ page }, use) => {
      // Each time a page loads, to catch navigations
      page.on('load', async () => {
        await page.evaluate(function () {
          // Create a wrapper for WebSocket that stores them globally
          // so we can look it up to trigger messages
          const store: Record<string, WebSocket> = (window.__ws__ = {})
          window.WebSocket = class extends window.WebSocket {
            constructor(
              ...rest: ConstructorParameters<typeof window.WebSocket>
            ) {
              super(...rest)
              store[this.url] = this
            }
          }
        })
      })

      await use({
        async trigger(data, url) {
          // Trigger a websocket event on the page
          await page.evaluate(
            function ([data, url]) {
              if (!url) {
                // If no URL specified, use page URL
                const u = new URL(window.location.toString())
                u.protocol = 'ws:'
                u.pathname = '/'
                url = u.toString() + 'ws'
              }
              const ws: WebSocket = window.__ws__![url]
              ws.dispatchEvent(
                new MessageEvent('message', {
                  data
                })
              )
            },
            [JSON.stringify(data), url]
          )
        }
      })
    },
    // We need this to run automatically as the first thing so it adds handlers as soon as the page loads
    { auto: true }
  ]
})
