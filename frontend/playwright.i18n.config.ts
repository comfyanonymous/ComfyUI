import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './scripts',
  use: {
    baseURL: 'http://localhost:5173',
    headless: true
  },
  reporter: 'list',
  workers: 1,
  timeout: 60000,
  testMatch: /collect-i18n-.*\.ts/
})
