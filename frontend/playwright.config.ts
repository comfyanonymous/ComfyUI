import { defineConfig, devices } from '@playwright/test'
import type { PlaywrightTestConfig } from '@playwright/test'

const maybeLocalOptions: PlaywrightTestConfig = process.env.PLAYWRIGHT_LOCAL
  ? {
      // VERY HELPFUL: Skip screenshot tests locally
      // grep: process.env.CI ? undefined : /^(?!.*screenshot).*$/,
      timeout: 30_000, // Longer timeout for breakpoints
      retries: 0, // No retries while debugging. Increase if writing new tests. that may be flaky.
      workers: 1, // Single worker for easier debugging. Increase to match CPU cores if you want to run a lot of tests in parallel.

      use: {
        trace: 'on', // Always capture traces (CI uses 'on-first-retry')
        video: 'on' // Always record video (CI uses 'retain-on-failure')
      }
    }
  : {
      retries: process.env.CI ? 3 : 0,
      use: {
        trace: 'on-first-retry'
      }
    }

export default defineConfig({
  testDir: './browser_tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: 'html',
  ...maybeLocalOptions,

  globalSetup: './browser_tests/globalSetup.ts',
  globalTeardown: './browser_tests/globalTeardown.ts',

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
      timeout: 15000,
      grepInvert: /@mobile|@perf/ // Run all tests except those tagged with @mobile or @perf
    },

    {
      name: 'performance',
      use: {
        ...devices['Desktop Chrome'],
        trace: 'retain-on-failure'
      },
      timeout: 60_000,
      grep: /@perf/,
      fullyParallel: false
    },

    {
      name: 'chromium-2x',
      use: { ...devices['Desktop Chrome'], deviceScaleFactor: 2 },
      timeout: 15000,
      grep: /@2x/ // Run all tests tagged with @2x
    },

    {
      name: 'chromium-0.5x',
      use: { ...devices['Desktop Chrome'], deviceScaleFactor: 0.5 },
      timeout: 15000,
      grep: /@0.5x/ // Run all tests tagged with @0.5x
    },

    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },

    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },

    /* Test against mobile viewports. */
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'], hasTouch: true },
      grep: /@mobile/ // Run only tests tagged with @mobile
    }
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: { ...devices['Desktop Edge'], channel: 'msedge' },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ]

  /* Run your local dev server before starting the tests */
  // webServer: {
  //   command: 'pnpm dev',
  //   url: 'http://127.0.0.1:5173',
  //   reuseExistingServer: !process.env.CI,
  // },
})
