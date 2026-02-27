#!/usr/bin/env tsx
import fs from 'fs'
import path from 'path'

interface TestStats {
  expected?: number
  unexpected?: number
  flaky?: number
  skipped?: number
  finished?: number
}

interface TestResult {
  status: string
  duration?: number
  error?: {
    message?: string
    stack?: string
  }
  attachments?: Array<{
    name: string
    path?: string
    contentType: string
  }>
}

interface TestCase {
  title: string
  ok: boolean
  outcome: string
  results: TestResult[]
}

interface Suite {
  title: string
  file: string
  suites?: Suite[]
  tests?: TestCase[]
}

interface FullReportData {
  stats?: TestStats
  suites?: Suite[]
}

interface ReportData {
  stats?: TestStats
}

interface FailedTest {
  name: string
  file: string
  traceUrl?: string
  error?: string
}

interface TestCounts {
  passed: number
  failed: number
  flaky: number
  skipped: number
  total: number
  failures?: FailedTest[]
}

/**
 * Extract failed test details from Playwright report
 */
function extractFailedTests(
  reportData: FullReportData,
  baseUrl?: string
): FailedTest[] {
  const failures: FailedTest[] = []

  function processTest(test: TestCase, file: string, suitePath: string[]) {
    // Check if test failed or is flaky
    const hasFailed = test.results.some(
      (r) => r.status === 'failed' || r.status === 'timedOut'
    )
    const isFlaky = test.outcome === 'flaky'

    if (hasFailed || isFlaky) {
      const fullTestName = [...suitePath, test.title]
        .filter(Boolean)
        .join(' › ')
      const failedResult = test.results.find(
        (r) => r.status === 'failed' || r.status === 'timedOut'
      )

      // Find trace attachment
      let traceUrl: string | undefined
      if (failedResult?.attachments) {
        const traceAttachment = failedResult.attachments.find(
          (a) => a.name === 'trace' && a.contentType === 'application/zip'
        )
        if (traceAttachment?.path) {
          // Convert local path to URL path
          const tracePath = traceAttachment.path.replace(/\\/g, '/')
          const traceFile = path.basename(tracePath)
          if (baseUrl) {
            // Construct trace viewer URL
            const traceDataUrl = `${baseUrl}/data/${traceFile}`
            traceUrl = `${baseUrl}/trace/?trace=${encodeURIComponent(traceDataUrl)}`
          }
        }
      }

      failures.push({
        name: fullTestName,
        file: file,
        traceUrl,
        error: failedResult?.error?.message
      })
    }
  }

  function processSuite(suite: Suite, parentPath: string[] = []) {
    const suitePath = suite.title ? [...parentPath, suite.title] : parentPath

    // Process tests in this suite
    if (suite.tests) {
      for (const test of suite.tests) {
        processTest(test, suite.file, suitePath)
      }
    }

    // Recursively process nested suites
    if (suite.suites) {
      for (const childSuite of suite.suites) {
        processSuite(childSuite, suitePath)
      }
    }
  }

  if (reportData.suites) {
    for (const suite of reportData.suites) {
      processSuite(suite)
    }
  }

  return failures
}

/**
 * Extract test counts from Playwright HTML report
 * @param reportDir - Path to the playwright-report directory
 * @param baseUrl - Base URL of the deployed report (for trace links)
 * @returns Test counts { passed, failed, flaky, skipped, total, failures }
 */
function extractTestCounts(reportDir: string, baseUrl?: string): TestCounts {
  const counts: TestCounts = {
    passed: 0,
    failed: 0,
    flaky: 0,
    skipped: 0,
    total: 0,
    failures: []
  }

  try {
    // First, try to find report.json which Playwright generates with JSON reporter
    const jsonReportFile = path.join(reportDir, 'report.json')
    if (fs.existsSync(jsonReportFile)) {
      const reportJson: FullReportData = JSON.parse(
        fs.readFileSync(jsonReportFile, 'utf-8')
      )
      if (reportJson.stats) {
        const stats = reportJson.stats
        counts.total = stats.expected || 0
        counts.passed =
          (stats.expected || 0) -
          (stats.unexpected || 0) -
          (stats.flaky || 0) -
          (stats.skipped || 0)
        counts.failed = stats.unexpected || 0
        counts.flaky = stats.flaky || 0
        counts.skipped = stats.skipped || 0

        // Extract detailed failure information
        if (counts.failed > 0 || counts.flaky > 0) {
          counts.failures = extractFailedTests(reportJson, baseUrl)
        }

        return counts
      }
    }

    // Try index.html - Playwright HTML report embeds data in a script tag
    const indexFile = path.join(reportDir, 'index.html')
    if (fs.existsSync(indexFile)) {
      const content = fs.readFileSync(indexFile, 'utf-8')

      // Look for the embedded report data in various formats
      // Format 1: window.playwrightReportBase64
      let dataMatch = content.match(
        /window\.playwrightReportBase64\s*=\s*["']([^"']+)["']/
      )
      if (dataMatch) {
        try {
          const decodedData = Buffer.from(dataMatch[1], 'base64').toString(
            'utf-8'
          )
          const reportData: ReportData = JSON.parse(decodedData)

          if (reportData.stats) {
            const stats = reportData.stats
            counts.total = stats.expected || 0
            counts.passed =
              (stats.expected || 0) -
              (stats.unexpected || 0) -
              (stats.flaky || 0) -
              (stats.skipped || 0)
            counts.failed = stats.unexpected || 0
            counts.flaky = stats.flaky || 0
            counts.skipped = stats.skipped || 0
            return counts
          }
        } catch (e) {
          // Continue to try other formats
        }
      }

      // Format 2: window.playwrightReport
      dataMatch = content.match(/window\.playwrightReport\s*=\s*({[\s\S]*?});/)
      if (dataMatch) {
        try {
          // Use Function constructor instead of eval for safety
          const reportData = new Function(
            'return ' + dataMatch[1]
          )() as ReportData

          if (reportData.stats) {
            const stats = reportData.stats
            counts.total = stats.expected || 0
            counts.passed =
              (stats.expected || 0) -
              (stats.unexpected || 0) -
              (stats.flaky || 0) -
              (stats.skipped || 0)
            counts.failed = stats.unexpected || 0
            counts.flaky = stats.flaky || 0
            counts.skipped = stats.skipped || 0
            return counts
          }
        } catch (e) {
          // Continue to try other formats
        }
      }

      // Format 3: Look for stats in the HTML content directly
      // Playwright sometimes renders stats in the UI
      const statsMatch = content.match(
        /(\d+)\s+passed[^0-9]*(\d+)\s+failed[^0-9]*(\d+)\s+flaky[^0-9]*(\d+)\s+skipped/i
      )
      if (statsMatch) {
        counts.passed = parseInt(statsMatch[1]) || 0
        counts.failed = parseInt(statsMatch[2]) || 0
        counts.flaky = parseInt(statsMatch[3]) || 0
        counts.skipped = parseInt(statsMatch[4]) || 0
        counts.total =
          counts.passed + counts.failed + counts.flaky + counts.skipped
        return counts
      }

      // Format 4: Try to extract from summary text patterns
      const passedMatch = content.match(/(\d+)\s+(?:tests?|specs?)\s+passed/i)
      const failedMatch = content.match(/(\d+)\s+(?:tests?|specs?)\s+failed/i)
      const flakyMatch = content.match(/(\d+)\s+(?:tests?|specs?)\s+flaky/i)
      const skippedMatch = content.match(/(\d+)\s+(?:tests?|specs?)\s+skipped/i)
      const totalMatch = content.match(
        /(\d+)\s+(?:tests?|specs?)\s+(?:total|ran)/i
      )

      if (passedMatch) counts.passed = parseInt(passedMatch[1]) || 0
      if (failedMatch) counts.failed = parseInt(failedMatch[1]) || 0
      if (flakyMatch) counts.flaky = parseInt(flakyMatch[1]) || 0
      if (skippedMatch) counts.skipped = parseInt(skippedMatch[1]) || 0
      if (totalMatch) {
        counts.total = parseInt(totalMatch[1]) || 0
      } else if (
        counts.passed ||
        counts.failed ||
        counts.flaky ||
        counts.skipped
      ) {
        counts.total =
          counts.passed + counts.failed + counts.flaky + counts.skipped
      }
    }
  } catch (error) {
    console.error(`Error reading report from ${reportDir}:`, error)
  }

  return counts
}

// Main execution
const reportDir = process.argv[2]
const baseUrl = process.argv[3] // Optional: base URL for trace links

if (!reportDir) {
  console.error(
    'Usage: extract-playwright-counts.ts <report-directory> [base-url]'
  )
  process.exit(1)
}

const counts = extractTestCounts(reportDir, baseUrl)

// Output as JSON for easy parsing in shell script
process.stdout.write(JSON.stringify(counts) + '\n')

export { extractTestCounts, extractFailedTests }
