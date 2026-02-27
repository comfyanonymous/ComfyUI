import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'

import type { PerfMeasurement } from '../fixtures/helpers/PerformanceHelper'

export interface PerfReport {
  timestamp: string
  gitSha: string
  branch: string
  measurements: PerfMeasurement[]
}

const TEMP_DIR = join('test-results', 'perf-temp')

export function recordMeasurement(m: PerfMeasurement) {
  mkdirSync(TEMP_DIR, { recursive: true })
  const filename = `${m.name}-${Date.now()}.json`
  writeFileSync(join(TEMP_DIR, filename), JSON.stringify(m))
}

export function writePerfReport(
  gitSha = process.env.GITHUB_SHA ?? 'local',
  branch = process.env.GITHUB_HEAD_REF ?? 'local'
) {
  if (!readdirSync('test-results', { withFileTypes: true }).length) return

  let tempFiles: string[]
  try {
    tempFiles = readdirSync(TEMP_DIR).filter((f) => f.endsWith('.json'))
  } catch {
    return
  }
  if (tempFiles.length === 0) return

  const measurements: PerfMeasurement[] = tempFiles.map((f) =>
    JSON.parse(readFileSync(join(TEMP_DIR, f), 'utf-8'))
  )

  const report: PerfReport = {
    timestamp: new Date().toISOString(),
    gitSha,
    branch,
    measurements
  }
  writeFileSync(
    join('test-results', 'perf-metrics.json'),
    JSON.stringify(report, null, 2)
  )
}
