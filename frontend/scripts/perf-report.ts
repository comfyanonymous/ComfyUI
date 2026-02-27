import { existsSync, readFileSync } from 'node:fs'

interface PerfMeasurement {
  name: string
  durationMs: number
  styleRecalcs: number
  styleRecalcDurationMs: number
  layouts: number
  layoutDurationMs: number
  taskDurationMs: number
  heapDeltaBytes: number
}

interface PerfReport {
  timestamp: string
  gitSha: string
  branch: string
  measurements: PerfMeasurement[]
}

const CURRENT_PATH = 'test-results/perf-metrics.json'
const BASELINE_PATH = 'temp/perf-baseline/perf-metrics.json'

function formatDelta(pct: number): string {
  if (pct >= 20) return `+${pct.toFixed(0)}% 🔴`
  if (pct >= 10) return `+${pct.toFixed(0)}% 🟠`
  if (pct > -10) return `${pct >= 0 ? '+' : ''}${pct.toFixed(0)}% ⚪`
  return `${pct.toFixed(0)}% 🟢`
}

function formatBytes(bytes: number): string {
  if (Math.abs(bytes) < 1024) return `${bytes} B`
  if (Math.abs(bytes) < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function calcDelta(
  baseline: number,
  current: number
): { pct: number; isNew: boolean } {
  if (baseline > 0) {
    return { pct: ((current - baseline) / baseline) * 100, isNew: false }
  }
  return current > 0 ? { pct: Infinity, isNew: true } : { pct: 0, isNew: false }
}

function formatDeltaCell(delta: { pct: number; isNew: boolean }): string {
  return delta.isNew ? 'new 🔴' : formatDelta(delta.pct)
}

function main() {
  if (!existsSync(CURRENT_PATH)) {
    process.stdout.write(
      '## ⚡ Performance Report\n\nNo perf metrics found. Perf tests may not have run.\n'
    )
    process.exit(0)
  }

  const current: PerfReport = JSON.parse(readFileSync(CURRENT_PATH, 'utf-8'))

  const baseline: PerfReport | null = existsSync(BASELINE_PATH)
    ? JSON.parse(readFileSync(BASELINE_PATH, 'utf-8'))
    : null

  const lines: string[] = []
  lines.push('## ⚡ Performance Report\n')

  if (baseline) {
    lines.push(
      '| Metric | Baseline | PR | Δ |',
      '|--------|----------|-----|---|'
    )

    for (const m of current.measurements) {
      const base = baseline.measurements.find((b) => b.name === m.name)
      if (!base) {
        lines.push(`| ${m.name}: style recalcs | — | ${m.styleRecalcs} | new |`)
        lines.push(`| ${m.name}: layouts | — | ${m.layouts} | new |`)
        lines.push(
          `| ${m.name}: task duration | — | ${m.taskDurationMs.toFixed(0)}ms | new |`
        )
        continue
      }

      const recalcDelta = calcDelta(base.styleRecalcs, m.styleRecalcs)
      lines.push(
        `| ${m.name}: style recalcs | ${base.styleRecalcs} | ${m.styleRecalcs} | ${formatDeltaCell(recalcDelta)} |`
      )

      const layoutDelta = calcDelta(base.layouts, m.layouts)
      lines.push(
        `| ${m.name}: layouts | ${base.layouts} | ${m.layouts} | ${formatDeltaCell(layoutDelta)} |`
      )

      const taskDelta = calcDelta(base.taskDurationMs, m.taskDurationMs)
      lines.push(
        `| ${m.name}: task duration | ${base.taskDurationMs.toFixed(0)}ms | ${m.taskDurationMs.toFixed(0)}ms | ${formatDeltaCell(taskDelta)} |`
      )
    }
  } else {
    lines.push(
      'No baseline found — showing absolute values.\n',
      '| Metric | Value |',
      '|--------|-------|'
    )
    for (const m of current.measurements) {
      lines.push(`| ${m.name}: style recalcs | ${m.styleRecalcs} |`)
      lines.push(`| ${m.name}: layouts | ${m.layouts} |`)
      lines.push(
        `| ${m.name}: task duration | ${m.taskDurationMs.toFixed(0)}ms |`
      )
      lines.push(`| ${m.name}: heap delta | ${formatBytes(m.heapDeltaBytes)} |`)
    }
  }

  lines.push('\n<details><summary>Raw data</summary>\n')
  lines.push('```json')
  lines.push(JSON.stringify(current, null, 2))
  lines.push('```')
  lines.push('\n</details>')

  process.stdout.write(lines.join('\n') + '\n')
}

main()
