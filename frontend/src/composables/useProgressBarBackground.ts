type ProgressPercent = number | undefined

const progressBarContainerClass = 'absolute inset-0'
const progressBarBaseClass =
  'pointer-events-none absolute inset-y-0 left-0 h-full transition-[width]'
const progressBarPrimaryClass = `${progressBarBaseClass} bg-interface-panel-job-progress-primary`
const progressBarSecondaryClass = `${progressBarBaseClass} bg-interface-panel-job-progress-secondary`

function clampPercent(value: number) {
  return Math.min(100, Math.max(0, value))
}

function normalizeProgressPercent(value: ProgressPercent) {
  if (value === undefined || !Number.isFinite(value)) return undefined

  return clampPercent(value)
}

function hasProgressPercent(value: ProgressPercent) {
  return normalizeProgressPercent(value) !== undefined
}

function hasAnyProgressPercent(
  totalPercent: ProgressPercent,
  currentPercent: ProgressPercent
) {
  return hasProgressPercent(totalPercent) || hasProgressPercent(currentPercent)
}

function progressPercentStyle(value: ProgressPercent) {
  const normalized = normalizeProgressPercent(value)

  if (normalized === undefined) return undefined

  return { width: `${normalized}%` }
}

export function useProgressBarBackground() {
  return {
    progressBarContainerClass,
    progressBarPrimaryClass,
    progressBarSecondaryClass,
    hasProgressPercent,
    hasAnyProgressPercent,
    progressPercentStyle
  }
}
