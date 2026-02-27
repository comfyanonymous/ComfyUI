import type { TaskItemImpl } from '@/stores/queueStore'
import type { JobState } from '@/types/queue'
import { formatDuration } from '@/utils/formatUtil'
import { clampPercentInt, formatPercent0 } from '@/utils/numberUtil'

export type BuildJobDisplayCtx = {
  t: (k: string, v?: Record<string, unknown>) => string
  locale: string
  formatClockTimeFn: (ts: number, locale: string) => string
  isActive: boolean
  totalPercent?: number
  currentNodePercent?: number
  currentNodeName?: string
  showAddedHint?: boolean
  /** Whether the app is running in cloud distribution */
  isCloud?: boolean
}

type JobDisplay = {
  iconName: string
  iconImageUrl?: string
  primary: string
  secondary: string
  showClear: boolean
}

export const iconForJobState = (state: JobState): string => {
  switch (state) {
    case 'pending':
      return 'icon-[lucide--loader-circle]'
    case 'initialization':
      return 'icon-[lucide--server-crash]'
    case 'running':
      return 'icon-[lucide--zap]'
    case 'completed':
      return 'icon-[lucide--check-check]'
    case 'failed':
      return 'icon-[lucide--alert-circle]'
    default:
      return 'icon-[lucide--circle]'
  }
}

const buildTitle = (task: TaskItemImpl, t: (k: string) => string): string => {
  const prefix = t('g.job')
  const shortId = String(task.jobId ?? '').split('-')[0]
  const idx = task.job.priority
  if (typeof idx === 'number') return `${prefix} #${idx}`
  if (shortId) return `${prefix} ${shortId}`
  return prefix
}

const buildQueuedTime = (
  task: TaskItemImpl,
  locale: string,
  formatClockTimeFn: (ts: number, locale: string) => string
): string => {
  const ts = task.createTime
  return ts !== undefined ? formatClockTimeFn(ts, locale) : ''
}

export const buildJobDisplay = (
  task: TaskItemImpl,
  state: JobState,
  ctx: BuildJobDisplayCtx
): JobDisplay => {
  if (state === 'pending') {
    if (ctx.showAddedHint) {
      return {
        iconName: 'icon-[lucide--check]',
        primary: ctx.t('queue.jobAddedToQueue'),
        secondary: buildQueuedTime(task, ctx.locale, ctx.formatClockTimeFn),
        showClear: true
      }
    }
    return {
      iconName: iconForJobState(state),
      primary: ctx.t('queue.inQueue'),
      secondary: buildQueuedTime(task, ctx.locale, ctx.formatClockTimeFn),
      showClear: true
    }
  }
  if (state === 'initialization') {
    return {
      iconName: iconForJobState(state),
      primary: ctx.t('queue.initializingAlmostReady'),
      secondary: buildQueuedTime(task, ctx.locale, ctx.formatClockTimeFn),
      showClear: true
    }
  }
  if (state === 'running') {
    if (ctx.isActive) {
      const total = formatPercent0(
        ctx.locale,
        clampPercentInt(ctx.totalPercent)
      )
      const curr = formatPercent0(
        ctx.locale,
        clampPercentInt(ctx.currentNodePercent)
      )
      const primary = ctx.t('sideToolbar.queueProgressOverlay.total', {
        percent: total
      })
      const right = ctx.currentNodeName
        ? `${ctx.currentNodeName} ${ctx.t(
            'sideToolbar.queueProgressOverlay.colonPercent',
            { percent: curr }
          )}`
        : ''
      return {
        iconName: iconForJobState(state),
        primary,
        secondary: right,
        showClear: true
      }
    }
    return {
      iconName: iconForJobState(state),
      primary: ctx.t('g.running'),
      secondary: '',
      showClear: true
    }
  }
  if (state === 'completed') {
    const time = task.executionTimeInSeconds
    const preview = task.previewOutput
    const iconImageUrl = preview && preview.isImage ? preview.url : undefined

    // Cloud shows "Completed in Xh Ym Zs", non-cloud shows filename
    const primary = ctx.isCloud
      ? ctx.t('queue.completedIn', {
          duration: formatDuration(task.executionTime ?? 0)
        })
      : preview?.filename && preview.filename.length
        ? preview.filename
        : buildTitle(task, ctx.t)

    return {
      iconName: iconForJobState(state),
      iconImageUrl,
      primary,
      secondary: time !== undefined ? `${time.toFixed(2)}s` : '',
      showClear: false
    }
  }
  if (state === 'failed') {
    return {
      iconName: iconForJobState(state),
      primary: ctx.t('g.failed'),
      secondary: ctx.t('g.failed'),
      showClear: true
    }
  }
  return {
    iconName: iconForJobState(state),
    primary: buildTitle(task, ctx.t),
    secondary: '',
    showClear: true
  }
}
