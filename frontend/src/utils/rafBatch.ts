export function createRafBatch(run: () => void) {
  let rafId: number | null = null

  const schedule = () => {
    if (rafId != null) return
    rafId = requestAnimationFrame(() => {
      rafId = null
      run()
    })
  }

  const cancel = () => {
    if (rafId != null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
  }

  const flush = () => {
    if (rafId == null) return
    cancelAnimationFrame(rafId)
    rafId = null
    run()
  }

  const isScheduled = () => rafId != null

  return { schedule, cancel, flush, isScheduled }
}
