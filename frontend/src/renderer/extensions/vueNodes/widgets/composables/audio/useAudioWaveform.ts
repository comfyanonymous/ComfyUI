import { onUnmounted, ref } from 'vue'
import type { Ref } from 'vue'

interface WaveformBar {
  height: number
}

interface AudioWaveformOptions {
  barCount?: number
  minHeight?: number
  maxHeight?: number
}

export function useAudioWaveform(options: AudioWaveformOptions = {}) {
  const { barCount = 18, minHeight = 4, maxHeight = 32 } = options

  const waveformBars = ref<WaveformBar[]>(
    Array.from({ length: barCount }, () => ({ height: 16 }))
  )
  const audioContext = ref<AudioContext | null>(null)
  const analyser = ref<AnalyserNode | null>(null)
  const dataArray = ref<Uint8Array | null>(null)
  const animationId = ref<number | null>(null)
  const mediaElementSource = ref<MediaElementAudioSourceNode | null>(null)

  function initWaveform() {
    waveformBars.value = Array.from({ length: barCount }, () => ({
      height: Math.random() * (maxHeight - minHeight) + minHeight
    }))
  }

  function updateWaveform(isActive: Ref<boolean>) {
    if (!isActive.value) return

    if (analyser.value && dataArray.value) {
      updateWaveformFromAudio()
    } else {
      updateWaveformRandom()
    }

    animationId.value = requestAnimationFrame(() => updateWaveform(isActive))
  }

  function updateWaveformFromAudio() {
    if (!analyser.value || !dataArray.value) return

    analyser.value.getByteFrequencyData(
      dataArray.value as Uint8Array<ArrayBuffer>
    )
    const samplesPerBar = Math.floor(dataArray.value.length / barCount)

    waveformBars.value = waveformBars.value.map((_, i) => {
      let sum = 0
      for (let j = 0; j < samplesPerBar; j++) {
        sum += dataArray.value![i * samplesPerBar + j] || 0
      }
      const average = sum / samplesPerBar
      const normalizedHeight =
        (average / 255) * (maxHeight - minHeight) + minHeight
      return { height: normalizedHeight }
    })
  }

  function updateWaveformRandom() {
    waveformBars.value = waveformBars.value.map((bar) => ({
      height: Math.max(
        minHeight,
        Math.min(maxHeight, bar.height + (Math.random() - 0.5) * 4)
      )
    }))
  }

  async function setupAudioContext() {
    if (audioContext.value && audioContext.value.state !== 'closed') {
      await audioContext.value.close()
    }
    audioContext.value = null
    mediaElementSource.value = null
  }

  async function setupRecordingVisualization(stream: MediaStream) {
    audioContext.value = new window.AudioContext()
    analyser.value = audioContext.value.createAnalyser()
    const source = audioContext.value.createMediaStreamSource(stream)
    source.connect(analyser.value)

    analyser.value.fftSize = 256
    dataArray.value = new Uint8Array(analyser.value.frequencyBinCount)
  }

  async function setupPlaybackVisualization(audioElement: HTMLAudioElement) {
    if (audioContext.value && audioContext.value.state !== 'closed') {
      await audioContext.value.close()
    }

    mediaElementSource.value = null

    if (!audioElement) return false

    audioContext.value = new window.AudioContext()
    analyser.value = audioContext.value.createAnalyser()

    mediaElementSource.value =
      audioContext.value.createMediaElementSource(audioElement)

    mediaElementSource.value.connect(analyser.value)
    analyser.value.connect(audioContext.value.destination)

    analyser.value.fftSize = 256
    dataArray.value = new Uint8Array(analyser.value.frequencyBinCount)

    return true
  }

  function stopWaveform() {
    if (animationId.value) {
      cancelAnimationFrame(animationId.value)
      animationId.value = null
    }
  }

  function dispose() {
    stopWaveform()
    if (audioContext.value && audioContext.value.state !== 'closed') {
      void audioContext.value.close()
    }
    audioContext.value = null
    mediaElementSource.value = null
  }

  onUnmounted(() => {
    dispose()
  })

  return {
    waveformBars,
    initWaveform,
    updateWaveform,
    setupAudioContext,
    setupRecordingVisualization,
    setupPlaybackVisualization,
    stopWaveform,
    dispose
  }
}
