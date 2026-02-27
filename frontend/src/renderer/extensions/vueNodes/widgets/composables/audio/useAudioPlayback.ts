import { nextTick, ref } from 'vue'
import type { Ref } from 'vue'

interface AudioPlaybackOptions {
  onPlaybackEnded?: () => void
  onMetadataLoaded?: (duration: number) => void
}

export function useAudioPlayback(
  audioRef: Ref<HTMLAudioElement | undefined>,
  options: AudioPlaybackOptions = {}
) {
  const isPlaying = ref(false)
  const audioElementKey = ref(0)
  const playbackTimerInterval = ref<ReturnType<typeof setInterval> | null>(null)

  async function play() {
    if (!audioRef.value) return false

    try {
      await audioRef.value.play()
      isPlaying.value = true
      return true
    } catch (error) {
      console.warn('Audio playback failed:', error)
      isPlaying.value = false
      return false
    }
  }

  function stop() {
    if (audioRef.value) {
      audioRef.value.pause()
      audioRef.value.currentTime = 0
    }
    isPlaying.value = false
    if (options.onPlaybackEnded) {
      options.onPlaybackEnded()
    }
  }

  function onPlaybackEnded() {
    isPlaying.value = false
    if (options.onPlaybackEnded) {
      options.onPlaybackEnded()
    }
  }

  function onMetadataLoaded() {
    if (audioRef.value?.duration && options.onMetadataLoaded) {
      options.onMetadataLoaded(audioRef.value.duration)
    }
  }

  async function resetAudioElement() {
    audioElementKey.value += 1
    await nextTick()
  }

  function getCurrentTime() {
    return audioRef.value?.currentTime || 0
  }

  function getDuration() {
    return audioRef.value?.duration || 0
  }

  return {
    isPlaying,
    audioElementKey,
    play,
    stop,
    onPlaybackEnded,
    onMetadataLoaded,
    resetAudioElement,
    getCurrentTime,
    getDuration,
    playbackTimerInterval
  }
}
