<template>
  <div class="relative">
    <div class="mb-4">
      <Button
        class="text-base-foreground w-full border-0 bg-secondary-background hover:bg-secondary-background-hover"
        :disabled="isRecording || readonly"
        @click="handleStartRecording"
      >
        {{ t('g.startRecording', 'Start Recording') }}
        <i-lucide:mic class="ml-1" />
      </Button>
    </div>
    <div
      v-if="isRecording || isPlaying || recordedURL"
      class="flex h-14 w-full min-w-0 items-center gap-2 rounded-lg px-3 bg-node-component-surface text-text-secondary"
    >
      <!-- Recording Status -->
      <div class="flex shrink-0 items-center gap-1">
        <span class="text-xs">
          {{
            isRecording
              ? t('g.listening', 'Listening...')
              : isPlaying
                ? t('g.playing', 'Playing...')
                : recordedURL
                  ? t('g.ready', 'Ready')
                  : ''
          }}
        </span>
        <span class="text-sm">{{ formatTime(timer) }}</span>
      </div>

      <!-- Waveform Visualization -->
      <div class="flex h-8 min-w-0 flex-1 items-center gap-2 overflow-hidden">
        <div
          v-for="(bar, index) in waveformBars"
          :key="index"
          class="max-h-8 min-h-1 w-0.75 rounded-[1.5px] bg-slate-100 transition-all duration-100"
          :style="{ height: bar.height + 'px' }"
          :title="`Bar ${index + 1}: ${bar.height}px`"
        />
      </div>

      <!-- Control Button -->
      <button
        v-if="isRecording"
        :title="t('g.stopRecording', 'Stop Recording')"
        class="flex shrink-0 size-8 animate-pulse items-center justify-center rounded-full border-0 bg-smoke-500/33 transition-colors"
        @click="handleStopRecording"
      >
        <div class="size-2.5 rounded-sm bg-danger-100" />
      </button>

      <button
        v-else-if="!isRecording && recordedURL && !isPlaying"
        :title="t('g.playRecording') || 'Play Recording'"
        class="flex shrink-0 size-8 items-center justify-center rounded-full border-0 bg-smoke-500/33 transition-colors"
        @click="handlePlayRecording"
      >
        <i class="text-text-secondary icon-[lucide--play] size-4" />
      </button>

      <button
        v-else-if="isPlaying"
        :title="t('g.stopPlayback') || 'Stop Playback'"
        class="flex shrink-0 size-8 items-center justify-center rounded-full border-0 bg-smoke-500/33 transition-colors"
        @click="handleStopPlayback"
      >
        <i class="text-text-secondary icon-[lucide--square] size-4" />
      </button>
    </div>
    <audio
      v-if="recordedURL"
      ref="audioRef"
      :key="audioElementKey"
      :src="recordedURL"
      class="hidden"
      @ended="playback.onPlaybackEnded"
      @loadedmetadata="playback.onMetadataLoaded"
    />
  </div>
</template>

<script setup lang="ts">
import { useIntervalFn } from '@vueuse/core'
import { Button } from 'primevue'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { app } from '@/scripts/app'
import { useAudioService } from '@/services/audioService'

import { useAudioPlayback } from '../composables/audio/useAudioPlayback'
import { useAudioRecorder } from '../composables/audio/useAudioRecorder'
import { useAudioWaveform } from '../composables/audio/useAudioWaveform'
import { formatTime } from '../utils/audioUtils'

const { t } = useI18n()

const props = defineProps<{
  readonly?: boolean
  nodeId: string
}>()

// Audio element ref
const audioRef = ref<HTMLAudioElement>()

// Keep track of the last uploaded path as a backup
let lastUploadedPath = ''

// Composables
const recorder = useAudioRecorder({
  onRecordingComplete: handleRecordingComplete,
  onError: () => {
    useToastStore().addAlert(
      t('g.micPermissionDenied') || 'Microphone permission denied'
    )
  }
})

const waveform = useAudioWaveform({
  barCount: 18,
  minHeight: 4,
  maxHeight: 32
})

const playback = useAudioPlayback(audioRef, {
  onPlaybackEnded: handlePlaybackEnded,
  onMetadataLoaded: (duration) => {
    if (!isPlaying.value && !isRecording.value) {
      timer.value = Math.floor(duration)
    }
  }
})

// Timer for recording
const timer = ref(0)
const { pause: pauseTimer, resume: resumeTimer } = useIntervalFn(
  () => {
    timer.value += 1
  },
  1000,
  { immediate: false }
)

// Destructure for template access
const { isRecording, recordedURL } = recorder
const { waveformBars } = waveform
const { isPlaying, audioElementKey } = playback

// Computed for waveform animation
const isWaveformActive = computed(() => isRecording.value || isPlaying.value)

const modelValue = defineModel<string>({ default: '' })

const litegraphNode = computed(() => {
  if (!props.nodeId || !app.canvas.graph) return null
  return app.canvas.graph.getNodeById(props.nodeId) as LGraphNode | null
})

async function handleRecordingComplete(blob: Blob) {
  try {
    const path = await useAudioService().convertBlobToFileAndSubmit(blob)
    modelValue.value = path
    lastUploadedPath = path
  } catch (e) {
    useToastStore().addAlert('Failed to upload recorded audio')
  }
}

async function handleStartRecording() {
  if (props.readonly) return

  try {
    await waveform.setupAudioContext()
    await recorder.startRecording()

    // Setup waveform visualization for recording
    if (recorder.mediaRecorder.value) {
      const stream = recorder.mediaRecorder.value.stream
      if (stream) {
        await waveform.setupRecordingVisualization(stream)
      }
    }

    // Start timer
    timer.value = 0
    resumeTimer()
    waveform.initWaveform()
    waveform.updateWaveform(isWaveformActive)
  } catch (err) {
    console.error('Failed to start recording:', err)
  }
}

function handleStopRecording() {
  recorder.stopRecording()
  pauseTimer()
  waveform.stopWaveform()
}

async function handlePlayRecording() {
  if (!recordedURL.value) return

  // Reset timer
  timer.value = 0

  // Reset and setup audio element
  await playback.resetAudioElement()

  // Wait for audio element to be ready
  await new Promise((resolve) => setTimeout(resolve, 50))

  if (!audioRef.value) return

  // Setup waveform visualization for playback
  const setupSuccess = await waveform.setupPlaybackVisualization(audioRef.value)
  if (!setupSuccess) return

  // Start playback
  await playback.play()

  // Update waveform
  waveform.initWaveform()
  waveform.updateWaveform(isWaveformActive)

  // Update timer from audio current time
  const timerInterval = setInterval(() => {
    timer.value = Math.floor(playback.getCurrentTime())
  }, 100)

  // Store interval for cleanup
  playback.playbackTimerInterval.value = timerInterval
}

function handleStopPlayback() {
  playback.stop()
  handlePlaybackEnded()
}

function handlePlaybackEnded() {
  waveform.stopWaveform()

  // Clear playback timer interval
  if (playback.playbackTimerInterval.value !== null) {
    clearInterval(playback.playbackTimerInterval.value)
    playback.playbackTimerInterval.value = null
  }

  const duration = playback.getDuration()
  if (duration) {
    timer.value = Math.floor(duration)
  } else {
    timer.value = 0
  }
}

// Serialization function for workflow execution
async function serializeValue() {
  if (isRecording.value && recorder.mediaRecorder.value) {
    recorder.mediaRecorder.value.stop()

    await new Promise((resolve, reject) => {
      let attempts = 0
      const maxAttempts = 50 // 5 seconds max (50 * 100ms)
      const checkRecording = () => {
        if (!isRecording.value && modelValue.value) {
          resolve(undefined)
        } else if (++attempts >= maxAttempts) {
          reject(new Error('Recording serialization timeout after 5 seconds'))
        } else {
          setTimeout(checkRecording, 100)
        }
      }
      checkRecording()
    })
  }

  return modelValue.value || lastUploadedPath || ''
}

function registerWidgetSerialization() {
  const node = litegraphNode.value
  if (!node?.widgets) return
  const targetWidget = node.widgets.find((w: IBaseWidget) => w.name === 'audio')
  if (targetWidget) {
    targetWidget.serializeValue = serializeValue
  }
}

onMounted(() => {
  waveform.initWaveform()
  registerWidgetSerialization()
})

onUnmounted(() => {
  if (playback.playbackTimerInterval.value !== null) {
    clearInterval(playback.playbackTimerInterval.value)
    playback.playbackTimerInterval.value = null
  }
})
</script>
