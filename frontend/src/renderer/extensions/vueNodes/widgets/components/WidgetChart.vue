<template>
  <div class="flex flex-col gap-1">
    <div class="max-h-[48rem] rounded border p-4">
      <Chart
        :type="chartType"
        :data="chartData"
        :options="chartOptions"
        :aria-label="`${widget.name || $t('g.chart')} - ${chartType} ${$t('g.chartLowercase')}`"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ChartData } from 'chart.js'
import Chart from 'primevue/chart'
import { computed } from 'vue'

import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'
import type { ChartInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

type ChartWidgetOptions = NonNullable<ChartInputSpec['options']> &
  IWidgetOptions

const value = defineModel<ChartData>({ required: true })

const props = defineProps<{
  widget: SimplifiedWidget<ChartData, ChartWidgetOptions>
}>()

const chartType = computed(() => props.widget.options?.type ?? 'line')

const chartData = computed(() => value.value || { labels: [], datasets: [] })

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: '#FFF',
        usePointStyle: true,
        pointStyle: 'circle'
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: '#9FA2BD'
      },
      grid: {
        display: true,
        color: '#9FA2BD',
        drawTicks: false,
        drawOnChartArea: true,
        drawBorder: false
      },
      border: {
        display: true,
        color: '#9FA2BD'
      }
    },
    y: {
      ticks: {
        color: '#9FA2BD'
      },
      grid: {
        display: false,
        drawTicks: false,
        drawOnChartArea: false,
        drawBorder: false
      },
      border: {
        display: true,
        color: '#9FA2BD'
      }
    }
  }
}))
</script>
