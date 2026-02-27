import type { ISerialisedGraph } from '@/lib/litegraph/src/litegraph'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { SystemStats } from '@/schemas/apiSchema'

export interface ErrorReportData {
  exceptionType: string
  exceptionMessage: string
  systemStats: SystemStats
  serverLogs: string
  workflow: ISerialisedGraph

  traceback?: string
  nodeId?: NodeId
  nodeType?: string
}

/**
 * Generate a report for an error.
 * @param error - The error to report.
 * @returns The report.
 */
export function generateErrorReport(error: ErrorReportData): string {
  // The default JSON workflow has about 3000 characters.
  const MAX_JSON_LENGTH = 20000
  const workflowJSONString = JSON.stringify(error.workflow)
  const workflowText =
    workflowJSONString.length > MAX_JSON_LENGTH
      ? 'Workflow too large. Please manually upload the workflow from local file system.'
      : workflowJSONString
  const systemStats = error.systemStats

  return `
# ComfyUI Error Report
${
  error
    ? `## Error Details
- **Node ID:** ${error.nodeId || 'N/A'}
- **Node Type:** ${error.nodeType || 'N/A'}
- **Exception Type:** ${error.exceptionType || 'N/A'}
- **Exception Message:** ${error.exceptionMessage || 'N/A'}
## Stack Trace
\`\`\`
${error.traceback || 'No stack trace available'}
\`\`\``
    : ''
}
## System Information
- **ComfyUI Version:** ${systemStats.system.comfyui_version}
- **Arguments:** ${systemStats.system.argv.join(' ')}
- **OS:** ${systemStats.system.os}
- **Python Version:** ${systemStats.system.python_version}
- **Embedded Python:** ${systemStats.system.embedded_python}
- **PyTorch Version:** ${systemStats.system.pytorch_version}
## Devices
${systemStats.devices
  .map(
    (device) => `
- **Name:** ${device.name}
  - **Type:** ${device.type}
  - **VRAM Total:** ${device.vram_total}
  - **VRAM Free:** ${device.vram_free}
  - **Torch VRAM Total:** ${device.torch_vram_total}
  - **Torch VRAM Free:** ${device.torch_vram_free}
`
  )
  .join('\n')}
## Logs
\`\`\`
${typeof error.serverLogs === 'string' ? error.serverLogs : JSON.stringify(error.serverLogs, null, 2)}
\`\`\`
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
\`\`\`
${workflowText}
\`\`\`

## Additional Context
(Please add any additional context or steps to reproduce the error here)
`
}
