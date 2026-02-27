import { SerializeAddon } from '@xterm/addon-serialize'
import { Terminal } from '@xterm/xterm'
import { markRaw, onMounted, onUnmounted } from 'vue'

export function useTerminalBuffer() {
  const serializeAddon = new SerializeAddon()
  const terminal = markRaw(new Terminal({ convertEol: true }))

  const copyTo = (destinationTerminal: Terminal) => {
    destinationTerminal.write(serializeAddon.serialize())
  }

  const write = (message: string) => terminal.write(message)

  const serialize = () => serializeAddon.serialize()

  onMounted(() => {
    terminal.loadAddon(serializeAddon)
  })

  onUnmounted(() => {
    terminal.dispose()
  })

  return {
    copyTo,
    serialize,
    write
  }
}
