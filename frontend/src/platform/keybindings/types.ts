import { z } from 'zod'

const zKeyCombo = z.object({
  key: z.string(),
  ctrl: z.boolean().optional(),
  alt: z.boolean().optional(),
  shift: z.boolean().optional(),
  meta: z.boolean().optional()
})

export const zKeybinding = z.object({
  commandId: z.string(),
  combo: zKeyCombo,
  targetElementId: z.string().optional()
})

export type KeyCombo = z.infer<typeof zKeyCombo>
export type Keybinding = z.infer<typeof zKeybinding>
