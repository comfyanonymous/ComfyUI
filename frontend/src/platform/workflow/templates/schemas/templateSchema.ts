import { z } from 'zod'

export const zLogoIndex = z.record(z.string(), z.string())

export type LogoIndex = z.infer<typeof zLogoIndex>
