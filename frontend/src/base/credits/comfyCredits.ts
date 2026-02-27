const DEFAULT_NUMBER_FORMAT: Intl.NumberFormatOptions = {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
}

const formatNumber = ({
  value,
  locale,
  options
}: {
  value: number
  locale?: string
  options?: Intl.NumberFormatOptions
}): string => {
  const merged: Intl.NumberFormatOptions = {
    ...DEFAULT_NUMBER_FORMAT,
    ...options
  }

  if (
    typeof merged.maximumFractionDigits === 'number' &&
    typeof merged.minimumFractionDigits === 'number' &&
    merged.maximumFractionDigits < merged.minimumFractionDigits
  ) {
    merged.minimumFractionDigits = merged.maximumFractionDigits
  }

  return new Intl.NumberFormat(locale, merged).format(value)
}

export const CREDITS_PER_USD = 211
export const COMFY_CREDIT_RATE_CENTS = CREDITS_PER_USD / 100 // credits per cent

export const usdToCents = (usd: number): number => Math.round(usd * 100)

export const centsToCredits = (cents: number): number =>
  Math.round(cents * COMFY_CREDIT_RATE_CENTS)

export const creditsToCents = (credits: number): number =>
  Math.round(credits / COMFY_CREDIT_RATE_CENTS)

export const usdToCredits = (usd: number): number =>
  Math.round(usd * CREDITS_PER_USD)

export const creditsToUsd = (credits: number): number =>
  Math.round((credits / CREDITS_PER_USD) * 100) / 100

export type FormatOptions = {
  value: number
  locale?: string
  numberOptions?: Intl.NumberFormatOptions
}

export type FormatFromCentsOptions = {
  cents: number
  locale?: string
  numberOptions?: Intl.NumberFormatOptions
}

export type FormatFromUsdOptions = {
  usd: number
  locale?: string
  numberOptions?: Intl.NumberFormatOptions
}

export const formatCredits = ({
  value,
  locale,
  numberOptions
}: FormatOptions): string =>
  formatNumber({ value, locale, options: numberOptions })

export const formatCreditsFromCents = ({
  cents,
  locale,
  numberOptions
}: FormatFromCentsOptions): string =>
  formatCredits({
    value: centsToCredits(cents),
    locale,
    numberOptions
  })

export const formatCreditsFromUsd = ({
  usd,
  locale,
  numberOptions
}: FormatFromUsdOptions): string =>
  formatCredits({
    value: usdToCredits(usd),
    locale,
    numberOptions
  })

export const formatUsd = ({
  value,
  locale,
  numberOptions
}: FormatOptions): string =>
  formatNumber({
    value,
    locale,
    options: numberOptions
  })

export const formatUsdFromCents = ({
  cents,
  locale,
  numberOptions
}: FormatFromCentsOptions): string =>
  formatUsd({
    value: cents / 100,
    locale,
    numberOptions
  })

/**
 * Clamps a USD value to the allowed range for credit purchases
 * @param value - The USD amount to clamp
 * @returns The clamped value between $1 and $1000, or 0 if NaN
 */
export const clampUsd = (value: number): number => {
  if (Number.isNaN(value)) return 0
  return Math.min(1000, Math.max(1, value))
}
