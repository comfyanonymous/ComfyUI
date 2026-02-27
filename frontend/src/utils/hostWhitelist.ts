/**
 * Whitelisting helper for enabling SSO on safe, local-only hosts.
 *
 * Built-ins (always allowed):
 *   • 'localhost' and any subdomain of '.localhost' (e.g., app.localhost)
 *   • IPv4 loopback 127.0.0.0/8 (e.g., 127.0.0.1, 127.1.2.3)
 *   • IPv6 loopback ::1 (supports compressed/expanded textual forms)
 *
 * No environment variables are used. To add more exact hostnames,
 * edit HOST_WHITELIST below.
 */

const HOST_WHITELIST: string[] = ['localhost']

/** Normalize for comparison: lowercase, strip port/brackets, trim trailing dot. */
export function normalizeHost(input: string): string {
  let h = (input || '').trim().toLowerCase()

  // Trim a trailing dot: 'localhost.' -> 'localhost'
  h = h.replace(/\.$/, '')

  // Remove ':port' safely.
  // Case 1: [IPv6]:port
  const mBracket = h.match(/^\[([^\]]+)\]:(\d+)$/)
  if (mBracket) {
    h = mBracket[1] // keep only the host inside the brackets
  } else {
    // Case 2: hostname/IPv4:port (exactly one ':')
    const mPort = h.match(/^([^:]+):(\d+)$/)
    if (mPort) h = mPort[1]
  }

  // Strip any remaining brackets (e.g., '[::1]' -> '::1')
  h = h.replace(/^\[|\]$/g, '')

  return h
}

/** Public check used by the UI. */
export function isHostWhitelisted(rawHost: string): boolean {
  const host = normalizeHost(rawHost)
  if (isLocalhostLabel(host)) return true
  if (isIPv4Loopback(host)) return true
  if (isIPv6Loopback(host)) return true
  if (isComfyOrgHost(host)) return true
  const normalizedList = HOST_WHITELIST.map(normalizeHost)
  return normalizedList.includes(host)
}

/* -------------------- Helpers -------------------- */

function isLocalhostLabel(h: string): boolean {
  // 'localhost' and any subdomain (e.g., 'app.localhost')
  return h === 'localhost' || h.endsWith('.localhost')
}

const IPV4_OCTET = '(?:25[0-5]|2[0-4]\\d|1\\d\\d|0?\\d?\\d)'
const V4_LOOPBACK_RE = new RegExp(
  '^127\\.' + IPV4_OCTET + '\\.' + IPV4_OCTET + '\\.' + IPV4_OCTET + '$'
)

function isIPv4Loopback(h: string): boolean {
  // 127/8 with strict 0–255 octets (leading zeros allowed, e.g., 127.000.000.001)
  return V4_LOOPBACK_RE.test(h)
}

// Fully expanded IPv6 loopback: 0:0:0:0:0:0:0:1 (allow leading zeros up to 4 chars)
const V6_FULL_LOOPBACK_RE = /^(?:0{1,4}:){7}0{0,3}1$/i

// Compressed IPv6 loopback forms around '::' with only zero groups before the final :1
// - Left side: zero groups separated by ':' (no trailing colon required)
// - Right side: zero groups each followed by ':' (so the final ':1' is provided by the pattern)
// The final group is exactly value 1, with up to 3 leading zeros (e.g., '0001').
const V6_COMPRESSED_LOOPBACK_RE =
  /^((?:0{1,4}(?::0{1,4}){0,6})?)::((?:0{1,4}:){0,6})0{0,3}1$/i

function isIPv6Loopback(h: string): boolean {
  // Exact full form: 0:0:0:0:0:0:0:1 (with up to 3 leading zeros on the final "1" group)
  if (V6_FULL_LOOPBACK_RE.test(h)) return true

  // Compressed forms that still equal ::1 (e.g., ::1, ::0001, 0:0::1, ::0:1, etc.)
  const m = h.match(V6_COMPRESSED_LOOPBACK_RE)
  if (!m) return false

  // Count explicit zero groups on each side of '::' to ensure at least one group is compressed.
  // (leftCount + rightCount) must be ≤ 6 so that the total expanded groups = 8.
  const leftCount = m[1] ? (m[1].match(/0{1,4}:/gi)?.length ?? 0) : 0
  const rightCount = m[2] ? (m[2].match(/0{1,4}:/gi)?.length ?? 0) : 0

  // Require that at least one group was actually compressed: i.e., leftCount + rightCount ≤ 6.
  return leftCount + rightCount <= 6
}

const COMFY_ORG_HOST = /\.comfy\.org$/

function isComfyOrgHost(h: string): boolean {
  return COMFY_ORG_HOST.test(h)
}
