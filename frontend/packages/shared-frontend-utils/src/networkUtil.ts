import axios from 'axios'

const VALID_STATUS_CODES = [200, 201, 301, 302, 307, 308]
export const checkUrlReachable = async (url: string): Promise<boolean> => {
  try {
    const response = await axios.head(url)
    // Additional check for successful response
    return VALID_STATUS_CODES.includes(response.status)
  } catch {
    return false
  }
}

/**
 * Checks if the user is likely in mainland China by:
 * 1. Checking navigator.language
 * 2. Testing connectivity to commonly blocked services
 * 3. Testing latency to China-specific domains
 */
export async function isInChina(): Promise<boolean> {
  // Quick check based on language/locale
  const isChineseLocale = navigator.language.toLowerCase().startsWith('zh-cn')

  try {
    // Test connectivity to Google - commonly blocked in China
    const googleTest = await Promise.race([
      fetch('https://www.google.com', {
        mode: 'no-cors',
        cache: 'no-cache'
      }),
      new Promise((_, reject) => setTimeout(() => reject(), 2000))
    ])

    // If Google is accessible, user is likely not in China
    if (googleTest) {
      return false
    }
  } catch {
    // Google is not accessible - potential indicator of being in China
    if (isChineseLocale) {
      return true
    }

    // Additional check - test latency to a reliable Chinese domain
    try {
      const start = performance.now()
      await fetch('https://www.baidu.com', {
        mode: 'no-cors',
        cache: 'no-cache'
      })
      const latency = performance.now() - start

      // If Baidu responds quickly (<150ms), user is likely in China
      return latency < 150
    } catch {
      // If both tests fail, default to locale check
      return isChineseLocale
    }
  }

  return false
}
