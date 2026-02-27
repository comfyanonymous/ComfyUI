import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  downloadFile,
  extractFilenameFromContentDisposition,
  openFileInNewTab
} from '@/base/common/downloadUtil'

const { mockIsCloud } = vi.hoisted(() => ({
  mockIsCloud: { value: false }
}))

vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockIsCloud.value
  }
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: vi.fn(() => ({ addAlert: vi.fn() }))
}))

// Global stubs
const createObjectURLSpy = vi
  .spyOn(URL, 'createObjectURL')
  .mockReturnValue('blob:mock-url')
const revokeObjectURLSpy = vi
  .spyOn(URL, 'revokeObjectURL')
  .mockImplementation(() => {})

describe('downloadUtil', () => {
  let mockLink: HTMLAnchorElement
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    mockIsCloud.value = false
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    createObjectURLSpy.mockClear().mockReturnValue('blob:mock-url')
    revokeObjectURLSpy.mockClear().mockImplementation(() => {})
    // Create a mock anchor element
    mockLink = {
      href: '',
      download: '',
      click: vi.fn(),
      style: { display: '' }
    } as unknown as HTMLAnchorElement

    // Spy on DOM methods
    vi.spyOn(document, 'createElement').mockReturnValue(mockLink)
    vi.spyOn(document.body, 'appendChild').mockImplementation(() => mockLink)
    vi.spyOn(document.body, 'removeChild').mockImplementation(() => mockLink)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('downloadFile', () => {
    it('should create and trigger download with basic URL', () => {
      const testUrl = 'https://example.com/image.png'

      downloadFile(testUrl)

      expect(document.createElement).toHaveBeenCalledWith('a')
      expect(mockLink.href).toBe(testUrl)
      expect(mockLink.download).toBe('download.png') // Default filename
      expect(document.body.appendChild).toHaveBeenCalledWith(mockLink)
      expect(mockLink.click).toHaveBeenCalled()
      expect(document.body.removeChild).toHaveBeenCalledWith(mockLink)
      expect(fetchMock).not.toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should use custom filename when provided', () => {
      const testUrl = 'https://example.com/image.png'
      const customFilename = 'my-custom-image.png'

      downloadFile(testUrl, customFilename)

      expect(mockLink.href).toBe(testUrl)
      expect(mockLink.download).toBe(customFilename)
      expect(fetchMock).not.toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should extract filename from URL query parameters', () => {
      const testUrl =
        'https://example.com/api/file?filename=extracted-image.jpg&other=param'

      downloadFile(testUrl)

      expect(mockLink.href).toBe(testUrl)
      expect(mockLink.download).toBe('extracted-image.jpg')
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should use default filename when URL has no filename parameter', () => {
      const testUrl = 'https://example.com/api/file?other=param'

      downloadFile(testUrl)

      expect(mockLink.href).toBe(testUrl)
      expect(mockLink.download).toBe('download.png')
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should handle invalid URLs gracefully', () => {
      const invalidUrl = 'not-a-valid-url'

      downloadFile(invalidUrl)

      expect(mockLink.href).toBe(invalidUrl)
      expect(mockLink.download).toBe('download.png')
      expect(mockLink.click).toHaveBeenCalled()
      expect(fetchMock).not.toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should prefer custom filename over extracted filename', () => {
      const testUrl =
        'https://example.com/api/file?filename=extracted-image.jpg'
      const customFilename = 'custom-override.png'

      downloadFile(testUrl, customFilename)

      expect(mockLink.download).toBe(customFilename)
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should handle URLs with empty filename parameter', () => {
      const testUrl = 'https://example.com/api/file?filename='

      downloadFile(testUrl)

      expect(mockLink.download).toBe('download.png')
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should handle relative URLs by using window.location.origin', () => {
      const relativeUrl = '/api/file?filename=relative-image.png'

      downloadFile(relativeUrl)

      expect(mockLink.href).toBe(relativeUrl)
      expect(mockLink.download).toBe('relative-image.png')
      expect(fetchMock).not.toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('should clean up DOM elements after download', () => {
      const testUrl = 'https://example.com/image.png'

      downloadFile(testUrl)

      // Verify the element was added and then removed
      expect(document.body.appendChild).toHaveBeenCalledWith(mockLink)
      expect(document.body.removeChild).toHaveBeenCalledWith(mockLink)
      expect(fetchMock).not.toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
    })

    it('streams downloads via blob when running in cloud', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/file.bin'
      const blob = new Blob(['test'])
      const blobFn = vi.fn().mockResolvedValue(blob)
      const headersMock = {
        get: vi.fn().mockReturnValue(null)
      }
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        blob: blobFn,
        headers: headersMock
      } as unknown as Response)

      downloadFile(testUrl)

      expect(fetchMock).toHaveBeenCalledWith(testUrl)
      const fetchPromise = fetchMock.mock.results[0].value as Promise<Response>
      await fetchPromise
      await Promise.resolve() // let fetchAsBlob return
      const blobPromise = blobFn.mock.results[0].value as Promise<Blob>
      await blobPromise
      await Promise.resolve()
      expect(blobFn).toHaveBeenCalled()
      expect(createObjectURLSpy).toHaveBeenCalledWith(blob)
      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url')
      expect(mockLink.click).toHaveBeenCalled()
    })

    it('logs an error when cloud fetch fails', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/missing.bin'
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      fetchMock.mockResolvedValue({
        ok: false,
        status: 404,
        blob: vi.fn()
      } as Partial<Response> as Response)

      downloadFile(testUrl)

      expect(fetchMock).toHaveBeenCalledWith(testUrl)
      const fetchPromise = fetchMock.mock.results[0].value as Promise<Response>
      await fetchPromise
      await Promise.resolve() // let fetchAsBlob throw
      await Promise.resolve() // let .catch handler run
      expect(consoleSpy).toHaveBeenCalled()
      expect(createObjectURLSpy).not.toHaveBeenCalled()
      consoleSpy.mockRestore()
    })

    it('uses filename from Content-Disposition header in cloud mode', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/abc123.png'
      const blob = new Blob(['test'])
      const blobFn = vi.fn().mockResolvedValue(blob)
      const headersMock = {
        get: vi.fn().mockReturnValue('attachment; filename="user-friendly.png"')
      }
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        blob: blobFn,
        headers: headersMock
      } as unknown as Response)

      downloadFile(testUrl)

      expect(fetchMock).toHaveBeenCalledWith(testUrl)
      const fetchPromise = fetchMock.mock.results[0].value as Promise<Response>
      await fetchPromise
      await Promise.resolve() // let fetchAsBlob return
      const blobPromise = blobFn.mock.results[0].value as Promise<Blob>
      await blobPromise
      await Promise.resolve()
      expect(headersMock.get).toHaveBeenCalledWith('Content-Disposition')
      expect(mockLink.download).toBe('user-friendly.png')
    })

    it('uses RFC 5987 filename from Content-Disposition header', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/abc123.png'
      const blob = new Blob(['test'])
      const blobFn = vi.fn().mockResolvedValue(blob)
      const headersMock = {
        get: vi
          .fn()
          .mockReturnValue(
            'attachment; filename="fallback.png"; filename*=UTF-8\'\'%E4%B8%AD%E6%96%87.png'
          )
      }
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        blob: blobFn,
        headers: headersMock
      } as unknown as Response)

      downloadFile(testUrl)

      const fetchPromise = fetchMock.mock.results[0].value as Promise<Response>
      await fetchPromise
      await Promise.resolve() // let fetchAsBlob return
      const blobPromise = blobFn.mock.results[0].value as Promise<Blob>
      await blobPromise
      await Promise.resolve()
      expect(mockLink.download).toBe('中文.png')
    })

    it('falls back to provided filename when Content-Disposition is missing', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/abc123.png'
      const blob = new Blob(['test'])
      const blobFn = vi.fn().mockResolvedValue(blob)
      const headersMock = {
        get: vi.fn().mockReturnValue(null)
      }
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        blob: blobFn,
        headers: headersMock
      } as unknown as Response)

      downloadFile(testUrl, 'my-fallback.png')

      const fetchPromise = fetchMock.mock.results[0].value as Promise<Response>
      await fetchPromise
      await Promise.resolve() // let fetchAsBlob return
      const blobPromise = blobFn.mock.results[0].value as Promise<Blob>
      await blobPromise
      await Promise.resolve()
      expect(mockLink.download).toBe('my-fallback.png')
    })
  })

  describe('openFileInNewTab', () => {
    let windowOpenSpy: ReturnType<typeof vi.spyOn>

    beforeEach(() => {
      vi.useFakeTimers()
      windowOpenSpy = vi.spyOn(window, 'open').mockImplementation(() => null)
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('opens URL directly when not in cloud mode', async () => {
      mockIsCloud.value = false
      const testUrl = 'https://example.com/image.png'

      await openFileInNewTab(testUrl)

      expect(windowOpenSpy).toHaveBeenCalledWith(testUrl, '_blank')
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it('opens blank tab synchronously then navigates to blob URL in cloud mode', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/image.png'
      const blob = new Blob(['test'], { type: 'image/png' })
      const mockTab = { location: { href: '' }, closed: false, close: vi.fn() }
      windowOpenSpy.mockReturnValue(mockTab as unknown as Window)
      fetchMock.mockResolvedValue({
        ok: true,
        blob: vi.fn().mockResolvedValue(blob)
      } as unknown as Response)

      await openFileInNewTab(testUrl)

      expect(windowOpenSpy).toHaveBeenCalledWith('', '_blank')
      expect(fetchMock).toHaveBeenCalledWith(testUrl)
      expect(createObjectURLSpy).toHaveBeenCalledWith(blob)
      expect(mockTab.location.href).toBe('blob:mock-url')
    })

    it('revokes blob URL after timeout in cloud mode', async () => {
      mockIsCloud.value = true
      const blob = new Blob(['test'], { type: 'image/png' })
      const mockTab = { location: { href: '' }, closed: false, close: vi.fn() }
      windowOpenSpy.mockReturnValue(mockTab as unknown as Window)
      fetchMock.mockResolvedValue({
        ok: true,
        blob: vi.fn().mockResolvedValue(blob)
      } as unknown as Response)

      await openFileInNewTab('https://example.com/image.png')

      expect(revokeObjectURLSpy).not.toHaveBeenCalled()
      vi.advanceTimersByTime(60_000)
      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url')
    })

    it('closes blank tab and logs error when cloud fetch fails', async () => {
      mockIsCloud.value = true
      const testUrl = 'https://storage.googleapis.com/bucket/missing.png'
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const mockTab = { location: { href: '' }, closed: false, close: vi.fn() }
      windowOpenSpy.mockReturnValue(mockTab as unknown as Window)
      fetchMock.mockResolvedValue({
        ok: false,
        status: 404
      } as unknown as Response)

      await openFileInNewTab(testUrl)

      expect(mockTab.close).toHaveBeenCalled()
      expect(consoleSpy).toHaveBeenCalled()
      consoleSpy.mockRestore()
    })

    it('revokes blob URL immediately if tab was closed by user', async () => {
      mockIsCloud.value = true
      const blob = new Blob(['test'], { type: 'image/png' })
      const mockTab = { location: { href: '' }, closed: true, close: vi.fn() }
      windowOpenSpy.mockReturnValue(mockTab as unknown as Window)
      fetchMock.mockResolvedValue({
        ok: true,
        blob: vi.fn().mockResolvedValue(blob)
      } as unknown as Response)

      await openFileInNewTab('https://example.com/image.png')

      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url')
      expect(mockTab.location.href).toBe('')
    })
  })

  describe('extractFilenameFromContentDisposition', () => {
    it('returns null for null header', () => {
      expect(extractFilenameFromContentDisposition(null)).toBeNull()
    })

    it('returns null for empty header', () => {
      expect(extractFilenameFromContentDisposition('')).toBeNull()
    })

    it('extracts filename from simple quoted format', () => {
      const header = 'attachment; filename="test-file.png"'
      expect(extractFilenameFromContentDisposition(header)).toBe(
        'test-file.png'
      )
    })

    it('extracts filename from unquoted format', () => {
      const header = 'attachment; filename=test-file.png'
      expect(extractFilenameFromContentDisposition(header)).toBe(
        'test-file.png'
      )
    })

    it('extracts filename from RFC 5987 format', () => {
      const header = "attachment; filename*=UTF-8''test%20file.png"
      expect(extractFilenameFromContentDisposition(header)).toBe(
        'test file.png'
      )
    })

    it('prefers RFC 5987 format over simple format', () => {
      const header =
        'attachment; filename="fallback.png"; filename*=UTF-8\'\'preferred.png'
      expect(extractFilenameFromContentDisposition(header)).toBe(
        'preferred.png'
      )
    })

    it('handles unicode characters in RFC 5987 format', () => {
      const header =
        "attachment; filename*=UTF-8''%E4%B8%AD%E6%96%87%E6%96%87%E4%BB%B6.png"
      expect(extractFilenameFromContentDisposition(header)).toBe('中文文件.png')
    })

    it('falls back to simple format when RFC 5987 decoding fails', () => {
      const header =
        'attachment; filename="fallback.png"; filename*=UTF-8\'\'%invalid'
      expect(extractFilenameFromContentDisposition(header)).toBe('fallback.png')
    })

    it('handles header with only attachment disposition', () => {
      const header = 'attachment'
      expect(extractFilenameFromContentDisposition(header)).toBeNull()
    })

    it('handles case-insensitive filename parameter', () => {
      const header = 'attachment; FILENAME="test.png"'
      expect(extractFilenameFromContentDisposition(header)).toBe('test.png')
    })
  })
})
