import { describe, expect, it } from 'vitest'

import { renderMarkdownToHtml } from '@/utils/markdownRendererUtil'

describe('markdownRendererUtil', () => {
  describe('renderMarkdownToHtml', () => {
    it('should render basic markdown to HTML', () => {
      const markdown = '# Hello\n\nThis is a test.'
      const html = renderMarkdownToHtml(markdown)

      expect(html).toContain('<h1')
      expect(html).toContain('Hello')
      expect(html).toContain('<p>')
      expect(html).toContain('This is a test.')
    })

    it('should render links with target="_blank" and rel="noopener noreferrer"', () => {
      const markdown = '[Click here](https://example.com)'
      const html = renderMarkdownToHtml(markdown)

      expect(html).toContain('target="_blank"')
      expect(html).toContain('rel="noopener noreferrer"')
      expect(html).toContain('href="https://example.com"')
      expect(html).toContain('Click here')
    })

    it('should render multiple links with target="_blank"', () => {
      const markdown =
        '[Link 1](https://example.com) and [Link 2](https://test.com)'
      const html = renderMarkdownToHtml(markdown)

      const targetBlankMatches = html.match(/target="_blank"/g)
      expect(targetBlankMatches).toHaveLength(2)

      const relMatches = html.match(/rel="noopener noreferrer"/g)
      expect(relMatches).toHaveLength(2)
    })

    it('should handle relative image paths with baseUrl', () => {
      const markdown = '![Alt text](image.png)'
      const baseUrl = 'https://cdn.example.com'
      const html = renderMarkdownToHtml(markdown, baseUrl)

      expect(html).toContain(`src="${baseUrl}/image.png"`)
      expect(html).toContain('alt="Alt text"')
    })

    it('should not modify absolute image URLs', () => {
      const markdown = '![Alt text](https://example.com/image.png)'
      const baseUrl = 'https://cdn.example.com'
      const html = renderMarkdownToHtml(markdown, baseUrl)

      expect(html).toContain('src="https://example.com/image.png"')
      expect(html).not.toContain(baseUrl)
    })

    it('should handle empty markdown', () => {
      const html = renderMarkdownToHtml('')

      expect(html).toBe('')
    })

    it('should sanitize potentially dangerous HTML', () => {
      const markdown = '<script>alert("xss")</script>'
      const html = renderMarkdownToHtml(markdown)

      expect(html).not.toContain('<script>')
      expect(html).not.toContain('alert')
    })

    it('should allow video tags with proper attributes', () => {
      const markdown =
        '<video src="video.mp4" controls autoplay loop muted></video>'
      const html = renderMarkdownToHtml(markdown)

      expect(html).toContain('<video')
      expect(html).toContain('src="video.mp4"')
      expect(html).toContain('controls')
    })

    it('should render links with title attribute', () => {
      const markdown = '[Link](https://example.com "This is a title")'
      const html = renderMarkdownToHtml(markdown)

      expect(html).toContain('title="This is a title"')
      expect(html).toContain('target="_blank"')
      expect(html).toContain('rel="noopener noreferrer"')
    })

    it('should handle bare URLs (autolinks)', () => {
      const markdown = 'Visit https://example.com for more info.'
      const html = renderMarkdownToHtml(markdown)

      expect(html).toContain('href="https://example.com"')
      expect(html).toContain('target="_blank"')
      expect(html).toContain('rel="noopener noreferrer"')
    })

    it('should render complex markdown with links, images, and text', () => {
      const markdown = `
# Release Notes

Check out our [documentation](https://docs.example.com) for more info.

![Screenshot](screenshot.png)

Visit our [homepage](https://example.com) to learn more.
      `
      const baseUrl = 'https://cdn.example.com'
      const html = renderMarkdownToHtml(markdown, baseUrl)

      // Check links have target="_blank"
      const targetBlankMatches = html.match(/target="_blank"/g)
      expect(targetBlankMatches).toHaveLength(2)

      // Check image has baseUrl prepended
      expect(html).toContain(`${baseUrl}/screenshot.png`)

      // Check heading
      expect(html).toContain('Release Notes')
    })
  })
})
