import { describe, expect, it } from 'vitest'

import {
  getMediaTypeFromFilename,
  highlightQuery,
  isPreviewableMediaType,
  truncateFilename
} from './formatUtil'

describe('formatUtil', () => {
  describe('truncateFilename', () => {
    it('should not truncate short filenames', () => {
      expect(truncateFilename('test.png')).toBe('test.png')
      expect(truncateFilename('short.jpg', 10)).toBe('short.jpg')
    })

    it('should truncate long filenames while preserving extension', () => {
      const longName = 'this-is-a-very-long-filename-that-needs-truncation.png'
      const truncated = truncateFilename(longName, 20)
      expect(truncated).toContain('...')
      expect(truncated.endsWith('.png')).toBe(true)
      expect(truncated.length).toBeLessThanOrEqual(25) // 20 + '...' + extension
    })

    it('should handle filenames without extensions', () => {
      const longName = 'this-is-a-very-long-filename-without-extension'
      const truncated = truncateFilename(longName, 20)
      expect(truncated).toContain('...')
      expect(truncated.length).toBeLessThanOrEqual(23) // 20 + '...'
    })

    it('should handle empty strings', () => {
      expect(truncateFilename('')).toBe('')
      expect(truncateFilename('', 10)).toBe('')
    })

    it('should preserve the start and end of the filename', () => {
      const longName = 'ComfyUI_00001_timestamp_2024_01_01.png'
      const truncated = truncateFilename(longName, 20)
      expect(truncated).toMatch(/^ComfyUI.*01\.png$/)
      expect(truncated).toContain('...')
    })

    it('should handle files with multiple dots', () => {
      const filename = 'my.file.with.multiple.dots.txt'
      const truncated = truncateFilename(filename, 15)
      expect(truncated.endsWith('.txt')).toBe(true)
      expect(truncated).toContain('...')
    })
  })

  describe('getMediaTypeFromFilename', () => {
    describe('image files', () => {
      const imageTestCases = [
        { filename: 'test.png', expected: 'image' },
        { filename: 'photo.jpg', expected: 'image' },
        { filename: 'image.jpeg', expected: 'image' },
        { filename: 'animation.gif', expected: 'image' },
        { filename: 'web.webp', expected: 'image' },
        { filename: 'bitmap.bmp', expected: 'image' },
        { filename: 'modern.avif', expected: 'image' }
      ]

      it.for(imageTestCases)(
        'should identify $filename as $expected',
        ({ filename, expected }) => {
          expect(getMediaTypeFromFilename(filename)).toBe(expected)
        }
      )

      it('should handle uppercase extensions', () => {
        expect(getMediaTypeFromFilename('test.PNG')).toBe('image')
        expect(getMediaTypeFromFilename('photo.JPG')).toBe('image')
      })
    })

    describe('video files', () => {
      it('should identify video extensions correctly', () => {
        expect(getMediaTypeFromFilename('video.mp4')).toBe('video')
        expect(getMediaTypeFromFilename('clip.webm')).toBe('video')
        expect(getMediaTypeFromFilename('movie.mov')).toBe('video')
        expect(getMediaTypeFromFilename('film.avi')).toBe('video')
      })
    })

    describe('audio files', () => {
      it('should identify audio extensions correctly', () => {
        expect(getMediaTypeFromFilename('song.mp3')).toBe('audio')
        expect(getMediaTypeFromFilename('sound.wav')).toBe('audio')
        expect(getMediaTypeFromFilename('music.ogg')).toBe('audio')
        expect(getMediaTypeFromFilename('audio.flac')).toBe('audio')
      })
    })

    describe('3D files', () => {
      it('should identify 3D file extensions correctly', () => {
        expect(getMediaTypeFromFilename('model.obj')).toBe('3D')
        expect(getMediaTypeFromFilename('scene.fbx')).toBe('3D')
        expect(getMediaTypeFromFilename('asset.gltf')).toBe('3D')
        expect(getMediaTypeFromFilename('binary.glb')).toBe('3D')
        expect(getMediaTypeFromFilename('apple.usdz')).toBe('3D')
      })
    })

    describe('text files', () => {
      it('should identify text file extensions correctly', () => {
        expect(getMediaTypeFromFilename('notes.txt')).toBe('text')
        expect(getMediaTypeFromFilename('readme.md')).toBe('text')
        expect(getMediaTypeFromFilename('data.json')).toBe('text')
        expect(getMediaTypeFromFilename('table.csv')).toBe('text')
        expect(getMediaTypeFromFilename('config.yaml')).toBe('text')
      })
    })

    describe('edge cases', () => {
      it('should handle empty strings', () => {
        expect(getMediaTypeFromFilename('')).toBe('other')
      })

      it('should handle files without extensions', () => {
        expect(getMediaTypeFromFilename('README')).toBe('other')
      })

      it('should handle unknown extensions', () => {
        expect(getMediaTypeFromFilename('document.pdf')).toBe('other')
        expect(getMediaTypeFromFilename('archive.bin')).toBe('other')
      })

      it('should handle files with multiple dots', () => {
        expect(getMediaTypeFromFilename('my.file.name.png')).toBe('image')
        expect(getMediaTypeFromFilename('archive.tar.gz')).toBe('other')
      })

      it('should handle paths with directories', () => {
        expect(getMediaTypeFromFilename('/path/to/image.png')).toBe('image')
        expect(getMediaTypeFromFilename('C:\\Windows\\video.mp4')).toBe('video')
      })

      it('should handle null and undefined gracefully', () => {
        expect(getMediaTypeFromFilename(null)).toBe('other')
        expect(getMediaTypeFromFilename(undefined)).toBe('other')
      })

      it('should handle special characters in filenames', () => {
        expect(getMediaTypeFromFilename('test@#$.png')).toBe('image')
        expect(getMediaTypeFromFilename('video (1).mp4')).toBe('video')
        expect(getMediaTypeFromFilename('[2024] audio.mp3')).toBe('audio')
      })

      it('should handle very long filenames', () => {
        const longFilename = 'a'.repeat(1000) + '.png'
        expect(getMediaTypeFromFilename(longFilename)).toBe('image')
      })

      it('should handle mixed case extensions', () => {
        expect(getMediaTypeFromFilename('test.PnG')).toBe('image')
        expect(getMediaTypeFromFilename('video.Mp4')).toBe('video')
        expect(getMediaTypeFromFilename('audio.WaV')).toBe('audio')
      })
    })
  })

  describe('highlightQuery', () => {
    it('should return text unchanged when query is empty', () => {
      expect(highlightQuery('Hello World', '')).toBe('Hello World')
    })

    it('should wrap matching text in highlight span', () => {
      const result = highlightQuery('Hello World', 'World')
      expect(result).toBe('Hello <span class="highlight">World</span>')
    })

    it('should be case-insensitive', () => {
      const result = highlightQuery('Hello World', 'hello')
      expect(result).toBe('<span class="highlight">Hello</span> World')
    })

    it('should sanitize text by default', () => {
      const result = highlightQuery('<script>alert("xss")</script>', 'alert')
      expect(result).not.toContain('<script>')
    })

    it('should skip sanitization when sanitize is false', () => {
      const result = highlightQuery('<b>bold</b>', 'bold', false)
      expect(result).toContain('<b>')
    })

    it('should escape special regex characters in query', () => {
      const result = highlightQuery('price is $10.00', '$10')
      expect(result).toContain('<span class="highlight">$10</span>')
    })

    it('should highlight multiple occurrences', () => {
      const result = highlightQuery('foo bar foo', 'foo')
      expect(result).toBe(
        '<span class="highlight">foo</span> bar <span class="highlight">foo</span>'
      )
    })
  })

  describe('isPreviewableMediaType', () => {
    it('returns true for image/video/audio/3D', () => {
      expect(isPreviewableMediaType('image')).toBe(true)
      expect(isPreviewableMediaType('video')).toBe(true)
      expect(isPreviewableMediaType('audio')).toBe(true)
      expect(isPreviewableMediaType('3D')).toBe(true)
    })

    it('returns false for text/other', () => {
      expect(isPreviewableMediaType('text')).toBe(false)
      expect(isPreviewableMediaType('other')).toBe(false)
    })
  })
})
