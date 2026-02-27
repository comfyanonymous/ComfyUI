import { describe, expect, it } from 'vitest'

import { isHostWhitelisted, normalizeHost } from '@/utils/hostWhitelist'

describe('hostWhitelist utils', () => {
  describe('normalizeHost', () => {
    it.each([
      ['LOCALHOST', 'localhost'],
      ['localhost.', 'localhost'], // trims trailing dot
      ['localhost:5173', 'localhost'], // strips :port
      ['127.0.0.1:5173', '127.0.0.1'], // strips :port
      ['[::1]:5173', '::1'], // strips brackets + :port
      ['[::1]', '::1'], // strips brackets
      ['::1', '::1'], // leaves plain IPv6
      ['  [::1]  ', '::1'], // trims whitespace
      ['APP.LOCALHOST', 'app.localhost'], // lowercases
      ['example.com.', 'example.com'], // trims trailing dot
      ['[2001:db8::1]:8443', '2001:db8::1'], // IPv6 with brackets+port
      ['2001:db8::1', '2001:db8::1'] // plain IPv6 stays
    ])('normalizeHost(%o) -> %o', (input, expected) => {
      expect(normalizeHost(input)).toBe(expected)
    })

    it('does not strip non-numeric suffixes (not a port pattern)', () => {
      expect(normalizeHost('example.com:abc')).toBe('example.com:abc')
      expect(normalizeHost('127.0.0.1:abc')).toBe('127.0.0.1:abc')
    })
  })

  describe('isHostWhitelisted', () => {
    describe('localhost label', () => {
      it.each([
        'localhost',
        'LOCALHOST',
        'localhost.',
        'localhost:5173',
        'foo.localhost',
        'Foo.Localhost',
        'sub.foo.localhost',
        'foo.localhost:5173'
      ])('should allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(true)
      })

      it.each([
        'localhost.com',
        'evil-localhost',
        'notlocalhost',
        'foo.localhost.evil'
      ])('should NOT allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(false)
      })
    })

    describe('IPv4 127/8 loopback', () => {
      it.each([
        '127.0.0.1',
        '127.1.2.3',
        '127.255.255.255',
        '127.0.0.1:3000',
        '127.000.000.001', // leading zeros are still digits 0-255
        '127.0.0.1.' // trailing dot should be tolerated
      ])('should allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(true)
      })

      it.each([
        '126.0.0.1',
        '127.256.0.1',
        '127.-1.0.1',
        '127.0.0.1:abc',
        '128.0.0.1',
        '192.168.1.10',
        '10.0.0.2',
        '0.0.0.0',
        '255.255.255.255',
        '127.0.0', // malformed
        '127.0.0.1.5' // malformed
      ])('should NOT allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(false)
      })
    })

    describe('IPv6 loopback ::1 (all textual forms)', () => {
      it.each([
        '::1',
        '[::1]',
        '[::1]:5173',
        '::0001',
        '0:0:0:0:0:0:0:1',
        '0000:0000:0000:0000:0000:0000:0000:0001',
        // Compressed equivalents of ::1 (with zeros compressed)
        '0:0::1',
        '0:0:0:0:0:0::1',
        '::0:1' // compressing the initial zeros (still ::1 when expanded)
      ])('should allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(true)
      })

      it.each([
        '::2',
        '::',
        '::0',
        '0:0:0:0:0:0:0:2',
        'fe80::1', // link-local, not loopback
        '2001:db8::1',
        '::1:5173', // bracketless "port-like" suffix must not pass
        ':::1', // invalid (triple colon)
        '0:0:0:0:0:0:::1', // invalid compression
        '[::1%25lo0]',
        '[::1%25lo0]:5173',
        '::1%25lo0'
      ])('should NOT allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(false)
      })

      it('should reject empty/whitespace-only input', () => {
        expect(isHostWhitelisted('')).toBe(false)
        expect(isHostWhitelisted('   ')).toBe(false)
      })
    })

    describe('comfy.org hosts', () => {
      it.each([
        'staging.comfy.org',
        'stagingcloud.comfy.org',
        'pr-123.testingcloud.comfy.org',
        'api.v2.staging.comfy.org'
      ])('should allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(true)
      })

      it.each([
        'comfy.org.evil.com',
        'evil-comfy.org',
        'comfy.organization',
        'notcomfy.org',
        'comfy.org.hacker.net',
        'mycomfy.org.example.com'
      ])('should NOT allow %o', (input) => {
        expect(isHostWhitelisted(input)).toBe(false)
      })
    })
  })
})
