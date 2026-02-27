import { describe, expect, it } from 'vitest'

import {
  isCloudValidationError,
  tryExtractValidationError,
  classifyCloudValidationError
} from '@/utils/executionErrorUtil'

describe('executionErrorUtil', () => {
  describe('isCloudValidationError', () => {
    it('should return true when object has error field', () => {
      expect(isCloudValidationError({ error: 'some error' })).toBe(true)
    })

    it('should return true when object has node_errors field', () => {
      expect(isCloudValidationError({ node_errors: {} })).toBe(true)
    })

    it('should return true when object has both fields', () => {
      expect(isCloudValidationError({ error: 'err', node_errors: {} })).toBe(
        true
      )
    })

    it('should return false for null', () => {
      expect(isCloudValidationError(null)).toBe(false)
    })

    it('should return false for non-object', () => {
      expect(isCloudValidationError('string')).toBe(false)
    })

    it('should return false for object without error or node_errors', () => {
      expect(isCloudValidationError({ foo: 'bar' })).toBe(false)
    })
  })

  describe('tryExtractValidationError', () => {
    it('should extract JSON from a message with embedded validation error', () => {
      const embedded = JSON.stringify({
        error: {
          type: 'prompt_no_outputs',
          message: 'No outputs',
          details: ''
        },
        node_errors: {}
      })
      const message = `Failed to send prompt request: status 400: ${embedded}`

      const result = tryExtractValidationError(message)

      expect(result).not.toBeNull()
      expect(result?.error).toEqual({
        type: 'prompt_no_outputs',
        message: 'No outputs',
        details: ''
      })
    })

    it('should return null when message has no JSON', () => {
      expect(tryExtractValidationError('plain error message')).toBeNull()
    })

    it('should return null when JSON is not a validation error shape', () => {
      const message = 'error: {"foo": "bar"}'
      expect(tryExtractValidationError(message)).toBeNull()
    })

    it('should return null when JSON is malformed', () => {
      const message = 'error: {invalid json'
      expect(tryExtractValidationError(message)).toBeNull()
    })
  })

  describe('classifyCloudValidationError', () => {
    it('should classify node errors when node_errors is present', () => {
      const nodeErrors = {
        '11:1': {
          errors: [
            {
              type: 'required_input_missing',
              message: 'Required input is missing',
              details: 'clip',
              extra_info: { input_name: 'clip' }
            }
          ],
          dependent_outputs: ['9'],
          class_type: 'CLIPTextEncode'
        }
      }
      const embedded = JSON.stringify({
        error: {
          type: 'prompt_outputs_failed_validation',
          message: 'Prompt outputs failed validation',
          details: ''
        },
        node_errors: nodeErrors
      })
      const message = `Failed to send prompt request: status 400: ${embedded}`

      const result = classifyCloudValidationError(message)

      expect(result).not.toBeNull()
      expect(result?.kind).toBe('nodeErrors')
      if (result?.kind === 'nodeErrors') {
        expect(result.nodeErrors['11:1'].class_type).toBe('CLIPTextEncode')
      }
    })

    it('should classify prompt error when error is an object and no node_errors', () => {
      const embedded = JSON.stringify({
        error: {
          type: 'prompt_no_outputs',
          message: 'Prompt has no outputs',
          details: ''
        }
      })
      const message = `Failed: ${embedded}`

      const result = classifyCloudValidationError(message)

      expect(result).not.toBeNull()
      expect(result?.kind).toBe('promptError')
      if (result?.kind === 'promptError') {
        expect(result.promptError.type).toBe('prompt_no_outputs')
        expect(result.promptError.message).toBe('Prompt has no outputs')
      }
    })

    it('should classify prompt error when error is a string', () => {
      const embedded = JSON.stringify({ error: 'Something went wrong' })
      const message = `Failed: ${embedded}`

      const result = classifyCloudValidationError(message)

      expect(result).not.toBeNull()
      expect(result?.kind).toBe('promptError')
      if (result?.kind === 'promptError') {
        expect(result.promptError.type).toBe('error')
        expect(result.promptError.message).toBe('Something went wrong')
      }
    })

    it('should return null when message has no embedded JSON', () => {
      expect(classifyCloudValidationError('plain error')).toBeNull()
    })

    it('should return null when embedded JSON has no error or node_errors', () => {
      const message = 'error: {"foo": "bar"}'
      expect(classifyCloudValidationError(message)).toBeNull()
    })

    it('should return null when error field is neither object nor string', () => {
      const embedded = JSON.stringify({ error: 123 })
      const message = `Failed: ${embedded}`

      expect(classifyCloudValidationError(message)).toBeNull()
    })

    it('should prefer node_errors over error when both present', () => {
      const embedded = JSON.stringify({
        error: { type: 'validation', message: 'fail', details: '' },
        node_errors: {
          '5': {
            errors: [{ type: 'err', message: 'bad', details: '' }],
            dependent_outputs: [],
            class_type: 'KSampler'
          }
        }
      })
      const message = `Failed: ${embedded}`

      const result = classifyCloudValidationError(message)

      expect(result?.kind).toBe('nodeErrors')
    })

    it('should treat empty node_errors as prompt error', () => {
      const embedded = JSON.stringify({
        error: { type: 'no_prompt', message: 'No prompt', details: '' },
        node_errors: {}
      })
      const message = `Failed: ${embedded}`

      const result = classifyCloudValidationError(message)

      expect(result?.kind).toBe('promptError')
    })
  })
})
