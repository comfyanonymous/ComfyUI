// TODO: Fix these tests after migration
import { describe, expect, vi } from 'vitest'

import { subgraphTest } from './__fixtures__/subgraphFixtures'
import { verifyEventSequence } from './__fixtures__/subgraphHelpers'

describe.skip('SubgraphEvents - Event Payload Verification', () => {
  subgraphTest(
    'dispatches input-added with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const input = subgraph.addInput('test_input', 'number')

      const addedEvents = capture.getEventsByType('input-added')
      expect(addedEvents).toHaveLength(1)

      expect(addedEvents[0].detail).toEqual({
        input: expect.objectContaining({
          name: 'test_input',
          type: 'number'
        })
      })

      expect(addedEvents[0].detail.input).toBe(input)
    }
  )

  subgraphTest(
    'dispatches output-added with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const output = subgraph.addOutput('test_output', 'string')

      const addedEvents = capture.getEventsByType('output-added')
      expect(addedEvents).toHaveLength(1)

      expect(addedEvents[0].detail).toEqual({
        output: expect.objectContaining({
          name: 'test_output',
          type: 'string'
        })
      })

      expect(addedEvents[0].detail.output).toBe(output)
    }
  )

  subgraphTest(
    'dispatches removing-input with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const input = subgraph.addInput('to_remove', 'boolean')

      capture.clear()

      subgraph.removeInput(input)

      const removingEvents = capture.getEventsByType('removing-input')
      expect(removingEvents).toHaveLength(1)

      expect(removingEvents[0].detail).toEqual({
        input: expect.objectContaining({
          name: 'to_remove',
          type: 'boolean'
        }),
        index: 0
      })

      expect(removingEvents[0].detail.input).toBe(input)
    }
  )

  subgraphTest(
    'dispatches removing-output with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const output = subgraph.addOutput('to_remove', 'number')

      capture.clear()

      subgraph.removeOutput(output)

      const removingEvents = capture.getEventsByType('removing-output')
      expect(removingEvents).toHaveLength(1)

      expect(removingEvents[0].detail).toEqual({
        output: expect.objectContaining({
          name: 'to_remove',
          type: 'number'
        }),
        index: 0
      })

      expect(removingEvents[0].detail.output).toBe(output)
    }
  )

  subgraphTest(
    'dispatches renaming-input with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const input = subgraph.addInput('old_name', 'string')

      capture.clear()

      subgraph.renameInput(input, 'new_name')

      const renamingEvents = capture.getEventsByType('renaming-input')
      expect(renamingEvents).toHaveLength(1)

      expect(renamingEvents[0].detail).toEqual({
        input: expect.objectContaining({
          type: 'string'
        }),
        index: 0,
        oldName: 'old_name',
        newName: 'new_name'
      })

      expect(renamingEvents[0].detail.input).toBe(input)

      // Verify the label was updated after the event (renameInput sets label, not name)
      expect(input.label).toBe('new_name')
      expect(input.displayName).toBe('new_name')
      expect(input.name).toBe('old_name')
    }
  )

  subgraphTest(
    'dispatches renaming-output with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const output = subgraph.addOutput('old_name', 'number')

      capture.clear()

      subgraph.renameOutput(output, 'new_name')

      const renamingEvents = capture.getEventsByType('renaming-output')
      expect(renamingEvents).toHaveLength(1)

      expect(renamingEvents[0].detail).toEqual({
        output: expect.objectContaining({
          name: 'old_name', // Should still have the old name when event is dispatched
          type: 'number'
        }),
        index: 0,
        oldName: 'old_name',
        newName: 'new_name'
      })

      expect(renamingEvents[0].detail.output).toBe(output)

      // Verify the label was updated after the event
      expect(output.label).toBe('new_name')
      expect(output.displayName).toBe('new_name')
      expect(output.name).toBe('old_name')
    }
  )

  subgraphTest(
    'dispatches adding-input with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addInput('test_input', 'number')

      const addingEvents = capture.getEventsByType('adding-input')
      expect(addingEvents).toHaveLength(1)

      expect(addingEvents[0].detail).toEqual({
        name: 'test_input',
        type: 'number'
      })
    }
  )

  subgraphTest(
    'dispatches adding-output with correct payload',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addOutput('test_output', 'string')

      const addingEvents = capture.getEventsByType('adding-output')
      expect(addingEvents).toHaveLength(1)

      expect(addingEvents[0].detail).toEqual({
        name: 'test_output',
        type: 'string'
      })
    }
  )
})

describe.skip('SubgraphEvents - Event Handler Isolation', () => {
  subgraphTest(
    'continues dispatching if handler throws',
    ({ emptySubgraph }) => {
      const handler1 = vi.fn(() => {
        throw new Error('Handler 1 error')
      })
      const handler2 = vi.fn()
      const handler3 = vi.fn()

      emptySubgraph.events.addEventListener('input-added', handler1)
      emptySubgraph.events.addEventListener('input-added', handler2)
      emptySubgraph.events.addEventListener('input-added', handler3)

      // The operation itself should not throw (error is isolated)
      expect(() => {
        emptySubgraph.addInput('test', 'number')
      }).not.toThrow()

      // Verify all handlers were called despite the first one throwing
      expect(handler1).toHaveBeenCalled()
      expect(handler2).toHaveBeenCalled()
      expect(handler3).toHaveBeenCalled()

      // Verify the throwing handler actually received the event
      expect(handler1).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'input-added'
        })
      )

      // Verify other handlers received correct event data
      expect(handler2).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'input-added',
          detail: expect.objectContaining({
            input: expect.objectContaining({
              name: 'test',
              type: 'number'
            })
          })
        })
      )
      expect(handler3).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'input-added'
        })
      )
    }
  )

  subgraphTest('maintains handler execution order', ({ emptySubgraph }) => {
    const executionOrder: number[] = []

    const handler1 = vi.fn(() => executionOrder.push(1))
    const handler2 = vi.fn(() => executionOrder.push(2))
    const handler3 = vi.fn(() => executionOrder.push(3))

    emptySubgraph.events.addEventListener('input-added', handler1)
    emptySubgraph.events.addEventListener('input-added', handler2)
    emptySubgraph.events.addEventListener('input-added', handler3)

    emptySubgraph.addInput('test', 'number')

    expect(executionOrder).toEqual([1, 2, 3])
  })

  subgraphTest(
    'prevents handler accumulation with proper cleanup',
    ({ emptySubgraph }) => {
      const handler = vi.fn()

      for (let i = 0; i < 5; i++) {
        emptySubgraph.events.addEventListener('input-added', handler)
        emptySubgraph.events.removeEventListener('input-added', handler)
      }

      emptySubgraph.events.addEventListener('input-added', handler)

      emptySubgraph.addInput('test', 'number')

      expect(handler).toHaveBeenCalledTimes(1)
    }
  )

  subgraphTest(
    'supports AbortController cleanup patterns',
    ({ emptySubgraph }) => {
      const abortController = new AbortController()
      const { signal } = abortController

      const handler = vi.fn()

      emptySubgraph.events.addEventListener('input-added', handler, { signal })

      emptySubgraph.addInput('test1', 'number')
      expect(handler).toHaveBeenCalledTimes(1)

      abortController.abort()

      emptySubgraph.addInput('test2', 'number')
      expect(handler).toHaveBeenCalledTimes(1)
    }
  )
})

describe.skip('SubgraphEvents - Event Sequence Testing', () => {
  subgraphTest(
    'maintains correct event sequence for inputs',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addInput('input1', 'number')

      verifyEventSequence(capture.events, ['adding-input', 'input-added'])
    }
  )

  subgraphTest(
    'maintains correct event sequence for outputs',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addOutput('output1', 'string')

      verifyEventSequence(capture.events, ['adding-output', 'output-added'])
    }
  )

  subgraphTest(
    'maintains correct event sequence for rapid operations',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addInput('input1', 'number')
      subgraph.addInput('input2', 'string')
      subgraph.addOutput('output1', 'boolean')
      subgraph.addOutput('output2', 'number')

      verifyEventSequence(capture.events, [
        'adding-input',
        'input-added',
        'adding-input',
        'input-added',
        'adding-output',
        'output-added',
        'adding-output',
        'output-added'
      ])
    }
  )

  subgraphTest('handles concurrent event handling', ({ eventCapture }) => {
    const { subgraph, capture } = eventCapture

    const handler1 = vi.fn(() => {
      return new Promise((resolve) => setTimeout(resolve, 1))
    })

    const handler2 = vi.fn()
    const handler3 = vi.fn()

    subgraph.events.addEventListener('input-added', handler1)
    subgraph.events.addEventListener('input-added', handler2)
    subgraph.events.addEventListener('input-added', handler3)

    subgraph.addInput('test', 'number')

    expect(handler1).toHaveBeenCalled()
    expect(handler2).toHaveBeenCalled()
    expect(handler3).toHaveBeenCalled()

    const addedEvents = capture.getEventsByType('input-added')
    expect(addedEvents).toHaveLength(1)
  })

  subgraphTest(
    'validates event timestamps are properly ordered',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      subgraph.addInput('input1', 'number')
      subgraph.addInput('input2', 'string')
      subgraph.addOutput('output1', 'boolean')

      for (let i = 1; i < capture.events.length; i++) {
        expect(capture.events[i].timestamp).toBeGreaterThanOrEqual(
          capture.events[i - 1].timestamp
        )
      }
    }
  )
})

describe.skip('SubgraphEvents - Event Cancellation', () => {
  subgraphTest(
    'supports preventDefault() for cancellable events',
    ({ emptySubgraph }) => {
      const preventHandler = vi.fn((event: Event) => {
        event.preventDefault()
      })

      emptySubgraph.events.addEventListener('removing-input', preventHandler)

      const input = emptySubgraph.addInput('test', 'number')

      emptySubgraph.removeInput(input)

      expect(emptySubgraph.inputs).toContain(input)
      expect(preventHandler).toHaveBeenCalled()
    }
  )

  subgraphTest(
    'supports preventDefault() for output removal',
    ({ emptySubgraph }) => {
      const preventHandler = vi.fn((event: Event) => {
        event.preventDefault()
      })

      emptySubgraph.events.addEventListener('removing-output', preventHandler)

      const output = emptySubgraph.addOutput('test', 'number')

      emptySubgraph.removeOutput(output)

      expect(emptySubgraph.outputs).toContain(output)
      expect(preventHandler).toHaveBeenCalled()
    }
  )

  subgraphTest('allows removal when not prevented', ({ emptySubgraph }) => {
    const allowHandler = vi.fn()

    emptySubgraph.events.addEventListener('removing-input', allowHandler)

    const input = emptySubgraph.addInput('test', 'number')

    emptySubgraph.removeInput(input)

    expect(emptySubgraph.inputs).not.toContain(input)
    expect(emptySubgraph.inputs).toHaveLength(0)
    expect(allowHandler).toHaveBeenCalled()
  })
})

describe.skip('SubgraphEvents - Event Detail Structure Validation', () => {
  subgraphTest(
    'validates all event detail structures match TypeScript types',
    ({ eventCapture }) => {
      const { subgraph, capture } = eventCapture

      const input = subgraph.addInput('test_input', 'number')
      subgraph.renameInput(input, 'renamed_input')
      subgraph.removeInput(input)

      const output = subgraph.addOutput('test_output', 'string')
      subgraph.renameOutput(output, 'renamed_output')
      subgraph.removeOutput(output)

      const addingInputEvent = capture.getEventsByType('adding-input')[0]
      expect(addingInputEvent.detail).toEqual({
        name: expect.any(String),
        type: expect.any(String)
      })

      const inputAddedEvent = capture.getEventsByType('input-added')[0]
      expect(inputAddedEvent.detail).toEqual({
        input: expect.any(Object)
      })

      const renamingInputEvent = capture.getEventsByType('renaming-input')[0]
      expect(renamingInputEvent.detail).toEqual({
        input: expect.any(Object),
        index: expect.any(Number),
        oldName: expect.any(String),
        newName: expect.any(String)
      })

      const removingInputEvent = capture.getEventsByType('removing-input')[0]
      expect(removingInputEvent.detail).toEqual({
        input: expect.any(Object),
        index: expect.any(Number)
      })

      const addingOutputEvent = capture.getEventsByType('adding-output')[0]
      expect(addingOutputEvent.detail).toEqual({
        name: expect.any(String),
        type: expect.any(String)
      })

      const outputAddedEvent = capture.getEventsByType('output-added')[0]
      expect(outputAddedEvent.detail).toEqual({
        output: expect.any(Object)
      })

      const renamingOutputEvent = capture.getEventsByType('renaming-output')[0]
      expect(renamingOutputEvent.detail).toEqual({
        output: expect.any(Object),
        index: expect.any(Number),
        oldName: expect.any(String),
        newName: expect.any(String)
      })

      const removingOutputEvent = capture.getEventsByType('removing-output')[0]
      expect(removingOutputEvent.detail).toEqual({
        output: expect.any(Object),
        index: expect.any(Number)
      })
    }
  )
})
