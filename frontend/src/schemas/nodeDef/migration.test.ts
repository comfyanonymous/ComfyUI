import { describe, expect, it } from 'vitest'

import { transformNodeDefV1ToV2 } from '@/schemas/nodeDef/migration'
import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

describe('NodeDef Migration', () => {
  it('should transform a plain object to V2 format', () => {
    const plainObject = {
      required: {
        intInput: ['INT', { min: 0, max: 100, default: 50 }],
        stringInput: ['STRING', { default: 'Hello', multiline: true }]
      },
      optional: {
        booleanInput: [
          'BOOLEAN',
          { default: true, labelOn: 'Yes', labelOff: 'No' }
        ],
        floatInput: ['FLOAT', { min: 0, max: 1, step: 0.1 }]
      },
      hidden: {
        someHiddenValue: 42
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: ['INT'],
      output_is_list: [false],
      output_name: ['intOutput'],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)

    expect(result).toBeDefined()
    expect(result.inputs).toBeDefined()
    expect(result.outputs).toBeDefined()
    expect(result.hidden).toBeDefined()
  })

  it('should correctly transform required input specs', () => {
    const plainObject = {
      required: {
        intInput: ['INT', { min: 0, max: 100, default: 50 }],
        stringInput: ['STRING', { default: 'Hello', multiline: true }]
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)

    const intInput = result.inputs['intInput']
    const stringInput = result.inputs['stringInput']

    expect(intInput.min).toBe(0)
    expect(intInput.max).toBe(100)
    expect(intInput.default).toBe(50)
    expect(intInput.name).toBe('intInput')
    expect(intInput.type).toBe('INT')
    expect(stringInput.default).toBe('Hello')
    expect(stringInput.multiline).toBe(true)
    expect(stringInput.name).toBe('stringInput')
    expect(stringInput.type).toBe('STRING')
  })

  it('should correctly transform optional input specs', () => {
    const plainObject = {
      optional: {
        booleanInput: [
          'BOOLEAN',
          { default: true, labelOn: 'Yes', labelOff: 'No' }
        ],
        floatInput: ['FLOAT', { min: 0, max: 1, step: 0.1 }]
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)

    const booleanInput = result.inputs['booleanInput']
    const floatInput = result.inputs['floatInput']

    expect(booleanInput.default).toBe(true)
    expect(booleanInput.labelOn).toBe('Yes')
    expect(booleanInput.labelOff).toBe('No')
    expect(booleanInput.type).toBe('BOOLEAN')
    expect(booleanInput.isOptional).toBe(true)
    expect(floatInput.min).toBe(0)
    expect(floatInput.max).toBe(1)
    expect(floatInput.step).toBe(0.1)
    expect(floatInput.type).toBe('FLOAT')
    expect(floatInput.isOptional).toBe(true)
  })

  it('should handle combo input specs', () => {
    const plainObject = {
      optional: {
        comboInput: [[1, 2, 3], { default: 2 }]
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)
    expect(result.inputs['comboInput'].type).toBe('COMBO')
    expect(result.inputs['comboInput'].default).toBe(2)
    expect(result.inputs['comboInput'].options).toEqual([1, 2, 3])
  })

  it('should handle combo input specs (auto-default)', () => {
    const plainObject = {
      optional: {
        comboInput: [[1, 2, 3], {}]
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)
    expect(result.inputs['comboInput'].type).toBe('COMBO')
    // Should pick the first choice as default
    expect(result.inputs['comboInput'].options).toEqual([1, 2, 3])
  })

  it('should handle custom input specs', () => {
    const plainObject = {
      optional: {
        customInput: ['CUSTOM_TYPE', { default: 'custom value' }]
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)
    expect(result.inputs['customInput'].type).toBe('CUSTOM_TYPE')
    expect(result.inputs['customInput'].default).toBe('custom value')
  })

  it('should not transform hidden fields', () => {
    const plainObject = {
      hidden: {
        someHiddenValue: 42,
        anotherHiddenValue: { nested: 'object' }
      }
    } as ComfyNodeDefV1['input']

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)

    // @ts-expect-error fixme ts strict error
    expect(result.hidden).toEqual(plainObject.hidden)
    expect(result.hidden?.someHiddenValue).toBe(42)
    expect(result.hidden?.anotherHiddenValue).toEqual({ nested: 'object' })
  })

  it('should handle empty or undefined fields', () => {
    const plainObject = {}

    const nodeDef: ComfyNodeDefV1 = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: plainObject,
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    }

    const result = transformNodeDefV1ToV2(nodeDef)

    expect(result).toBeDefined()
    expect(result.inputs).toEqual({})
    expect(result.hidden).toBeUndefined()
  })
})

describe('ComfyNodeDefImpl', () => {
  it('should transform a basic node definition correctly', () => {
    const plainObject = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: {
        required: {
          intInput: ['INT', { min: 0, max: 100, default: 50 }]
        }
      },
      output: ['INT'],
      output_is_list: [false],
      output_name: ['intOutput'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)

    expect(result).toBeInstanceOf(ComfyNodeDefImpl)
    expect(result.name).toBe('TestNode')
    expect(result.display_name).toBe('Test Node')
    expect(result.category).toBe('Testing')
    expect(result.python_module).toBe('test_module')
    expect(result.description).toBe('A test node')
    expect(result.inputs).toBeDefined()
    expect(result.outputs).toEqual([
      {
        index: 0,
        name: 'intOutput',
        type: 'INT',
        is_list: false
      }
    ])
    expect(result.deprecated).toBe(false)
  })

  it('should transform a deprecated basic node definition', () => {
    const plainObject = {
      name: 'TestNode',
      display_name: 'Test Node',
      category: 'Testing',
      python_module: 'test_module',
      description: 'A test node',
      input: {
        required: {
          intInput: ['INT', { min: 0, max: 100, default: 50 }]
        }
      },
      output: ['INT'],
      output_is_list: [false],
      output_name: ['intOutput'],
      deprecated: true,
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)
    expect(result.deprecated).toBe(true)
  })

  // Legacy way of marking a node as deprecated
  it('should mark deprecated with empty category', () => {
    const plainObject = {
      name: 'TestNode',
      display_name: 'Test Node',
      // Empty category should be treated as deprecated
      category: '',
      python_module: 'test_module',
      description: 'A test node',
      input: {
        required: {
          intInput: ['INT', { min: 0, max: 100, default: 50 }]
        }
      },
      output: ['INT'],
      output_is_list: [false],
      output_name: ['intOutput'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)
    expect(result.deprecated).toBe(true)
  })

  it('should handle multiple outputs including COMBO type', () => {
    const plainObject = {
      name: 'MultiOutputNode',
      display_name: 'Multi Output Node',
      category: 'Advanced',
      python_module: 'advanced_module',
      description: 'A node with multiple outputs',
      input: {},
      output: ['STRING', ['COMBO', 'option1', 'option2'], 'FLOAT'],
      output_is_list: [true, false, false],
      output_name: ['stringOutput', 'comboOutput', 'floatOutput'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)

    expect(result.outputs).toEqual([
      {
        index: 0,
        name: 'stringOutput',
        type: 'STRING',
        is_list: true
      },
      {
        index: 1,
        name: 'comboOutput',
        type: 'COMBO',
        is_list: false,
        options: ['COMBO', 'option1', 'option2']
      },
      {
        index: 2,
        name: 'floatOutput',
        type: 'FLOAT',
        is_list: false
      }
    ])
  })

  it('should use index for output names if matches type', () => {
    const plainObject = {
      name: 'MissingNamesNode',
      display_name: 'Missing Names Node',
      category: 'Test',
      python_module: 'test_module',
      description: 'A node with missing output names',
      input: {},
      output: ['INT', 'FLOAT', 'FLOAT'],
      output_is_list: [false, true, true],
      output_name: ['INT', 'FLOAT', 'FLOAT'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)

    expect(result.outputs).toEqual([
      {
        index: 0,
        name: 'INT',
        type: 'INT',
        is_list: false
      },
      {
        index: 1,
        name: 'FLOAT',
        type: 'FLOAT',
        is_list: true
      },
      {
        index: 2,
        name: 'FLOAT',
        type: 'FLOAT',
        is_list: true
      }
    ])
  })

  it('should handle duplicate output names', () => {
    const plainObject = {
      name: 'DuplicateOutputNode',
      display_name: 'Duplicate Output Node',
      category: 'Test',
      python_module: 'test_module',
      description: 'A node with duplicate output names',
      input: {},
      output: ['INT', 'FLOAT', 'STRING'],
      output_is_list: [false, false, false],
      output_name: ['output', 'output', 'uniqueOutput'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)
    expect(result.outputs).toEqual([
      {
        index: 0,
        name: 'output',
        type: 'INT',
        is_list: false
      },
      {
        index: 1,
        name: 'output',
        type: 'FLOAT',
        is_list: false
      },
      {
        index: 2,
        name: 'uniqueOutput',
        type: 'STRING',
        is_list: false
      }
    ])
  })

  it('should handle empty output', () => {
    const plainObject = {
      name: 'EmptyOutputNode',
      display_name: 'Empty Output Node',
      category: 'Test',
      python_module: 'test_module',
      description: 'A node with no outputs',
      input: {},
      output: [],
      output_is_list: [],
      output_name: [],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)

    expect(result.outputs).toEqual([])
  })

  it('should handle undefined fields', () => {
    const plainObject = {
      name: 'EmptyOutputNode',
      display_name: 'Empty Output Node',
      category: 'Test',
      python_module: 'test_module',
      description: 'A node with no outputs',
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)
    expect(result.outputs).toEqual([])
    expect(Object.keys(result.inputs)).toHaveLength(0)
  })

  it('should handle complex input specifications', () => {
    const plainObject = {
      name: 'ComplexInputNode',
      display_name: 'Complex Input Node',
      category: 'Advanced',
      python_module: 'advanced_module',
      description: 'A node with complex input',
      input: {
        required: {
          intInput: ['INT', { min: 0, max: 100, default: 50 }],
          stringInput: ['STRING', { multiline: true }]
        },
        optional: {
          booleanInput: ['BOOLEAN', { default: true }],
          floatInput: ['FLOAT', { step: 0.1 }]
        }
      },
      output: ['INT'],
      output_is_list: [false],
      output_name: ['result'],
      output_node: false
    } as ComfyNodeDefV1

    const result = new ComfyNodeDefImpl(plainObject)

    expect(result.inputs).toBeDefined()
    expect(Object.keys(result.inputs)).toHaveLength(4)
    expect(result.inputs['intInput']).toBeDefined()
    expect(result.inputs['stringInput']).toBeDefined()
    expect(result.inputs['booleanInput']).toBeDefined()
    expect(result.inputs['floatInput']).toBeDefined()
  })

  it.each([
    { api_node: true, expected: true },
    { api_node: false, expected: false },
    { api_node: undefined, expected: false }
  ] as { api_node: boolean | undefined; expected: boolean }[])(
    'should handle api_node field: $api_node',
    ({ api_node, expected }) => {
      const result = new ComfyNodeDefImpl({
        name: 'ApiNode',
        display_name: 'API Node',
        category: 'Test',
        python_module: 'test_module',
        description: 'A node with API',
        api_node
      } as ComfyNodeDefV1)
      expect(result.api_node).toBe(expected)
    }
  )
})
