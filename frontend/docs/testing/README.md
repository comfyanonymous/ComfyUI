# ComfyUI Frontend Testing Guide

This guide provides an overview of testing approaches used in the ComfyUI Frontend codebase. These guides are meant to document any particularities or nuances of writing tests in this codebase, rather than being a comprehensive guide to testing in general. By reading these guides first, you may save yourself some time when encountering issues.

## Testing Documentation

Documentation for unit tests is organized into three guides:

- [Component Testing](./component-testing.md) - How to test Vue components
- [Unit Testing](./unit-testing.md) - How to test utility functions, composables, and other non-component code
- [Store Testing](./store-testing.md) - How to test Pinia stores specifically

## Testing Structure

The ComfyUI Frontend project uses **colocated tests** - test files are placed alongside their source files:

- **Component Tests**: Located directly alongside their components (e.g., `MyComponent.test.ts` next to `MyComponent.vue`)
- **Unit Tests**: Located alongside their source files (e.g., `myUtil.test.ts` next to `myUtil.ts`)
- **Store Tests**: Located in `src/stores/` alongside their store files
- **Browser Tests**: Located in the `browser_tests/` directory (see dedicated README there)

### Test File Naming

- Use `.test.ts` extension for test files
- Name tests after their source file: `sourceFile.test.ts`

## Test Frameworks and Libraries

Our tests use the following frameworks and libraries:

- [Vitest](https://vitest.dev/) - Test runner and assertion library
- [@vue/test-utils](https://test-utils.vuejs.org/) - Vue component testing utilities
- [Pinia](https://pinia.vuejs.org/cookbook/testing.html) - For store testing

## Getting Started

To run the tests locally:

```bash
# Run unit tests
pnpm test:unit

# Run a specific test file
pnpm test:unit -- src/path/to/file.test.ts

# Run unit tests in watch mode
pnpm test:unit -- --watch
```

Refer to the specific guides for more detailed information on each testing type.
