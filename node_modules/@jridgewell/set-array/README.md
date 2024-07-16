# @jridgewell/set-array

> Like a Set, but provides the index of the `key` in the backing array

This is designed to allow synchronizing a second array with the contents of the backing array, like
how in a sourcemap `sourcesContent[i]` is the source content associated with `source[i]`, and there
are never duplicates.

## Installation

```sh
npm install @jridgewell/set-array
```

## Usage

```js
import { SetArray, get, put, pop } from '@jridgewell/set-array';

const sa = new SetArray();

let index = put(sa, 'first');
assert.strictEqual(index, 0);

index = put(sa, 'second');
assert.strictEqual(index, 1);

assert.deepEqual(sa.array, [ 'first', 'second' ]);

index = get(sa, 'first');
assert.strictEqual(index, 0);

pop(sa);
index = get(sa, 'second');
assert.strictEqual(index, undefined);
assert.deepEqual(sa.array, [ 'first' ]);
```
