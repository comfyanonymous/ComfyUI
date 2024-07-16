# @jridgewell/sourcemap-codec

Encode/decode the `mappings` property of a [sourcemap](https://docs.google.com/document/d/1U1RGAehQwRypUTovF1KRlpiOFze0b-_2gc6fAH0KY0k/edit).


## Why?

Sourcemaps are difficult to generate and manipulate, because the `mappings` property – the part that actually links the generated code back to the original source – is encoded using an obscure method called [Variable-length quantity](https://en.wikipedia.org/wiki/Variable-length_quantity). On top of that, each segment in the mapping contains offsets rather than absolute indices, which means that you can't look at a segment in isolation – you have to understand the whole sourcemap.

This package makes the process slightly easier.


## Installation

```bash
npm install @jridgewell/sourcemap-codec
```


## Usage

```js
import { encode, decode } from '@jridgewell/sourcemap-codec';

var decoded = decode( ';EAEEA,EAAE,EAAC,CAAE;ECQY,UACC' );

assert.deepEqual( decoded, [
	// the first line (of the generated code) has no mappings,
	// as shown by the starting semi-colon (which separates lines)
	[],

	// the second line contains four (comma-separated) segments
	[
		// segments are encoded as you'd expect:
		// [ generatedCodeColumn, sourceIndex, sourceCodeLine, sourceCodeColumn, nameIndex ]

		// i.e. the first segment begins at column 2, and maps back to the second column
		// of the second line (both zero-based) of the 0th source, and uses the 0th
		// name in the `map.names` array
		[ 2, 0, 2, 2, 0 ],

		// the remaining segments are 4-length rather than 5-length,
		// because they don't map a name
		[ 4, 0, 2, 4 ],
		[ 6, 0, 2, 5 ],
		[ 7, 0, 2, 7 ]
	],

	// the final line contains two segments
	[
		[ 2, 1, 10, 19 ],
		[ 12, 1, 11, 20 ]
	]
]);

var encoded = encode( decoded );
assert.equal( encoded, ';EAEEA,EAAE,EAAC,CAAE;ECQY,UACC' );
```

## Benchmarks

```
node v18.0.0

amp.js.map - 45120 segments

Decode Memory Usage:
@jridgewell/sourcemap-codec       5479160 bytes
sourcemap-codec                   5659336 bytes
source-map-0.6.1                 17144440 bytes
source-map-0.8.0                  6867424 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Decode speed:
decode: @jridgewell/sourcemap-codec x 502 ops/sec ±1.03% (90 runs sampled)
decode: sourcemap-codec x 445 ops/sec ±0.97% (92 runs sampled)
decode: source-map-0.6.1 x 36.01 ops/sec ±1.64% (49 runs sampled)
decode: source-map-0.8.0 x 367 ops/sec ±0.04% (95 runs sampled)
Fastest is decode: @jridgewell/sourcemap-codec

Encode Memory Usage:
@jridgewell/sourcemap-codec       1261620 bytes
sourcemap-codec                   9119248 bytes
source-map-0.6.1                  8968560 bytes
source-map-0.8.0                  8952952 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Encode speed:
encode: @jridgewell/sourcemap-codec x 738 ops/sec ±0.42% (98 runs sampled)
encode: sourcemap-codec x 238 ops/sec ±0.73% (88 runs sampled)
encode: source-map-0.6.1 x 162 ops/sec ±0.43% (84 runs sampled)
encode: source-map-0.8.0 x 191 ops/sec ±0.34% (90 runs sampled)
Fastest is encode: @jridgewell/sourcemap-codec


***


babel.min.js.map - 347793 segments

Decode Memory Usage:
@jridgewell/sourcemap-codec      35338184 bytes
sourcemap-codec                  35922736 bytes
source-map-0.6.1                 62366360 bytes
source-map-0.8.0                 44337416 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Decode speed:
decode: @jridgewell/sourcemap-codec x 40.35 ops/sec ±4.47% (54 runs sampled)
decode: sourcemap-codec x 36.76 ops/sec ±3.67% (51 runs sampled)
decode: source-map-0.6.1 x 4.44 ops/sec ±2.15% (16 runs sampled)
decode: source-map-0.8.0 x 59.35 ops/sec ±0.05% (78 runs sampled)
Fastest is decode: source-map-0.8.0

Encode Memory Usage:
@jridgewell/sourcemap-codec       7212604 bytes
sourcemap-codec                  21421456 bytes
source-map-0.6.1                 25286888 bytes
source-map-0.8.0                 25498744 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Encode speed:
encode: @jridgewell/sourcemap-codec x 112 ops/sec ±0.13% (84 runs sampled)
encode: sourcemap-codec x 30.23 ops/sec ±2.76% (53 runs sampled)
encode: source-map-0.6.1 x 19.43 ops/sec ±3.70% (37 runs sampled)
encode: source-map-0.8.0 x 19.40 ops/sec ±3.26% (37 runs sampled)
Fastest is encode: @jridgewell/sourcemap-codec


***


preact.js.map - 1992 segments

Decode Memory Usage:
@jridgewell/sourcemap-codec        500272 bytes
sourcemap-codec                    516864 bytes
source-map-0.6.1                  1596672 bytes
source-map-0.8.0                   517272 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Decode speed:
decode: @jridgewell/sourcemap-codec x 16,137 ops/sec ±0.17% (99 runs sampled)
decode: sourcemap-codec x 12,139 ops/sec ±0.13% (99 runs sampled)
decode: source-map-0.6.1 x 1,264 ops/sec ±0.12% (100 runs sampled)
decode: source-map-0.8.0 x 9,894 ops/sec ±0.08% (101 runs sampled)
Fastest is decode: @jridgewell/sourcemap-codec

Encode Memory Usage:
@jridgewell/sourcemap-codec        321026 bytes
sourcemap-codec                    830832 bytes
source-map-0.6.1                   586608 bytes
source-map-0.8.0                   586680 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Encode speed:
encode: @jridgewell/sourcemap-codec x 19,876 ops/sec ±0.78% (95 runs sampled)
encode: sourcemap-codec x 6,983 ops/sec ±0.15% (100 runs sampled)
encode: source-map-0.6.1 x 5,070 ops/sec ±0.12% (102 runs sampled)
encode: source-map-0.8.0 x 5,641 ops/sec ±0.17% (100 runs sampled)
Fastest is encode: @jridgewell/sourcemap-codec


***


react.js.map - 5726 segments

Decode Memory Usage:
@jridgewell/sourcemap-codec        734848 bytes
sourcemap-codec                    954200 bytes
source-map-0.6.1                  2276432 bytes
source-map-0.8.0                   955488 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Decode speed:
decode: @jridgewell/sourcemap-codec x 5,723 ops/sec ±0.12% (98 runs sampled)
decode: sourcemap-codec x 4,555 ops/sec ±0.09% (101 runs sampled)
decode: source-map-0.6.1 x 437 ops/sec ±0.11% (93 runs sampled)
decode: source-map-0.8.0 x 3,441 ops/sec ±0.15% (100 runs sampled)
Fastest is decode: @jridgewell/sourcemap-codec

Encode Memory Usage:
@jridgewell/sourcemap-codec        638672 bytes
sourcemap-codec                   1109840 bytes
source-map-0.6.1                  1321224 bytes
source-map-0.8.0                  1324448 bytes
Smallest memory usage is @jridgewell/sourcemap-codec

Encode speed:
encode: @jridgewell/sourcemap-codec x 6,801 ops/sec ±0.48% (98 runs sampled)
encode: sourcemap-codec x 2,533 ops/sec ±0.13% (101 runs sampled)
encode: source-map-0.6.1 x 2,248 ops/sec ±0.08% (100 runs sampled)
encode: source-map-0.8.0 x 2,303 ops/sec ±0.15% (100 runs sampled)
Fastest is encode: @jridgewell/sourcemap-codec
```

# License

MIT
