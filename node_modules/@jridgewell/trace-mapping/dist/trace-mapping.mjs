import { encode, decode } from '@jridgewell/sourcemap-codec';
import resolveUri from '@jridgewell/resolve-uri';

function resolve(input, base) {
    // The base is always treated as a directory, if it's not empty.
    // https://github.com/mozilla/source-map/blob/8cb3ee57/lib/util.js#L327
    // https://github.com/chromium/chromium/blob/da4adbb3/third_party/blink/renderer/devtools/front_end/sdk/SourceMap.js#L400-L401
    if (base && !base.endsWith('/'))
        base += '/';
    return resolveUri(input, base);
}

/**
 * Removes everything after the last "/", but leaves the slash.
 */
function stripFilename(path) {
    if (!path)
        return '';
    const index = path.lastIndexOf('/');
    return path.slice(0, index + 1);
}

const COLUMN = 0;
const SOURCES_INDEX = 1;
const SOURCE_LINE = 2;
const SOURCE_COLUMN = 3;
const NAMES_INDEX = 4;
const REV_GENERATED_LINE = 1;
const REV_GENERATED_COLUMN = 2;

function maybeSort(mappings, owned) {
    const unsortedIndex = nextUnsortedSegmentLine(mappings, 0);
    if (unsortedIndex === mappings.length)
        return mappings;
    // If we own the array (meaning we parsed it from JSON), then we're free to directly mutate it. If
    // not, we do not want to modify the consumer's input array.
    if (!owned)
        mappings = mappings.slice();
    for (let i = unsortedIndex; i < mappings.length; i = nextUnsortedSegmentLine(mappings, i + 1)) {
        mappings[i] = sortSegments(mappings[i], owned);
    }
    return mappings;
}
function nextUnsortedSegmentLine(mappings, start) {
    for (let i = start; i < mappings.length; i++) {
        if (!isSorted(mappings[i]))
            return i;
    }
    return mappings.length;
}
function isSorted(line) {
    for (let j = 1; j < line.length; j++) {
        if (line[j][COLUMN] < line[j - 1][COLUMN]) {
            return false;
        }
    }
    return true;
}
function sortSegments(line, owned) {
    if (!owned)
        line = line.slice();
    return line.sort(sortComparator);
}
function sortComparator(a, b) {
    return a[COLUMN] - b[COLUMN];
}

let found = false;
/**
 * A binary search implementation that returns the index if a match is found.
 * If no match is found, then the left-index (the index associated with the item that comes just
 * before the desired index) is returned. To maintain proper sort order, a splice would happen at
 * the next index:
 *
 * ```js
 * const array = [1, 3];
 * const needle = 2;
 * const index = binarySearch(array, needle, (item, needle) => item - needle);
 *
 * assert.equal(index, 0);
 * array.splice(index + 1, 0, needle);
 * assert.deepEqual(array, [1, 2, 3]);
 * ```
 */
function binarySearch(haystack, needle, low, high) {
    while (low <= high) {
        const mid = low + ((high - low) >> 1);
        const cmp = haystack[mid][COLUMN] - needle;
        if (cmp === 0) {
            found = true;
            return mid;
        }
        if (cmp < 0) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }
    found = false;
    return low - 1;
}
function upperBound(haystack, needle, index) {
    for (let i = index + 1; i < haystack.length; index = i++) {
        if (haystack[i][COLUMN] !== needle)
            break;
    }
    return index;
}
function lowerBound(haystack, needle, index) {
    for (let i = index - 1; i >= 0; index = i--) {
        if (haystack[i][COLUMN] !== needle)
            break;
    }
    return index;
}
function memoizedState() {
    return {
        lastKey: -1,
        lastNeedle: -1,
        lastIndex: -1,
    };
}
/**
 * This overly complicated beast is just to record the last tested line/column and the resulting
 * index, allowing us to skip a few tests if mappings are monotonically increasing.
 */
function memoizedBinarySearch(haystack, needle, state, key) {
    const { lastKey, lastNeedle, lastIndex } = state;
    let low = 0;
    let high = haystack.length - 1;
    if (key === lastKey) {
        if (needle === lastNeedle) {
            found = lastIndex !== -1 && haystack[lastIndex][COLUMN] === needle;
            return lastIndex;
        }
        if (needle >= lastNeedle) {
            // lastIndex may be -1 if the previous needle was not found.
            low = lastIndex === -1 ? 0 : lastIndex;
        }
        else {
            high = lastIndex;
        }
    }
    state.lastKey = key;
    state.lastNeedle = needle;
    return (state.lastIndex = binarySearch(haystack, needle, low, high));
}

// Rebuilds the original source files, with mappings that are ordered by source line/column instead
// of generated line/column.
function buildBySources(decoded, memos) {
    const sources = memos.map(buildNullArray);
    for (let i = 0; i < decoded.length; i++) {
        const line = decoded[i];
        for (let j = 0; j < line.length; j++) {
            const seg = line[j];
            if (seg.length === 1)
                continue;
            const sourceIndex = seg[SOURCES_INDEX];
            const sourceLine = seg[SOURCE_LINE];
            const sourceColumn = seg[SOURCE_COLUMN];
            const originalSource = sources[sourceIndex];
            const originalLine = (originalSource[sourceLine] || (originalSource[sourceLine] = []));
            const memo = memos[sourceIndex];
            // The binary search either found a match, or it found the left-index just before where the
            // segment should go. Either way, we want to insert after that. And there may be multiple
            // generated segments associated with an original location, so there may need to move several
            // indexes before we find where we need to insert.
            let index = upperBound(originalLine, sourceColumn, memoizedBinarySearch(originalLine, sourceColumn, memo, sourceLine));
            memo.lastIndex = ++index;
            insert(originalLine, index, [sourceColumn, i, seg[COLUMN]]);
        }
    }
    return sources;
}
function insert(array, index, value) {
    for (let i = array.length; i > index; i--) {
        array[i] = array[i - 1];
    }
    array[index] = value;
}
// Null arrays allow us to use ordered index keys without actually allocating contiguous memory like
// a real array. We use a null-prototype object to avoid prototype pollution and deoptimizations.
// Numeric properties on objects are magically sorted in ascending order by the engine regardless of
// the insertion order. So, by setting any numeric keys, even out of order, we'll get ascending
// order when iterating with for-in.
function buildNullArray() {
    return { __proto__: null };
}

const AnyMap = function (map, mapUrl) {
    const parsed = parse(map);
    if (!('sections' in parsed)) {
        return new TraceMap(parsed, mapUrl);
    }
    const mappings = [];
    const sources = [];
    const sourcesContent = [];
    const names = [];
    const ignoreList = [];
    recurse(parsed, mapUrl, mappings, sources, sourcesContent, names, ignoreList, 0, 0, Infinity, Infinity);
    const joined = {
        version: 3,
        file: parsed.file,
        names,
        sources,
        sourcesContent,
        mappings,
        ignoreList,
    };
    return presortedDecodedMap(joined);
};
function parse(map) {
    return typeof map === 'string' ? JSON.parse(map) : map;
}
function recurse(input, mapUrl, mappings, sources, sourcesContent, names, ignoreList, lineOffset, columnOffset, stopLine, stopColumn) {
    const { sections } = input;
    for (let i = 0; i < sections.length; i++) {
        const { map, offset } = sections[i];
        let sl = stopLine;
        let sc = stopColumn;
        if (i + 1 < sections.length) {
            const nextOffset = sections[i + 1].offset;
            sl = Math.min(stopLine, lineOffset + nextOffset.line);
            if (sl === stopLine) {
                sc = Math.min(stopColumn, columnOffset + nextOffset.column);
            }
            else if (sl < stopLine) {
                sc = columnOffset + nextOffset.column;
            }
        }
        addSection(map, mapUrl, mappings, sources, sourcesContent, names, ignoreList, lineOffset + offset.line, columnOffset + offset.column, sl, sc);
    }
}
function addSection(input, mapUrl, mappings, sources, sourcesContent, names, ignoreList, lineOffset, columnOffset, stopLine, stopColumn) {
    const parsed = parse(input);
    if ('sections' in parsed)
        return recurse(...arguments);
    const map = new TraceMap(parsed, mapUrl);
    const sourcesOffset = sources.length;
    const namesOffset = names.length;
    const decoded = decodedMappings(map);
    const { resolvedSources, sourcesContent: contents, ignoreList: ignores } = map;
    append(sources, resolvedSources);
    append(names, map.names);
    if (contents)
        append(sourcesContent, contents);
    else
        for (let i = 0; i < resolvedSources.length; i++)
            sourcesContent.push(null);
    if (ignores)
        for (let i = 0; i < ignores.length; i++)
            ignoreList.push(ignores[i] + sourcesOffset);
    for (let i = 0; i < decoded.length; i++) {
        const lineI = lineOffset + i;
        // We can only add so many lines before we step into the range that the next section's map
        // controls. When we get to the last line, then we'll start checking the segments to see if
        // they've crossed into the column range. But it may not have any columns that overstep, so we
        // still need to check that we don't overstep lines, too.
        if (lineI > stopLine)
            return;
        // The out line may already exist in mappings (if we're continuing the line started by a
        // previous section). Or, we may have jumped ahead several lines to start this section.
        const out = getLine(mappings, lineI);
        // On the 0th loop, the section's column offset shifts us forward. On all other lines (since the
        // map can be multiple lines), it doesn't.
        const cOffset = i === 0 ? columnOffset : 0;
        const line = decoded[i];
        for (let j = 0; j < line.length; j++) {
            const seg = line[j];
            const column = cOffset + seg[COLUMN];
            // If this segment steps into the column range that the next section's map controls, we need
            // to stop early.
            if (lineI === stopLine && column >= stopColumn)
                return;
            if (seg.length === 1) {
                out.push([column]);
                continue;
            }
            const sourcesIndex = sourcesOffset + seg[SOURCES_INDEX];
            const sourceLine = seg[SOURCE_LINE];
            const sourceColumn = seg[SOURCE_COLUMN];
            out.push(seg.length === 4
                ? [column, sourcesIndex, sourceLine, sourceColumn]
                : [column, sourcesIndex, sourceLine, sourceColumn, namesOffset + seg[NAMES_INDEX]]);
        }
    }
}
function append(arr, other) {
    for (let i = 0; i < other.length; i++)
        arr.push(other[i]);
}
function getLine(arr, index) {
    for (let i = arr.length; i <= index; i++)
        arr[i] = [];
    return arr[index];
}

const LINE_GTR_ZERO = '`line` must be greater than 0 (lines start at line 1)';
const COL_GTR_EQ_ZERO = '`column` must be greater than or equal to 0 (columns start at column 0)';
const LEAST_UPPER_BOUND = -1;
const GREATEST_LOWER_BOUND = 1;
class TraceMap {
    constructor(map, mapUrl) {
        const isString = typeof map === 'string';
        if (!isString && map._decodedMemo)
            return map;
        const parsed = (isString ? JSON.parse(map) : map);
        const { version, file, names, sourceRoot, sources, sourcesContent } = parsed;
        this.version = version;
        this.file = file;
        this.names = names || [];
        this.sourceRoot = sourceRoot;
        this.sources = sources;
        this.sourcesContent = sourcesContent;
        this.ignoreList = parsed.ignoreList || parsed.x_google_ignoreList || undefined;
        const from = resolve(sourceRoot || '', stripFilename(mapUrl));
        this.resolvedSources = sources.map((s) => resolve(s || '', from));
        const { mappings } = parsed;
        if (typeof mappings === 'string') {
            this._encoded = mappings;
            this._decoded = undefined;
        }
        else {
            this._encoded = undefined;
            this._decoded = maybeSort(mappings, isString);
        }
        this._decodedMemo = memoizedState();
        this._bySources = undefined;
        this._bySourceMemos = undefined;
    }
}
/**
 * Typescript doesn't allow friend access to private fields, so this just casts the map into a type
 * with public access modifiers.
 */
function cast(map) {
    return map;
}
/**
 * Returns the encoded (VLQ string) form of the SourceMap's mappings field.
 */
function encodedMappings(map) {
    var _a;
    var _b;
    return ((_a = (_b = cast(map))._encoded) !== null && _a !== void 0 ? _a : (_b._encoded = encode(cast(map)._decoded)));
}
/**
 * Returns the decoded (array of lines of segments) form of the SourceMap's mappings field.
 */
function decodedMappings(map) {
    var _a;
    return ((_a = cast(map))._decoded || (_a._decoded = decode(cast(map)._encoded)));
}
/**
 * A low-level API to find the segment associated with a generated line/column (think, from a
 * stack trace). Line and column here are 0-based, unlike `originalPositionFor`.
 */
function traceSegment(map, line, column) {
    const decoded = decodedMappings(map);
    // It's common for parent source maps to have pointers to lines that have no
    // mapping (like a "//# sourceMappingURL=") at the end of the child file.
    if (line >= decoded.length)
        return null;
    const segments = decoded[line];
    const index = traceSegmentInternal(segments, cast(map)._decodedMemo, line, column, GREATEST_LOWER_BOUND);
    return index === -1 ? null : segments[index];
}
/**
 * A higher-level API to find the source/line/column associated with a generated line/column
 * (think, from a stack trace). Line is 1-based, but column is 0-based, due to legacy behavior in
 * `source-map` library.
 */
function originalPositionFor(map, needle) {
    let { line, column, bias } = needle;
    line--;
    if (line < 0)
        throw new Error(LINE_GTR_ZERO);
    if (column < 0)
        throw new Error(COL_GTR_EQ_ZERO);
    const decoded = decodedMappings(map);
    // It's common for parent source maps to have pointers to lines that have no
    // mapping (like a "//# sourceMappingURL=") at the end of the child file.
    if (line >= decoded.length)
        return OMapping(null, null, null, null);
    const segments = decoded[line];
    const index = traceSegmentInternal(segments, cast(map)._decodedMemo, line, column, bias || GREATEST_LOWER_BOUND);
    if (index === -1)
        return OMapping(null, null, null, null);
    const segment = segments[index];
    if (segment.length === 1)
        return OMapping(null, null, null, null);
    const { names, resolvedSources } = map;
    return OMapping(resolvedSources[segment[SOURCES_INDEX]], segment[SOURCE_LINE] + 1, segment[SOURCE_COLUMN], segment.length === 5 ? names[segment[NAMES_INDEX]] : null);
}
/**
 * Finds the generated line/column position of the provided source/line/column source position.
 */
function generatedPositionFor(map, needle) {
    const { source, line, column, bias } = needle;
    return generatedPosition(map, source, line, column, bias || GREATEST_LOWER_BOUND, false);
}
/**
 * Finds all generated line/column positions of the provided source/line/column source position.
 */
function allGeneratedPositionsFor(map, needle) {
    const { source, line, column, bias } = needle;
    // SourceMapConsumer uses LEAST_UPPER_BOUND for some reason, so we follow suit.
    return generatedPosition(map, source, line, column, bias || LEAST_UPPER_BOUND, true);
}
/**
 * Iterates each mapping in generated position order.
 */
function eachMapping(map, cb) {
    const decoded = decodedMappings(map);
    const { names, resolvedSources } = map;
    for (let i = 0; i < decoded.length; i++) {
        const line = decoded[i];
        for (let j = 0; j < line.length; j++) {
            const seg = line[j];
            const generatedLine = i + 1;
            const generatedColumn = seg[0];
            let source = null;
            let originalLine = null;
            let originalColumn = null;
            let name = null;
            if (seg.length !== 1) {
                source = resolvedSources[seg[1]];
                originalLine = seg[2] + 1;
                originalColumn = seg[3];
            }
            if (seg.length === 5)
                name = names[seg[4]];
            cb({
                generatedLine,
                generatedColumn,
                source,
                originalLine,
                originalColumn,
                name,
            });
        }
    }
}
function sourceIndex(map, source) {
    const { sources, resolvedSources } = map;
    let index = sources.indexOf(source);
    if (index === -1)
        index = resolvedSources.indexOf(source);
    return index;
}
/**
 * Retrieves the source content for a particular source, if its found. Returns null if not.
 */
function sourceContentFor(map, source) {
    const { sourcesContent } = map;
    if (sourcesContent == null)
        return null;
    const index = sourceIndex(map, source);
    return index === -1 ? null : sourcesContent[index];
}
/**
 * Determines if the source is marked to ignore by the source map.
 */
function isIgnored(map, source) {
    const { ignoreList } = map;
    if (ignoreList == null)
        return false;
    const index = sourceIndex(map, source);
    return index === -1 ? false : ignoreList.includes(index);
}
/**
 * A helper that skips sorting of the input map's mappings array, which can be expensive for larger
 * maps.
 */
function presortedDecodedMap(map, mapUrl) {
    const tracer = new TraceMap(clone(map, []), mapUrl);
    cast(tracer)._decoded = map.mappings;
    return tracer;
}
/**
 * Returns a sourcemap object (with decoded mappings) suitable for passing to a library that expects
 * a sourcemap, or to JSON.stringify.
 */
function decodedMap(map) {
    return clone(map, decodedMappings(map));
}
/**
 * Returns a sourcemap object (with encoded mappings) suitable for passing to a library that expects
 * a sourcemap, or to JSON.stringify.
 */
function encodedMap(map) {
    return clone(map, encodedMappings(map));
}
function clone(map, mappings) {
    return {
        version: map.version,
        file: map.file,
        names: map.names,
        sourceRoot: map.sourceRoot,
        sources: map.sources,
        sourcesContent: map.sourcesContent,
        mappings,
        ignoreList: map.ignoreList || map.x_google_ignoreList,
    };
}
function OMapping(source, line, column, name) {
    return { source, line, column, name };
}
function GMapping(line, column) {
    return { line, column };
}
function traceSegmentInternal(segments, memo, line, column, bias) {
    let index = memoizedBinarySearch(segments, column, memo, line);
    if (found) {
        index = (bias === LEAST_UPPER_BOUND ? upperBound : lowerBound)(segments, column, index);
    }
    else if (bias === LEAST_UPPER_BOUND)
        index++;
    if (index === -1 || index === segments.length)
        return -1;
    return index;
}
function sliceGeneratedPositions(segments, memo, line, column, bias) {
    let min = traceSegmentInternal(segments, memo, line, column, GREATEST_LOWER_BOUND);
    // We ignored the bias when tracing the segment so that we're guarnateed to find the first (in
    // insertion order) segment that matched. Even if we did respect the bias when tracing, we would
    // still need to call `lowerBound()` to find the first segment, which is slower than just looking
    // for the GREATEST_LOWER_BOUND to begin with. The only difference that matters for us is when the
    // binary search didn't match, in which case GREATEST_LOWER_BOUND just needs to increment to
    // match LEAST_UPPER_BOUND.
    if (!found && bias === LEAST_UPPER_BOUND)
        min++;
    if (min === -1 || min === segments.length)
        return [];
    // We may have found the segment that started at an earlier column. If this is the case, then we
    // need to slice all generated segments that match _that_ column, because all such segments span
    // to our desired column.
    const matchedColumn = found ? column : segments[min][COLUMN];
    // The binary search is not guaranteed to find the lower bound when a match wasn't found.
    if (!found)
        min = lowerBound(segments, matchedColumn, min);
    const max = upperBound(segments, matchedColumn, min);
    const result = [];
    for (; min <= max; min++) {
        const segment = segments[min];
        result.push(GMapping(segment[REV_GENERATED_LINE] + 1, segment[REV_GENERATED_COLUMN]));
    }
    return result;
}
function generatedPosition(map, source, line, column, bias, all) {
    var _a;
    line--;
    if (line < 0)
        throw new Error(LINE_GTR_ZERO);
    if (column < 0)
        throw new Error(COL_GTR_EQ_ZERO);
    const { sources, resolvedSources } = map;
    let sourceIndex = sources.indexOf(source);
    if (sourceIndex === -1)
        sourceIndex = resolvedSources.indexOf(source);
    if (sourceIndex === -1)
        return all ? [] : GMapping(null, null);
    const generated = ((_a = cast(map))._bySources || (_a._bySources = buildBySources(decodedMappings(map), (cast(map)._bySourceMemos = sources.map(memoizedState)))));
    const segments = generated[sourceIndex][line];
    if (segments == null)
        return all ? [] : GMapping(null, null);
    const memo = cast(map)._bySourceMemos[sourceIndex];
    if (all)
        return sliceGeneratedPositions(segments, memo, line, column, bias);
    const index = traceSegmentInternal(segments, memo, line, column, bias);
    if (index === -1)
        return GMapping(null, null);
    const segment = segments[index];
    return GMapping(segment[REV_GENERATED_LINE] + 1, segment[REV_GENERATED_COLUMN]);
}

export { AnyMap, GREATEST_LOWER_BOUND, LEAST_UPPER_BOUND, TraceMap, allGeneratedPositionsFor, decodedMap, decodedMappings, eachMapping, encodedMap, encodedMappings, generatedPositionFor, isIgnored, originalPositionFor, presortedDecodedMap, sourceContentFor, traceSegment };
//# sourceMappingURL=trace-mapping.mjs.map
