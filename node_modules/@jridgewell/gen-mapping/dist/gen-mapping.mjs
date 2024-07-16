import { SetArray, put, remove } from '@jridgewell/set-array';
import { encode } from '@jridgewell/sourcemap-codec';
import { TraceMap, decodedMappings } from '@jridgewell/trace-mapping';

const COLUMN = 0;
const SOURCES_INDEX = 1;
const SOURCE_LINE = 2;
const SOURCE_COLUMN = 3;
const NAMES_INDEX = 4;

const NO_NAME = -1;
/**
 * Provides the state to generate a sourcemap.
 */
class GenMapping {
    constructor({ file, sourceRoot } = {}) {
        this._names = new SetArray();
        this._sources = new SetArray();
        this._sourcesContent = [];
        this._mappings = [];
        this.file = file;
        this.sourceRoot = sourceRoot;
        this._ignoreList = new SetArray();
    }
}
/**
 * Typescript doesn't allow friend access to private fields, so this just casts the map into a type
 * with public access modifiers.
 */
function cast(map) {
    return map;
}
function addSegment(map, genLine, genColumn, source, sourceLine, sourceColumn, name, content) {
    return addSegmentInternal(false, map, genLine, genColumn, source, sourceLine, sourceColumn, name, content);
}
function addMapping(map, mapping) {
    return addMappingInternal(false, map, mapping);
}
/**
 * Same as `addSegment`, but will only add the segment if it generates useful information in the
 * resulting map. This only works correctly if segments are added **in order**, meaning you should
 * not add a segment with a lower generated line/column than one that came before.
 */
const maybeAddSegment = (map, genLine, genColumn, source, sourceLine, sourceColumn, name, content) => {
    return addSegmentInternal(true, map, genLine, genColumn, source, sourceLine, sourceColumn, name, content);
};
/**
 * Same as `addMapping`, but will only add the mapping if it generates useful information in the
 * resulting map. This only works correctly if mappings are added **in order**, meaning you should
 * not add a mapping with a lower generated line/column than one that came before.
 */
const maybeAddMapping = (map, mapping) => {
    return addMappingInternal(true, map, mapping);
};
/**
 * Adds/removes the content of the source file to the source map.
 */
function setSourceContent(map, source, content) {
    const { _sources: sources, _sourcesContent: sourcesContent } = cast(map);
    const index = put(sources, source);
    sourcesContent[index] = content;
}
function setIgnore(map, source, ignore = true) {
    const { _sources: sources, _sourcesContent: sourcesContent, _ignoreList: ignoreList } = cast(map);
    const index = put(sources, source);
    if (index === sourcesContent.length)
        sourcesContent[index] = null;
    if (ignore)
        put(ignoreList, index);
    else
        remove(ignoreList, index);
}
/**
 * Returns a sourcemap object (with decoded mappings) suitable for passing to a library that expects
 * a sourcemap, or to JSON.stringify.
 */
function toDecodedMap(map) {
    const { _mappings: mappings, _sources: sources, _sourcesContent: sourcesContent, _names: names, _ignoreList: ignoreList, } = cast(map);
    removeEmptyFinalLines(mappings);
    return {
        version: 3,
        file: map.file || undefined,
        names: names.array,
        sourceRoot: map.sourceRoot || undefined,
        sources: sources.array,
        sourcesContent,
        mappings,
        ignoreList: ignoreList.array,
    };
}
/**
 * Returns a sourcemap object (with encoded mappings) suitable for passing to a library that expects
 * a sourcemap, or to JSON.stringify.
 */
function toEncodedMap(map) {
    const decoded = toDecodedMap(map);
    return Object.assign(Object.assign({}, decoded), { mappings: encode(decoded.mappings) });
}
/**
 * Constructs a new GenMapping, using the already present mappings of the input.
 */
function fromMap(input) {
    const map = new TraceMap(input);
    const gen = new GenMapping({ file: map.file, sourceRoot: map.sourceRoot });
    putAll(cast(gen)._names, map.names);
    putAll(cast(gen)._sources, map.sources);
    cast(gen)._sourcesContent = map.sourcesContent || map.sources.map(() => null);
    cast(gen)._mappings = decodedMappings(map);
    if (map.ignoreList)
        putAll(cast(gen)._ignoreList, map.ignoreList);
    return gen;
}
/**
 * Returns an array of high-level mapping objects for every recorded segment, which could then be
 * passed to the `source-map` library.
 */
function allMappings(map) {
    const out = [];
    const { _mappings: mappings, _sources: sources, _names: names } = cast(map);
    for (let i = 0; i < mappings.length; i++) {
        const line = mappings[i];
        for (let j = 0; j < line.length; j++) {
            const seg = line[j];
            const generated = { line: i + 1, column: seg[COLUMN] };
            let source = undefined;
            let original = undefined;
            let name = undefined;
            if (seg.length !== 1) {
                source = sources.array[seg[SOURCES_INDEX]];
                original = { line: seg[SOURCE_LINE] + 1, column: seg[SOURCE_COLUMN] };
                if (seg.length === 5)
                    name = names.array[seg[NAMES_INDEX]];
            }
            out.push({ generated, source, original, name });
        }
    }
    return out;
}
// This split declaration is only so that terser can elminiate the static initialization block.
function addSegmentInternal(skipable, map, genLine, genColumn, source, sourceLine, sourceColumn, name, content) {
    const { _mappings: mappings, _sources: sources, _sourcesContent: sourcesContent, _names: names, } = cast(map);
    const line = getLine(mappings, genLine);
    const index = getColumnIndex(line, genColumn);
    if (!source) {
        if (skipable && skipSourceless(line, index))
            return;
        return insert(line, index, [genColumn]);
    }
    const sourcesIndex = put(sources, source);
    const namesIndex = name ? put(names, name) : NO_NAME;
    if (sourcesIndex === sourcesContent.length)
        sourcesContent[sourcesIndex] = content !== null && content !== void 0 ? content : null;
    if (skipable && skipSource(line, index, sourcesIndex, sourceLine, sourceColumn, namesIndex)) {
        return;
    }
    return insert(line, index, name
        ? [genColumn, sourcesIndex, sourceLine, sourceColumn, namesIndex]
        : [genColumn, sourcesIndex, sourceLine, sourceColumn]);
}
function getLine(mappings, index) {
    for (let i = mappings.length; i <= index; i++) {
        mappings[i] = [];
    }
    return mappings[index];
}
function getColumnIndex(line, genColumn) {
    let index = line.length;
    for (let i = index - 1; i >= 0; index = i--) {
        const current = line[i];
        if (genColumn >= current[COLUMN])
            break;
    }
    return index;
}
function insert(array, index, value) {
    for (let i = array.length; i > index; i--) {
        array[i] = array[i - 1];
    }
    array[index] = value;
}
function removeEmptyFinalLines(mappings) {
    const { length } = mappings;
    let len = length;
    for (let i = len - 1; i >= 0; len = i, i--) {
        if (mappings[i].length > 0)
            break;
    }
    if (len < length)
        mappings.length = len;
}
function putAll(setarr, array) {
    for (let i = 0; i < array.length; i++)
        put(setarr, array[i]);
}
function skipSourceless(line, index) {
    // The start of a line is already sourceless, so adding a sourceless segment to the beginning
    // doesn't generate any useful information.
    if (index === 0)
        return true;
    const prev = line[index - 1];
    // If the previous segment is also sourceless, then adding another sourceless segment doesn't
    // genrate any new information. Else, this segment will end the source/named segment and point to
    // a sourceless position, which is useful.
    return prev.length === 1;
}
function skipSource(line, index, sourcesIndex, sourceLine, sourceColumn, namesIndex) {
    // A source/named segment at the start of a line gives position at that genColumn
    if (index === 0)
        return false;
    const prev = line[index - 1];
    // If the previous segment is sourceless, then we're transitioning to a source.
    if (prev.length === 1)
        return false;
    // If the previous segment maps to the exact same source position, then this segment doesn't
    // provide any new position information.
    return (sourcesIndex === prev[SOURCES_INDEX] &&
        sourceLine === prev[SOURCE_LINE] &&
        sourceColumn === prev[SOURCE_COLUMN] &&
        namesIndex === (prev.length === 5 ? prev[NAMES_INDEX] : NO_NAME));
}
function addMappingInternal(skipable, map, mapping) {
    const { generated, source, original, name, content } = mapping;
    if (!source) {
        return addSegmentInternal(skipable, map, generated.line - 1, generated.column, null, null, null, null, null);
    }
    return addSegmentInternal(skipable, map, generated.line - 1, generated.column, source, original.line - 1, original.column, name, content);
}

export { GenMapping, addMapping, addSegment, allMappings, fromMap, maybeAddMapping, maybeAddSegment, setIgnore, setSourceContent, toDecodedMap, toEncodedMap };
//# sourceMappingURL=gen-mapping.mjs.map
