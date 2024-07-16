/**
 * SetArray acts like a `Set` (allowing only one occurrence of a string `key`), but provides the
 * index of the `key` in the backing array.
 *
 * This is designed to allow synchronizing a second array with the contents of the backing array,
 * like how in a sourcemap `sourcesContent[i]` is the source content associated with `source[i]`,
 * and there are never duplicates.
 */
class SetArray {
    constructor() {
        this._indexes = { __proto__: null };
        this.array = [];
    }
}
/**
 * Typescript doesn't allow friend access to private fields, so this just casts the set into a type
 * with public access modifiers.
 */
function cast(set) {
    return set;
}
/**
 * Gets the index associated with `key` in the backing array, if it is already present.
 */
function get(setarr, key) {
    return cast(setarr)._indexes[key];
}
/**
 * Puts `key` into the backing array, if it is not already present. Returns
 * the index of the `key` in the backing array.
 */
function put(setarr, key) {
    // The key may or may not be present. If it is present, it's a number.
    const index = get(setarr, key);
    if (index !== undefined)
        return index;
    const { array, _indexes: indexes } = cast(setarr);
    const length = array.push(key);
    return (indexes[key] = length - 1);
}
/**
 * Pops the last added item out of the SetArray.
 */
function pop(setarr) {
    const { array, _indexes: indexes } = cast(setarr);
    if (array.length === 0)
        return;
    const last = array.pop();
    indexes[last] = undefined;
}
/**
 * Removes the key, if it exists in the set.
 */
function remove(setarr, key) {
    const index = get(setarr, key);
    if (index === undefined)
        return;
    const { array, _indexes: indexes } = cast(setarr);
    for (let i = index + 1; i < array.length; i++) {
        const k = array[i];
        array[i - 1] = k;
        indexes[k]--;
    }
    indexes[key] = undefined;
    array.pop();
}

export { SetArray, get, pop, put, remove };
//# sourceMappingURL=set-array.mjs.map
