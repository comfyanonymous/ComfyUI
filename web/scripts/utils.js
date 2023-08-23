export function range(size, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}

export function hook(klass, fnName, cb) {
    const orig = location[fnName];
    location[fnName] = (...args) => {
        cb(orig, ...args);
    }
}
