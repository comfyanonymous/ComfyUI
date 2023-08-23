export function range(size, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}

export function hook(klass, fnName, cb) {
    const fnLocation = klass;
    const orig = fnLocation[fnName];
    fnLocation[fnName] = (...args) => {
        cb(orig, ...args);
    }
}

const a = {
    b: (c, d, e) => {
        console.log(c, d, e);
    }
}

a.b(1,2,3)
hook(a, "b", (orig, c, d, e) => {
    orig(c, d, e);
})
a.b(1,2,3)
