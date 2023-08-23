export function range(size, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}

function isClass(obj) {
  const isCtorClass = obj.constructor
      && obj.constructor.toString().substring(0, 5) === 'class'
  if(obj.prototype === undefined) {
    return isCtorClass
  }
  const isPrototypeCtorClass = obj.prototype.constructor
    && obj.prototype.constructor.toString
    && obj.prototype.constructor.toString().substring(0, 5) === 'class'
  return isCtorClass || isPrototypeCtorClass
}

export function hook(klass, fnName, cb) {
    let fnLocation = klass;
    if (isClass(klass)) {
        fnLocation = klass.prototype;
    }
    const orig = fnLocation[fnName];
    fnLocation[fnName] = function(...args) {
        return cb.bind(this)(orig, args);
    }
}
