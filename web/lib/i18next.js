(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
        typeof define === 'function' && define.amd ? define(factory) :
            (global = typeof globalThis !== 'undefined' ? globalThis : global || self, global.i18next = factory());
})(this, (function () {
    'use strict';

    const consoleLogger = {
        type: 'logger',
        log(args) {
            this.output('log', args);
        },
        warn(args) {
            this.output('warn', args);
        },
        error(args) {
            this.output('error', args);
        },
        output(type, args) {
            if (console && console[type]) console[type].apply(console, args);
        }
    };
    class Logger {
        constructor(concreteLogger) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            this.init(concreteLogger, options);
        }
        init(concreteLogger) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            this.prefix = options.prefix || 'i18next:';
            this.logger = concreteLogger || consoleLogger;
            this.options = options;
            this.debug = options.debug;
        }
        log() {
            for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
                args[_key] = arguments[_key];
            }
            return this.forward(args, 'log', '', true);
        }
        warn() {
            for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
                args[_key2] = arguments[_key2];
            }
            return this.forward(args, 'warn', '', true);
        }
        error() {
            for (var _len3 = arguments.length, args = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
                args[_key3] = arguments[_key3];
            }
            return this.forward(args, 'error', '');
        }
        deprecate() {
            for (var _len4 = arguments.length, args = new Array(_len4), _key4 = 0; _key4 < _len4; _key4++) {
                args[_key4] = arguments[_key4];
            }
            return this.forward(args, 'warn', 'WARNING DEPRECATED: ', true);
        }
        forward(args, lvl, prefix, debugOnly) {
            if (debugOnly && !this.debug) return null;
            if (typeof args[0] === 'string') args[0] = `${prefix}${this.prefix} ${args[0]}`;
            return this.logger[lvl](args);
        }
        create(moduleName) {
            return new Logger(this.logger, {
                ...{
                    prefix: `${this.prefix}:${moduleName}:`
                },
                ...this.options
            });
        }
        clone(options) {
            options = options || this.options;
            options.prefix = options.prefix || this.prefix;
            return new Logger(this.logger, options);
        }
    }
    var baseLogger = new Logger();

    class EventEmitter {
        constructor() {
            this.observers = {};
        }
        on(events, listener) {
            events.split(' ').forEach(event => {
                this.observers[event] = this.observers[event] || [];
                this.observers[event].push(listener);
            });
            return this;
        }
        off(event, listener) {
            if (!this.observers[event]) return;
            if (!listener) {
                delete this.observers[event];
                return;
            }
            this.observers[event] = this.observers[event].filter(l => l !== listener);
        }
        emit(event) {
            for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
                args[_key - 1] = arguments[_key];
            }
            if (this.observers[event]) {
                const cloned = [].concat(this.observers[event]);
                cloned.forEach(observer => {
                    observer(...args);
                });
            }
            if (this.observers['*']) {
                const cloned = [].concat(this.observers['*']);
                cloned.forEach(observer => {
                    observer.apply(observer, [event, ...args]);
                });
            }
        }
    }

    function defer() {
        let res;
        let rej;
        const promise = new Promise((resolve, reject) => {
            res = resolve;
            rej = reject;
        });
        promise.resolve = res;
        promise.reject = rej;
        return promise;
    }
    function makeString(object) {
        if (object == null) return '';
        return '' + object;
    }
    function copy(a, s, t) {
        a.forEach(m => {
            if (s[m]) t[m] = s[m];
        });
    }
    function getLastOfPath(object, path, Empty) {
        function cleanKey(key) {
            return key && key.indexOf('###') > -1 ? key.replace(/###/g, '.') : key;
        }
        function canNotTraverseDeeper() {
            return !object || typeof object === 'string';
        }
        const stack = typeof path !== 'string' ? [].concat(path) : path.split('.');
        while (stack.length > 1) {
            if (canNotTraverseDeeper()) return {};
            const key = cleanKey(stack.shift());
            if (!object[key] && Empty) object[key] = new Empty();
            if (Object.prototype.hasOwnProperty.call(object, key)) {
                object = object[key];
            } else {
                object = {};
            }
        }
        if (canNotTraverseDeeper()) return {};
        return {
            obj: object,
            k: cleanKey(stack.shift())
        };
    }
    function setPath(object, path, newValue) {
        const {
            obj,
            k
        } = getLastOfPath(object, path, Object);
        obj[k] = newValue;
    }
    function pushPath(object, path, newValue, concat) {
        const {
            obj,
            k
        } = getLastOfPath(object, path, Object);
        obj[k] = obj[k] || [];
        if (concat) obj[k] = obj[k].concat(newValue);
        if (!concat) obj[k].push(newValue);
    }
    function getPath(object, path) {
        const {
            obj,
            k
        } = getLastOfPath(object, path);
        if (!obj) return undefined;
        return obj[k];
    }
    function getPathWithDefaults(data, defaultData, key) {
        const value = getPath(data, key);
        if (value !== undefined) {
            return value;
        }
        return getPath(defaultData, key);
    }
    function deepExtend(target, source, overwrite) {
        for (const prop in source) {
            if (prop !== '__proto__' && prop !== 'constructor') {
                if (prop in target) {
                    if (typeof target[prop] === 'string' || target[prop] instanceof String || typeof source[prop] === 'string' || source[prop] instanceof String) {
                        if (overwrite) target[prop] = source[prop];
                    } else {
                        deepExtend(target[prop], source[prop], overwrite);
                    }
                } else {
                    target[prop] = source[prop];
                }
            }
        }
        return target;
    }
    function regexEscape(str) {
        return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, '\\$&');
    }
    var _entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;'
    };
    function escape(data) {
        if (typeof data === 'string') {
            return data.replace(/[&<>"'\/]/g, s => _entityMap[s]);
        }
        return data;
    }
    const chars = [' ', ',', '?', '!', ';'];
    function looksLikeObjectPath(key, nsSeparator, keySeparator) {
        nsSeparator = nsSeparator || '';
        keySeparator = keySeparator || '';
        const possibleChars = chars.filter(c => nsSeparator.indexOf(c) < 0 && keySeparator.indexOf(c) < 0);
        if (possibleChars.length === 0) return true;
        const r = new RegExp(`(${possibleChars.map(c => c === '?' ? '\\?' : c).join('|')})`);
        let matched = !r.test(key);
        if (!matched) {
            const ki = key.indexOf(keySeparator);
            if (ki > 0 && !r.test(key.substring(0, ki))) {
                matched = true;
            }
        }
        return matched;
    }
    function deepFind(obj, path) {
        let keySeparator = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : '.';
        if (!obj) return undefined;
        if (obj[path]) return obj[path];
        const paths = path.split(keySeparator);
        let current = obj;
        for (let i = 0; i < paths.length; ++i) {
            if (!current) return undefined;
            if (typeof current[paths[i]] === 'string' && i + 1 < paths.length) {
                return undefined;
            }
            if (current[paths[i]] === undefined) {
                let j = 2;
                let p = paths.slice(i, i + j).join(keySeparator);
                let mix = current[p];
                while (mix === undefined && paths.length > i + j) {
                    j++;
                    p = paths.slice(i, i + j).join(keySeparator);
                    mix = current[p];
                }
                if (mix === undefined) return undefined;
                if (mix === null) return null;
                if (path.endsWith(p)) {
                    if (typeof mix === 'string') return mix;
                    if (p && typeof mix[p] === 'string') return mix[p];
                }
                const joinedPath = paths.slice(i + j).join(keySeparator);
                if (joinedPath) return deepFind(mix, joinedPath, keySeparator);
                return undefined;
            }
            current = current[paths[i]];
        }
        return current;
    }
    function getCleanedCode(code) {
        if (code && code.indexOf('_') > 0) return code.replace('_', '-');
        return code;
    }

    class ResourceStore extends EventEmitter {
        constructor(data) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {
                ns: ['translation'],
                defaultNS: 'translation'
            };
            super();
            this.data = data || {};
            this.options = options;
            if (this.options.keySeparator === undefined) {
                this.options.keySeparator = '.';
            }
            if (this.options.ignoreJSONStructure === undefined) {
                this.options.ignoreJSONStructure = true;
            }
        }
        addNamespaces(ns) {
            if (this.options.ns.indexOf(ns) < 0) {
                this.options.ns.push(ns);
            }
        }
        removeNamespaces(ns) {
            const index = this.options.ns.indexOf(ns);
            if (index > -1) {
                this.options.ns.splice(index, 1);
            }
        }
        getResource(lng, ns, key) {
            let options = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : {};
            const keySeparator = options.keySeparator !== undefined ? options.keySeparator : this.options.keySeparator;
            const ignoreJSONStructure = options.ignoreJSONStructure !== undefined ? options.ignoreJSONStructure : this.options.ignoreJSONStructure;
            let path = [lng, ns];
            if (key && typeof key !== 'string') path = path.concat(key);
            if (key && typeof key === 'string') path = path.concat(keySeparator ? key.split(keySeparator) : key);
            if (lng.indexOf('.') > -1) {
                path = lng.split('.');
            }
            const result = getPath(this.data, path);
            if (result || !ignoreJSONStructure || typeof key !== 'string') return result;
            return deepFind(this.data && this.data[lng] && this.data[lng][ns], key, keySeparator);
        }
        addResource(lng, ns, key, value) {
            let options = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : {
                silent: false
            };
            const keySeparator = options.keySeparator !== undefined ? options.keySeparator : this.options.keySeparator;
            let path = [lng, ns];
            if (key) path = path.concat(keySeparator ? key.split(keySeparator) : key);
            if (lng.indexOf('.') > -1) {
                path = lng.split('.');
                value = ns;
                ns = path[1];
            }
            this.addNamespaces(ns);
            setPath(this.data, path, value);
            if (!options.silent) this.emit('added', lng, ns, key, value);
        }
        addResources(lng, ns, resources) {
            let options = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : {
                silent: false
            };
            for (const m in resources) {
                if (typeof resources[m] === 'string' || Object.prototype.toString.apply(resources[m]) === '[object Array]') this.addResource(lng, ns, m, resources[m], {
                    silent: true
                });
            }
            if (!options.silent) this.emit('added', lng, ns, resources);
        }
        addResourceBundle(lng, ns, resources, deep, overwrite) {
            let options = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : {
                silent: false
            };
            let path = [lng, ns];
            if (lng.indexOf('.') > -1) {
                path = lng.split('.');
                deep = resources;
                resources = ns;
                ns = path[1];
            }
            this.addNamespaces(ns);
            let pack = getPath(this.data, path) || {};
            if (deep) {
                deepExtend(pack, resources, overwrite);
            } else {
                pack = {
                    ...pack,
                    ...resources
                };
            }
            setPath(this.data, path, pack);
            if (!options.silent) this.emit('added', lng, ns, resources);
        }
        removeResourceBundle(lng, ns) {
            if (this.hasResourceBundle(lng, ns)) {
                delete this.data[lng][ns];
            }
            this.removeNamespaces(ns);
            this.emit('removed', lng, ns);
        }
        hasResourceBundle(lng, ns) {
            return this.getResource(lng, ns) !== undefined;
        }
        getResourceBundle(lng, ns) {
            if (!ns) ns = this.options.defaultNS;
            if (this.options.compatibilityAPI === 'v1') return {
                ...{},
                ...this.getResource(lng, ns)
            };
            return this.getResource(lng, ns);
        }
        getDataByLanguage(lng) {
            return this.data[lng];
        }
        hasLanguageSomeTranslations(lng) {
            const data = this.getDataByLanguage(lng);
            const n = data && Object.keys(data) || [];
            return !!n.find(v => data[v] && Object.keys(data[v]).length > 0);
        }
        toJSON() {
            return this.data;
        }
    }

    var postProcessor = {
        processors: {},
        addPostProcessor(module) {
            this.processors[module.name] = module;
        },
        handle(processors, value, key, options, translator) {
            processors.forEach(processor => {
                if (this.processors[processor]) value = this.processors[processor].process(value, key, options, translator);
            });
            return value;
        }
    };

    const checkedLoadedFor = {};
    class Translator extends EventEmitter {
        constructor(services) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            super();
            copy(['resourceStore', 'languageUtils', 'pluralResolver', 'interpolator', 'backendConnector', 'i18nFormat', 'utils'], services, this);
            this.options = options;
            if (this.options.keySeparator === undefined) {
                this.options.keySeparator = '.';
            }
            this.logger = baseLogger.create('translator');
        }
        changeLanguage(lng) {
            if (lng) this.language = lng;
        }
        exists(key) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {
                interpolation: {}
            };
            if (key === undefined || key === null) {
                return false;
            }
            const resolved = this.resolve(key, options);
            return resolved && resolved.res !== undefined;
        }
        extractFromKey(key, options) {
            let nsSeparator = options.nsSeparator !== undefined ? options.nsSeparator : this.options.nsSeparator;
            if (nsSeparator === undefined) nsSeparator = ':';
            const keySeparator = options.keySeparator !== undefined ? options.keySeparator : this.options.keySeparator;
            let namespaces = options.ns || this.options.defaultNS || [];
            const wouldCheckForNsInKey = nsSeparator && key.indexOf(nsSeparator) > -1;
            const seemsNaturalLanguage = !this.options.userDefinedKeySeparator && !options.keySeparator && !this.options.userDefinedNsSeparator && !options.nsSeparator && !looksLikeObjectPath(key, nsSeparator, keySeparator);
            if (wouldCheckForNsInKey && !seemsNaturalLanguage) {
                const m = key.match(this.interpolator.nestingRegexp);
                if (m && m.length > 0) {
                    return {
                        key,
                        namespaces
                    };
                }
                const parts = key.split(nsSeparator);
                if (nsSeparator !== keySeparator || nsSeparator === keySeparator && this.options.ns.indexOf(parts[0]) > -1) namespaces = parts.shift();
                key = parts.join(keySeparator);
            }
            if (typeof namespaces === 'string') namespaces = [namespaces];
            return {
                key,
                namespaces
            };
        }
        translate(keys, options, lastKey) {
            if (typeof options !== 'object' && this.options.overloadTranslationOptionHandler) {
                options = this.options.overloadTranslationOptionHandler(arguments);
            }
            if (typeof options === 'object') options = {
                ...options
            };
            if (!options) options = {};
            if (keys === undefined || keys === null) return '';
            if (!Array.isArray(keys)) keys = [String(keys)];
            const returnDetails = options.returnDetails !== undefined ? options.returnDetails : this.options.returnDetails;
            const keySeparator = options.keySeparator !== undefined ? options.keySeparator : this.options.keySeparator;
            const {
                key,
                namespaces
            } = this.extractFromKey(keys[keys.length - 1], options);
            const namespace = namespaces[namespaces.length - 1];
            const lng = options.lng || this.language;
            const appendNamespaceToCIMode = options.appendNamespaceToCIMode || this.options.appendNamespaceToCIMode;
            if (lng && lng.toLowerCase() === 'cimode') {
                if (appendNamespaceToCIMode) {
                    const nsSeparator = options.nsSeparator || this.options.nsSeparator;
                    if (returnDetails) {
                        return {
                            res: `${namespace}${nsSeparator}${key}`,
                            usedKey: key,
                            exactUsedKey: key,
                            usedLng: lng,
                            usedNS: namespace
                        };
                    }
                    return `${namespace}${nsSeparator}${key}`;
                }
                if (returnDetails) {
                    return {
                        res: key,
                        usedKey: key,
                        exactUsedKey: key,
                        usedLng: lng,
                        usedNS: namespace
                    };
                }
                return key;
            }
            const resolved = this.resolve(keys, options);
            let res = resolved && resolved.res;
            const resUsedKey = resolved && resolved.usedKey || key;
            const resExactUsedKey = resolved && resolved.exactUsedKey || key;
            const resType = Object.prototype.toString.apply(res);
            const noObject = ['[object Number]', '[object Function]', '[object RegExp]'];
            const joinArrays = options.joinArrays !== undefined ? options.joinArrays : this.options.joinArrays;
            const handleAsObjectInI18nFormat = !this.i18nFormat || this.i18nFormat.handleAsObject;
            const handleAsObject = typeof res !== 'string' && typeof res !== 'boolean' && typeof res !== 'number';
            if (handleAsObjectInI18nFormat && res && handleAsObject && noObject.indexOf(resType) < 0 && !(typeof joinArrays === 'string' && resType === '[object Array]')) {
                if (!options.returnObjects && !this.options.returnObjects) {
                    if (!this.options.returnedObjectHandler) {
                        this.logger.warn('accessing an object - but returnObjects options is not enabled!');
                    }
                    const r = this.options.returnedObjectHandler ? this.options.returnedObjectHandler(resUsedKey, res, {
                        ...options,
                        ns: namespaces
                    }) : `key '${key} (${this.language})' returned an object instead of string.`;
                    if (returnDetails) {
                        resolved.res = r;
                        return resolved;
                    }
                    return r;
                }
                if (keySeparator) {
                    const resTypeIsArray = resType === '[object Array]';
                    const copy = resTypeIsArray ? [] : {};
                    const newKeyToUse = resTypeIsArray ? resExactUsedKey : resUsedKey;
                    for (const m in res) {
                        if (Object.prototype.hasOwnProperty.call(res, m)) {
                            const deepKey = `${newKeyToUse}${keySeparator}${m}`;
                            copy[m] = this.translate(deepKey, {
                                ...options,
                                ...{
                                    joinArrays: false,
                                    ns: namespaces
                                }
                            });
                            if (copy[m] === deepKey) copy[m] = res[m];
                        }
                    }
                    res = copy;
                }
            } else if (handleAsObjectInI18nFormat && typeof joinArrays === 'string' && resType === '[object Array]') {
                res = res.join(joinArrays);
                if (res) res = this.extendTranslation(res, keys, options, lastKey);
            } else {
                let usedDefault = false;
                let usedKey = false;
                const needsPluralHandling = options.count !== undefined && typeof options.count !== 'string';
                const hasDefaultValue = Translator.hasDefaultValue(options);
                const defaultValueSuffix = needsPluralHandling ? this.pluralResolver.getSuffix(lng, options.count, options) : '';
                const defaultValueSuffixOrdinalFallback = options.ordinal && needsPluralHandling ? this.pluralResolver.getSuffix(lng, options.count, {
                    ordinal: false
                }) : '';
                const defaultValue = options[`defaultValue${defaultValueSuffix}`] || options[`defaultValue${defaultValueSuffixOrdinalFallback}`] || options.defaultValue;
                if (!this.isValidLookup(res) && hasDefaultValue) {
                    usedDefault = true;
                    res = defaultValue;
                }
                if (!this.isValidLookup(res)) {
                    usedKey = true;
                    res = key;
                }
                const missingKeyNoValueFallbackToKey = options.missingKeyNoValueFallbackToKey || this.options.missingKeyNoValueFallbackToKey;
                const resForMissing = missingKeyNoValueFallbackToKey && usedKey ? undefined : res;
                const updateMissing = hasDefaultValue && defaultValue !== res && this.options.updateMissing;
                if (usedKey || usedDefault || updateMissing) {
                    this.logger.log(updateMissing ? 'updateKey' : 'missingKey', lng, namespace, key, updateMissing ? defaultValue : res);
                    if (keySeparator) {
                        const fk = this.resolve(key, {
                            ...options,
                            keySeparator: false
                        });
                        if (fk && fk.res) this.logger.warn('Seems the loaded translations were in flat JSON format instead of nested. Either set keySeparator: false on init or make sure your translations are published in nested format.');
                    }
                    let lngs = [];
                    const fallbackLngs = this.languageUtils.getFallbackCodes(this.options.fallbackLng, options.lng || this.language);
                    if (this.options.saveMissingTo === 'fallback' && fallbackLngs && fallbackLngs[0]) {
                        for (let i = 0; i < fallbackLngs.length; i++) {
                            lngs.push(fallbackLngs[i]);
                        }
                    } else if (this.options.saveMissingTo === 'all') {
                        lngs = this.languageUtils.toResolveHierarchy(options.lng || this.language);
                    } else {
                        lngs.push(options.lng || this.language);
                    }
                    const send = (l, k, specificDefaultValue) => {
                        const defaultForMissing = hasDefaultValue && specificDefaultValue !== res ? specificDefaultValue : resForMissing;
                        if (this.options.missingKeyHandler) {
                            this.options.missingKeyHandler(l, namespace, k, defaultForMissing, updateMissing, options);
                        } else if (this.backendConnector && this.backendConnector.saveMissing) {
                            this.backendConnector.saveMissing(l, namespace, k, defaultForMissing, updateMissing, options);
                        }
                        this.emit('missingKey', l, namespace, k, res);
                    };
                    if (this.options.saveMissing) {
                        if (this.options.saveMissingPlurals && needsPluralHandling) {
                            lngs.forEach(language => {
                                this.pluralResolver.getSuffixes(language, options).forEach(suffix => {
                                    send([language], key + suffix, options[`defaultValue${suffix}`] || defaultValue);
                                });
                            });
                        } else {
                            send(lngs, key, defaultValue);
                        }
                    }
                }
                res = this.extendTranslation(res, keys, options, resolved, lastKey);
                if (usedKey && res === key && this.options.appendNamespaceToMissingKey) res = `${namespace}:${key}`;
                if ((usedKey || usedDefault) && this.options.parseMissingKeyHandler) {
                    if (this.options.compatibilityAPI !== 'v1') {
                        res = this.options.parseMissingKeyHandler(this.options.appendNamespaceToMissingKey ? `${namespace}:${key}` : key, usedDefault ? res : undefined);
                    } else {
                        res = this.options.parseMissingKeyHandler(res);
                    }
                }
            }
            if (returnDetails) {
                resolved.res = res;
                return resolved;
            }
            return res;
        }
        extendTranslation(res, key, options, resolved, lastKey) {
            var _this = this;
            if (this.i18nFormat && this.i18nFormat.parse) {
                res = this.i18nFormat.parse(res, {
                    ...this.options.interpolation.defaultVariables,
                    ...options
                }, resolved.usedLng, resolved.usedNS, resolved.usedKey, {
                    resolved
                });
            } else if (!options.skipInterpolation) {
                if (options.interpolation) this.interpolator.init({
                    ...options,
                    ...{
                        interpolation: {
                            ...this.options.interpolation,
                            ...options.interpolation
                        }
                    }
                });
                const skipOnVariables = typeof res === 'string' && (options && options.interpolation && options.interpolation.skipOnVariables !== undefined ? options.interpolation.skipOnVariables : this.options.interpolation.skipOnVariables);
                let nestBef;
                if (skipOnVariables) {
                    const nb = res.match(this.interpolator.nestingRegexp);
                    nestBef = nb && nb.length;
                }
                let data = options.replace && typeof options.replace !== 'string' ? options.replace : options;
                if (this.options.interpolation.defaultVariables) data = {
                    ...this.options.interpolation.defaultVariables,
                    ...data
                };
                res = this.interpolator.interpolate(res, data, options.lng || this.language, options);
                if (skipOnVariables) {
                    const na = res.match(this.interpolator.nestingRegexp);
                    const nestAft = na && na.length;
                    if (nestBef < nestAft) options.nest = false;
                }
                if (!options.lng && this.options.compatibilityAPI !== 'v1' && resolved && resolved.res) options.lng = resolved.usedLng;
                if (options.nest !== false) res = this.interpolator.nest(res, function () {
                    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
                        args[_key] = arguments[_key];
                    }
                    if (lastKey && lastKey[0] === args[0] && !options.context) {
                        _this.logger.warn(`It seems you are nesting recursively key: ${args[0]} in key: ${key[0]}`);
                        return null;
                    }
                    return _this.translate(...args, key);
                }, options);
                if (options.interpolation) this.interpolator.reset();
            }
            const postProcess = options.postProcess || this.options.postProcess;
            const postProcessorNames = typeof postProcess === 'string' ? [postProcess] : postProcess;
            if (res !== undefined && res !== null && postProcessorNames && postProcessorNames.length && options.applyPostProcessor !== false) {
                res = postProcessor.handle(postProcessorNames, res, key, this.options && this.options.postProcessPassResolved ? {
                    i18nResolved: resolved,
                    ...options
                } : options, this);
            }
            return res;
        }
        resolve(keys) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            let found;
            let usedKey;
            let exactUsedKey;
            let usedLng;
            let usedNS;
            if (typeof keys === 'string') keys = [keys];
            keys.forEach(k => {
                if (this.isValidLookup(found)) return;
                const extracted = this.extractFromKey(k, options);
                const key = extracted.key;
                usedKey = key;
                let namespaces = extracted.namespaces;
                if (this.options.fallbackNS) namespaces = namespaces.concat(this.options.fallbackNS);
                const needsPluralHandling = options.count !== undefined && typeof options.count !== 'string';
                const needsZeroSuffixLookup = needsPluralHandling && !options.ordinal && options.count === 0 && this.pluralResolver.shouldUseIntlApi();
                const needsContextHandling = options.context !== undefined && (typeof options.context === 'string' || typeof options.context === 'number') && options.context !== '';
                const codes = options.lngs ? options.lngs : this.languageUtils.toResolveHierarchy(options.lng || this.language, options.fallbackLng);
                namespaces.forEach(ns => {
                    if (this.isValidLookup(found)) return;
                    usedNS = ns;
                    if (!checkedLoadedFor[`${codes[0]}-${ns}`] && this.utils && this.utils.hasLoadedNamespace && !this.utils.hasLoadedNamespace(usedNS)) {
                        checkedLoadedFor[`${codes[0]}-${ns}`] = true;
                        this.logger.warn(`key "${usedKey}" for languages "${codes.join(', ')}" won't get resolved as namespace "${usedNS}" was not yet loaded`, 'This means something IS WRONG in your setup. You access the t function before i18next.init / i18next.loadNamespace / i18next.changeLanguage was done. Wait for the callback or Promise to resolve before accessing it!!!');
                    }
                    codes.forEach(code => {
                        if (this.isValidLookup(found)) return;
                        usedLng = code;
                        const finalKeys = [key];
                        if (this.i18nFormat && this.i18nFormat.addLookupKeys) {
                            this.i18nFormat.addLookupKeys(finalKeys, key, code, ns, options);
                        } else {
                            let pluralSuffix;
                            if (needsPluralHandling) pluralSuffix = this.pluralResolver.getSuffix(code, options.count, options);
                            const zeroSuffix = `${this.options.pluralSeparator}zero`;
                            const ordinalPrefix = `${this.options.pluralSeparator}ordinal${this.options.pluralSeparator}`;
                            if (needsPluralHandling) {
                                finalKeys.push(key + pluralSuffix);
                                if (options.ordinal && pluralSuffix.indexOf(ordinalPrefix) === 0) {
                                    finalKeys.push(key + pluralSuffix.replace(ordinalPrefix, this.options.pluralSeparator));
                                }
                                if (needsZeroSuffixLookup) {
                                    finalKeys.push(key + zeroSuffix);
                                }
                            }
                            if (needsContextHandling) {
                                const contextKey = `${key}${this.options.contextSeparator}${options.context}`;
                                finalKeys.push(contextKey);
                                if (needsPluralHandling) {
                                    finalKeys.push(contextKey + pluralSuffix);
                                    if (options.ordinal && pluralSuffix.indexOf(ordinalPrefix) === 0) {
                                        finalKeys.push(contextKey + pluralSuffix.replace(ordinalPrefix, this.options.pluralSeparator));
                                    }
                                    if (needsZeroSuffixLookup) {
                                        finalKeys.push(contextKey + zeroSuffix);
                                    }
                                }
                            }
                        }
                        let possibleKey;
                        while (possibleKey = finalKeys.pop()) {
                            if (!this.isValidLookup(found)) {
                                exactUsedKey = possibleKey;
                                found = this.getResource(code, ns, possibleKey, options);
                            }
                        }
                    });
                });
            });
            return {
                res: found,
                usedKey,
                exactUsedKey,
                usedLng,
                usedNS
            };
        }
        isValidLookup(res) {
            return res !== undefined && !(!this.options.returnNull && res === null) && !(!this.options.returnEmptyString && res === '');
        }
        getResource(code, ns, key) {
            let options = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : {};
            if (this.i18nFormat && this.i18nFormat.getResource) return this.i18nFormat.getResource(code, ns, key, options);
            return this.resourceStore.getResource(code, ns, key, options);
        }
        static hasDefaultValue(options) {
            const prefix = 'defaultValue';
            for (const option in options) {
                if (Object.prototype.hasOwnProperty.call(options, option) && prefix === option.substring(0, prefix.length) && undefined !== options[option]) {
                    return true;
                }
            }
            return false;
        }
    }

    function capitalize(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    class LanguageUtil {
        constructor(options) {
            this.options = options;
            this.supportedLngs = this.options.supportedLngs || false;
            this.logger = baseLogger.create('languageUtils');
        }
        getScriptPartFromCode(code) {
            code = getCleanedCode(code);
            if (!code || code.indexOf('-') < 0) return null;
            const p = code.split('-');
            if (p.length === 2) return null;
            p.pop();
            if (p[p.length - 1].toLowerCase() === 'x') return null;
            return this.formatLanguageCode(p.join('-'));
        }
        getLanguagePartFromCode(code) {
            code = getCleanedCode(code);
            if (!code || code.indexOf('-') < 0) return code;
            const p = code.split('-');
            return this.formatLanguageCode(p[0]);
        }
        formatLanguageCode(code) {
            if (typeof code === 'string' && code.indexOf('-') > -1) {
                const specialCases = ['hans', 'hant', 'latn', 'cyrl', 'cans', 'mong', 'arab'];
                let p = code.split('-');
                if (this.options.lowerCaseLng) {
                    p = p.map(part => part.toLowerCase());
                } else if (p.length === 2) {
                    p[0] = p[0].toLowerCase();
                    p[1] = p[1].toUpperCase();
                    if (specialCases.indexOf(p[1].toLowerCase()) > -1) p[1] = capitalize(p[1].toLowerCase());
                } else if (p.length === 3) {
                    p[0] = p[0].toLowerCase();
                    if (p[1].length === 2) p[1] = p[1].toUpperCase();
                    if (p[0] !== 'sgn' && p[2].length === 2) p[2] = p[2].toUpperCase();
                    if (specialCases.indexOf(p[1].toLowerCase()) > -1) p[1] = capitalize(p[1].toLowerCase());
                    if (specialCases.indexOf(p[2].toLowerCase()) > -1) p[2] = capitalize(p[2].toLowerCase());
                }
                return p.join('-');
            }
            return this.options.cleanCode || this.options.lowerCaseLng ? code.toLowerCase() : code;
        }
        isSupportedCode(code) {
            if (this.options.load === 'languageOnly' || this.options.nonExplicitSupportedLngs) {
                code = this.getLanguagePartFromCode(code);
            }
            return !this.supportedLngs || !this.supportedLngs.length || this.supportedLngs.indexOf(code) > -1;
        }
        getBestMatchFromCodes(codes) {
            if (!codes) return null;
            let found;
            codes.forEach(code => {
                if (found) return;
                const cleanedLng = this.formatLanguageCode(code);
                if (!this.options.supportedLngs || this.isSupportedCode(cleanedLng)) found = cleanedLng;
            });
            if (!found && this.options.supportedLngs) {
                codes.forEach(code => {
                    if (found) return;
                    const lngOnly = this.getLanguagePartFromCode(code);
                    if (this.isSupportedCode(lngOnly)) return found = lngOnly;
                    found = this.options.supportedLngs.find(supportedLng => {
                        if (supportedLng === lngOnly) return supportedLng;
                        if (supportedLng.indexOf('-') < 0 && lngOnly.indexOf('-') < 0) return;
                        if (supportedLng.indexOf(lngOnly) === 0) return supportedLng;
                    });
                });
            }
            if (!found) found = this.getFallbackCodes(this.options.fallbackLng)[0];
            return found;
        }
        getFallbackCodes(fallbacks, code) {
            if (!fallbacks) return [];
            if (typeof fallbacks === 'function') fallbacks = fallbacks(code);
            if (typeof fallbacks === 'string') fallbacks = [fallbacks];
            if (Object.prototype.toString.apply(fallbacks) === '[object Array]') return fallbacks;
            if (!code) return fallbacks.default || [];
            let found = fallbacks[code];
            if (!found) found = fallbacks[this.getScriptPartFromCode(code)];
            if (!found) found = fallbacks[this.formatLanguageCode(code)];
            if (!found) found = fallbacks[this.getLanguagePartFromCode(code)];
            if (!found) found = fallbacks.default;
            return found || [];
        }
        toResolveHierarchy(code, fallbackCode) {
            const fallbackCodes = this.getFallbackCodes(fallbackCode || this.options.fallbackLng || [], code);
            const codes = [];
            const addCode = c => {
                if (!c) return;
                if (this.isSupportedCode(c)) {
                    codes.push(c);
                } else {
                    this.logger.warn(`rejecting language code not found in supportedLngs: ${c}`);
                }
            };
            if (typeof code === 'string' && (code.indexOf('-') > -1 || code.indexOf('_') > -1)) {
                if (this.options.load !== 'languageOnly') addCode(this.formatLanguageCode(code));
                if (this.options.load !== 'languageOnly' && this.options.load !== 'currentOnly') addCode(this.getScriptPartFromCode(code));
                if (this.options.load !== 'currentOnly') addCode(this.getLanguagePartFromCode(code));
            } else if (typeof code === 'string') {
                addCode(this.formatLanguageCode(code));
            }
            fallbackCodes.forEach(fc => {
                if (codes.indexOf(fc) < 0) addCode(this.formatLanguageCode(fc));
            });
            return codes;
        }
    }

    let sets = [{
        lngs: ['ach', 'ak', 'am', 'arn', 'br', 'fil', 'gun', 'ln', 'mfe', 'mg', 'mi', 'oc', 'pt', 'pt-BR', 'tg', 'tl', 'ti', 'tr', 'uz', 'wa'],
        nr: [1, 2],
        fc: 1
    }, {
        lngs: ['af', 'an', 'ast', 'az', 'bg', 'bn', 'ca', 'da', 'de', 'dev', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fi', 'fo', 'fur', 'fy', 'gl', 'gu', 'ha', 'hi', 'hu', 'hy', 'ia', 'it', 'kk', 'kn', 'ku', 'lb', 'mai', 'ml', 'mn', 'mr', 'nah', 'nap', 'nb', 'ne', 'nl', 'nn', 'no', 'nso', 'pa', 'pap', 'pms', 'ps', 'pt-PT', 'rm', 'sco', 'se', 'si', 'so', 'son', 'sq', 'sv', 'sw', 'ta', 'te', 'tk', 'ur', 'yo'],
        nr: [1, 2],
        fc: 2
    }, {
        lngs: ['ay', 'bo', 'cgg', 'fa', 'ht', 'id', 'ja', 'jbo', 'ka', 'km', 'ko', 'ky', 'lo', 'ms', 'sah', 'su', 'th', 'tt', 'ug', 'vi', 'wo', 'zh'],
        nr: [1],
        fc: 3
    }, {
        lngs: ['be', 'bs', 'cnr', 'dz', 'hr', 'ru', 'sr', 'uk'],
        nr: [1, 2, 5],
        fc: 4
    }, {
        lngs: ['ar'],
        nr: [0, 1, 2, 3, 11, 100],
        fc: 5
    }, {
        lngs: ['cs', 'sk'],
        nr: [1, 2, 5],
        fc: 6
    }, {
        lngs: ['csb', 'pl'],
        nr: [1, 2, 5],
        fc: 7
    }, {
        lngs: ['cy'],
        nr: [1, 2, 3, 8],
        fc: 8
    }, {
        lngs: ['fr'],
        nr: [1, 2],
        fc: 9
    }, {
        lngs: ['ga'],
        nr: [1, 2, 3, 7, 11],
        fc: 10
    }, {
        lngs: ['gd'],
        nr: [1, 2, 3, 20],
        fc: 11
    }, {
        lngs: ['is'],
        nr: [1, 2],
        fc: 12
    }, {
        lngs: ['jv'],
        nr: [0, 1],
        fc: 13
    }, {
        lngs: ['kw'],
        nr: [1, 2, 3, 4],
        fc: 14
    }, {
        lngs: ['lt'],
        nr: [1, 2, 10],
        fc: 15
    }, {
        lngs: ['lv'],
        nr: [1, 2, 0],
        fc: 16
    }, {
        lngs: ['mk'],
        nr: [1, 2],
        fc: 17
    }, {
        lngs: ['mnk'],
        nr: [0, 1, 2],
        fc: 18
    }, {
        lngs: ['mt'],
        nr: [1, 2, 11, 20],
        fc: 19
    }, {
        lngs: ['or'],
        nr: [2, 1],
        fc: 2
    }, {
        lngs: ['ro'],
        nr: [1, 2, 20],
        fc: 20
    }, {
        lngs: ['sl'],
        nr: [5, 1, 2, 3],
        fc: 21
    }, {
        lngs: ['he', 'iw'],
        nr: [1, 2, 20, 21],
        fc: 22
    }];
    let _rulesPluralsTypes = {
        1: function (n) {
            return Number(n > 1);
        },
        2: function (n) {
            return Number(n != 1);
        },
        3: function (n) {
            return 0;
        },
        4: function (n) {
            return Number(n % 10 == 1 && n % 100 != 11 ? 0 : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 10 || n % 100 >= 20) ? 1 : 2);
        },
        5: function (n) {
            return Number(n == 0 ? 0 : n == 1 ? 1 : n == 2 ? 2 : n % 100 >= 3 && n % 100 <= 10 ? 3 : n % 100 >= 11 ? 4 : 5);
        },
        6: function (n) {
            return Number(n == 1 ? 0 : n >= 2 && n <= 4 ? 1 : 2);
        },
        7: function (n) {
            return Number(n == 1 ? 0 : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 10 || n % 100 >= 20) ? 1 : 2);
        },
        8: function (n) {
            return Number(n == 1 ? 0 : n == 2 ? 1 : n != 8 && n != 11 ? 2 : 3);
        },
        9: function (n) {
            return Number(n >= 2);
        },
        10: function (n) {
            return Number(n == 1 ? 0 : n == 2 ? 1 : n < 7 ? 2 : n < 11 ? 3 : 4);
        },
        11: function (n) {
            return Number(n == 1 || n == 11 ? 0 : n == 2 || n == 12 ? 1 : n > 2 && n < 20 ? 2 : 3);
        },
        12: function (n) {
            return Number(n % 10 != 1 || n % 100 == 11);
        },
        13: function (n) {
            return Number(n !== 0);
        },
        14: function (n) {
            return Number(n == 1 ? 0 : n == 2 ? 1 : n == 3 ? 2 : 3);
        },
        15: function (n) {
            return Number(n % 10 == 1 && n % 100 != 11 ? 0 : n % 10 >= 2 && (n % 100 < 10 || n % 100 >= 20) ? 1 : 2);
        },
        16: function (n) {
            return Number(n % 10 == 1 && n % 100 != 11 ? 0 : n !== 0 ? 1 : 2);
        },
        17: function (n) {
            return Number(n == 1 || n % 10 == 1 && n % 100 != 11 ? 0 : 1);
        },
        18: function (n) {
            return Number(n == 0 ? 0 : n == 1 ? 1 : 2);
        },
        19: function (n) {
            return Number(n == 1 ? 0 : n == 0 || n % 100 > 1 && n % 100 < 11 ? 1 : n % 100 > 10 && n % 100 < 20 ? 2 : 3);
        },
        20: function (n) {
            return Number(n == 1 ? 0 : n == 0 || n % 100 > 0 && n % 100 < 20 ? 1 : 2);
        },
        21: function (n) {
            return Number(n % 100 == 1 ? 1 : n % 100 == 2 ? 2 : n % 100 == 3 || n % 100 == 4 ? 3 : 0);
        },
        22: function (n) {
            return Number(n == 1 ? 0 : n == 2 ? 1 : (n < 0 || n > 10) && n % 10 == 0 ? 2 : 3);
        }
    };
    const nonIntlVersions = ['v1', 'v2', 'v3'];
    const intlVersions = ['v4'];
    const suffixesOrder = {
        zero: 0,
        one: 1,
        two: 2,
        few: 3,
        many: 4,
        other: 5
    };
    function createRules() {
        const rules = {};
        sets.forEach(set => {
            set.lngs.forEach(l => {
                rules[l] = {
                    numbers: set.nr,
                    plurals: _rulesPluralsTypes[set.fc]
                };
            });
        });
        return rules;
    }
    class PluralResolver {
        constructor(languageUtils) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            this.languageUtils = languageUtils;
            this.options = options;
            this.logger = baseLogger.create('pluralResolver');
            if ((!this.options.compatibilityJSON || intlVersions.includes(this.options.compatibilityJSON)) && (typeof Intl === 'undefined' || !Intl.PluralRules)) {
                this.options.compatibilityJSON = 'v3';
                this.logger.error('Your environment seems not to be Intl API compatible, use an Intl.PluralRules polyfill. Will fallback to the compatibilityJSON v3 format handling.');
            }
            this.rules = createRules();
        }
        addRule(lng, obj) {
            this.rules[lng] = obj;
        }
        getRule(code) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            if (this.shouldUseIntlApi()) {
                try {
                    return new Intl.PluralRules(getCleanedCode(code), {
                        type: options.ordinal ? 'ordinal' : 'cardinal'
                    });
                } catch {
                    return;
                }
            }
            return this.rules[code] || this.rules[this.languageUtils.getLanguagePartFromCode(code)];
        }
        needsPlural(code) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            const rule = this.getRule(code, options);
            if (this.shouldUseIntlApi()) {
                return rule && rule.resolvedOptions().pluralCategories.length > 1;
            }
            return rule && rule.numbers.length > 1;
        }
        getPluralFormsOfKey(code, key) {
            let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
            return this.getSuffixes(code, options).map(suffix => `${key}${suffix}`);
        }
        getSuffixes(code) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            const rule = this.getRule(code, options);
            if (!rule) {
                return [];
            }
            if (this.shouldUseIntlApi()) {
                return rule.resolvedOptions().pluralCategories.sort((pluralCategory1, pluralCategory2) => suffixesOrder[pluralCategory1] - suffixesOrder[pluralCategory2]).map(pluralCategory => `${this.options.prepend}${options.ordinal ? `ordinal${this.options.prepend}` : ''}${pluralCategory}`);
            }
            return rule.numbers.map(number => this.getSuffix(code, number, options));
        }
        getSuffix(code, count) {
            let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
            const rule = this.getRule(code, options);
            if (rule) {
                if (this.shouldUseIntlApi()) {
                    return `${this.options.prepend}${options.ordinal ? `ordinal${this.options.prepend}` : ''}${rule.select(count)}`;
                }
                return this.getSuffixRetroCompatible(rule, count);
            }
            this.logger.warn(`no plural rule found for: ${code}`);
            return '';
        }
        getSuffixRetroCompatible(rule, count) {
            const idx = rule.noAbs ? rule.plurals(count) : rule.plurals(Math.abs(count));
            let suffix = rule.numbers[idx];
            if (this.options.simplifyPluralSuffix && rule.numbers.length === 2 && rule.numbers[0] === 1) {
                if (suffix === 2) {
                    suffix = 'plural';
                } else if (suffix === 1) {
                    suffix = '';
                }
            }
            const returnSuffix = () => this.options.prepend && suffix.toString() ? this.options.prepend + suffix.toString() : suffix.toString();
            if (this.options.compatibilityJSON === 'v1') {
                if (suffix === 1) return '';
                if (typeof suffix === 'number') return `_plural_${suffix.toString()}`;
                return returnSuffix();
            } else if (this.options.compatibilityJSON === 'v2') {
                return returnSuffix();
            } else if (this.options.simplifyPluralSuffix && rule.numbers.length === 2 && rule.numbers[0] === 1) {
                return returnSuffix();
            }
            return this.options.prepend && idx.toString() ? this.options.prepend + idx.toString() : idx.toString();
        }
        shouldUseIntlApi() {
            return !nonIntlVersions.includes(this.options.compatibilityJSON);
        }
    }

    function deepFindWithDefaults(data, defaultData, key) {
        let keySeparator = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : '.';
        let ignoreJSONStructure = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : true;
        let path = getPathWithDefaults(data, defaultData, key);
        if (!path && ignoreJSONStructure && typeof key === 'string') {
            path = deepFind(data, key, keySeparator);
            if (path === undefined) path = deepFind(defaultData, key, keySeparator);
        }
        return path;
    }
    class Interpolator {
        constructor() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            this.logger = baseLogger.create('interpolator');
            this.options = options;
            this.format = options.interpolation && options.interpolation.format || (value => value);
            this.init(options);
        }
        init() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            if (!options.interpolation) options.interpolation = {
                escapeValue: true
            };
            const iOpts = options.interpolation;
            this.escape = iOpts.escape !== undefined ? iOpts.escape : escape;
            this.escapeValue = iOpts.escapeValue !== undefined ? iOpts.escapeValue : true;
            this.useRawValueToEscape = iOpts.useRawValueToEscape !== undefined ? iOpts.useRawValueToEscape : false;
            this.prefix = iOpts.prefix ? regexEscape(iOpts.prefix) : iOpts.prefixEscaped || '{{';
            this.suffix = iOpts.suffix ? regexEscape(iOpts.suffix) : iOpts.suffixEscaped || '}}';
            this.formatSeparator = iOpts.formatSeparator ? iOpts.formatSeparator : iOpts.formatSeparator || ',';
            this.unescapePrefix = iOpts.unescapeSuffix ? '' : iOpts.unescapePrefix || '-';
            this.unescapeSuffix = this.unescapePrefix ? '' : iOpts.unescapeSuffix || '';
            this.nestingPrefix = iOpts.nestingPrefix ? regexEscape(iOpts.nestingPrefix) : iOpts.nestingPrefixEscaped || regexEscape('$t(');
            this.nestingSuffix = iOpts.nestingSuffix ? regexEscape(iOpts.nestingSuffix) : iOpts.nestingSuffixEscaped || regexEscape(')');
            this.nestingOptionsSeparator = iOpts.nestingOptionsSeparator ? iOpts.nestingOptionsSeparator : iOpts.nestingOptionsSeparator || ',';
            this.maxReplaces = iOpts.maxReplaces ? iOpts.maxReplaces : 1000;
            this.alwaysFormat = iOpts.alwaysFormat !== undefined ? iOpts.alwaysFormat : false;
            this.resetRegExp();
        }
        reset() {
            if (this.options) this.init(this.options);
        }
        resetRegExp() {
            const regexpStr = `${this.prefix}(.+?)${this.suffix}`;
            this.regexp = new RegExp(regexpStr, 'g');
            const regexpUnescapeStr = `${this.prefix}${this.unescapePrefix}(.+?)${this.unescapeSuffix}${this.suffix}`;
            this.regexpUnescape = new RegExp(regexpUnescapeStr, 'g');
            const nestingRegexpStr = `${this.nestingPrefix}(.+?)${this.nestingSuffix}`;
            this.nestingRegexp = new RegExp(nestingRegexpStr, 'g');
        }
        interpolate(str, data, lng, options) {
            let match;
            let value;
            let replaces;
            const defaultData = this.options && this.options.interpolation && this.options.interpolation.defaultVariables || {};
            function regexSafe(val) {
                return val.replace(/\$/g, '$$$$');
            }
            const handleFormat = key => {
                if (key.indexOf(this.formatSeparator) < 0) {
                    const path = deepFindWithDefaults(data, defaultData, key, this.options.keySeparator, this.options.ignoreJSONStructure);
                    return this.alwaysFormat ? this.format(path, undefined, lng, {
                        ...options,
                        ...data,
                        interpolationkey: key
                    }) : path;
                }
                const p = key.split(this.formatSeparator);
                const k = p.shift().trim();
                const f = p.join(this.formatSeparator).trim();
                return this.format(deepFindWithDefaults(data, defaultData, k, this.options.keySeparator, this.options.ignoreJSONStructure), f, lng, {
                    ...options,
                    ...data,
                    interpolationkey: k
                });
            };
            this.resetRegExp();
            const missingInterpolationHandler = options && options.missingInterpolationHandler || this.options.missingInterpolationHandler;
            const skipOnVariables = options && options.interpolation && options.interpolation.skipOnVariables !== undefined ? options.interpolation.skipOnVariables : this.options.interpolation.skipOnVariables;
            const todos = [{
                regex: this.regexpUnescape,
                safeValue: val => regexSafe(val)
            }, {
                regex: this.regexp,
                safeValue: val => this.escapeValue ? regexSafe(this.escape(val)) : regexSafe(val)
            }];
            todos.forEach(todo => {
                replaces = 0;
                while (match = todo.regex.exec(str)) {
                    const matchedVar = match[1].trim();
                    value = handleFormat(matchedVar);
                    if (value === undefined) {
                        if (typeof missingInterpolationHandler === 'function') {
                            const temp = missingInterpolationHandler(str, match, options);
                            value = typeof temp === 'string' ? temp : '';
                        } else if (options && Object.prototype.hasOwnProperty.call(options, matchedVar)) {
                            value = '';
                        } else if (skipOnVariables) {
                            value = match[0];
                            continue;
                        } else {
                            this.logger.warn(`missed to pass in variable ${matchedVar} for interpolating ${str}`);
                            value = '';
                        }
                    } else if (typeof value !== 'string' && !this.useRawValueToEscape) {
                        value = makeString(value);
                    }
                    const safeValue = todo.safeValue(value);
                    str = str.replace(match[0], safeValue);
                    if (skipOnVariables) {
                        todo.regex.lastIndex += value.length;
                        todo.regex.lastIndex -= match[0].length;
                    } else {
                        todo.regex.lastIndex = 0;
                    }
                    replaces++;
                    if (replaces >= this.maxReplaces) {
                        break;
                    }
                }
            });
            return str;
        }
        nest(str, fc) {
            let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
            let match;
            let value;
            let clonedOptions;
            function handleHasOptions(key, inheritedOptions) {
                const sep = this.nestingOptionsSeparator;
                if (key.indexOf(sep) < 0) return key;
                const c = key.split(new RegExp(`${sep}[ ]*{`));
                let optionsString = `{${c[1]}`;
                key = c[0];
                optionsString = this.interpolate(optionsString, clonedOptions);
                const matchedSingleQuotes = optionsString.match(/'/g);
                const matchedDoubleQuotes = optionsString.match(/"/g);
                if (matchedSingleQuotes && matchedSingleQuotes.length % 2 === 0 && !matchedDoubleQuotes || matchedDoubleQuotes.length % 2 !== 0) {
                    optionsString = optionsString.replace(/'/g, '"');
                }
                try {
                    clonedOptions = JSON.parse(optionsString);
                    if (inheritedOptions) clonedOptions = {
                        ...inheritedOptions,
                        ...clonedOptions
                    };
                } catch (e) {
                    this.logger.warn(`failed parsing options string in nesting for key ${key}`, e);
                    return `${key}${sep}${optionsString}`;
                }
                delete clonedOptions.defaultValue;
                return key;
            }
            while (match = this.nestingRegexp.exec(str)) {
                let formatters = [];
                clonedOptions = {
                    ...options
                };
                clonedOptions = clonedOptions.replace && typeof clonedOptions.replace !== 'string' ? clonedOptions.replace : clonedOptions;
                clonedOptions.applyPostProcessor = false;
                delete clonedOptions.defaultValue;
                let doReduce = false;
                if (match[0].indexOf(this.formatSeparator) !== -1 && !/{.*}/.test(match[1])) {
                    const r = match[1].split(this.formatSeparator).map(elem => elem.trim());
                    match[1] = r.shift();
                    formatters = r;
                    doReduce = true;
                }
                value = fc(handleHasOptions.call(this, match[1].trim(), clonedOptions), clonedOptions);
                if (value && match[0] === str && typeof value !== 'string') return value;
                if (typeof value !== 'string') value = makeString(value);
                if (!value) {
                    this.logger.warn(`missed to resolve ${match[1]} for nesting ${str}`);
                    value = '';
                }
                if (doReduce) {
                    value = formatters.reduce((v, f) => this.format(v, f, options.lng, {
                        ...options,
                        interpolationkey: match[1].trim()
                    }), value.trim());
                }
                str = str.replace(match[0], value);
                this.regexp.lastIndex = 0;
            }
            return str;
        }
    }

    function parseFormatStr(formatStr) {
        let formatName = formatStr.toLowerCase().trim();
        const formatOptions = {};
        if (formatStr.indexOf('(') > -1) {
            const p = formatStr.split('(');
            formatName = p[0].toLowerCase().trim();
            const optStr = p[1].substring(0, p[1].length - 1);
            if (formatName === 'currency' && optStr.indexOf(':') < 0) {
                if (!formatOptions.currency) formatOptions.currency = optStr.trim();
            } else if (formatName === 'relativetime' && optStr.indexOf(':') < 0) {
                if (!formatOptions.range) formatOptions.range = optStr.trim();
            } else {
                const opts = optStr.split(';');
                opts.forEach(opt => {
                    if (!opt) return;
                    const [key, ...rest] = opt.split(':');
                    const val = rest.join(':').trim().replace(/^'+|'+$/g, '');
                    if (!formatOptions[key.trim()]) formatOptions[key.trim()] = val;
                    if (val === 'false') formatOptions[key.trim()] = false;
                    if (val === 'true') formatOptions[key.trim()] = true;
                    if (!isNaN(val)) formatOptions[key.trim()] = parseInt(val, 10);
                });
            }
        }
        return {
            formatName,
            formatOptions
        };
    }
    function createCachedFormatter(fn) {
        const cache = {};
        return function invokeFormatter(val, lng, options) {
            const key = lng + JSON.stringify(options);
            let formatter = cache[key];
            if (!formatter) {
                formatter = fn(getCleanedCode(lng), options);
                cache[key] = formatter;
            }
            return formatter(val);
        };
    }
    class Formatter {
        constructor() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            this.logger = baseLogger.create('formatter');
            this.options = options;
            this.formats = {
                number: createCachedFormatter((lng, opt) => {
                    const formatter = new Intl.NumberFormat(lng, {
                        ...opt
                    });
                    return val => formatter.format(val);
                }),
                currency: createCachedFormatter((lng, opt) => {
                    const formatter = new Intl.NumberFormat(lng, {
                        ...opt,
                        style: 'currency'
                    });
                    return val => formatter.format(val);
                }),
                datetime: createCachedFormatter((lng, opt) => {
                    const formatter = new Intl.DateTimeFormat(lng, {
                        ...opt
                    });
                    return val => formatter.format(val);
                }),
                relativetime: createCachedFormatter((lng, opt) => {
                    const formatter = new Intl.RelativeTimeFormat(lng, {
                        ...opt
                    });
                    return val => formatter.format(val, opt.range || 'day');
                }),
                list: createCachedFormatter((lng, opt) => {
                    const formatter = new Intl.ListFormat(lng, {
                        ...opt
                    });
                    return val => formatter.format(val);
                })
            };
            this.init(options);
        }
        init(services) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {
                interpolation: {}
            };
            const iOpts = options.interpolation;
            this.formatSeparator = iOpts.formatSeparator ? iOpts.formatSeparator : iOpts.formatSeparator || ',';
        }
        add(name, fc) {
            this.formats[name.toLowerCase().trim()] = fc;
        }
        addCached(name, fc) {
            this.formats[name.toLowerCase().trim()] = createCachedFormatter(fc);
        }
        format(value, format, lng) {
            let options = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : {};
            const formats = format.split(this.formatSeparator);
            const result = formats.reduce((mem, f) => {
                const {
                    formatName,
                    formatOptions
                } = parseFormatStr(f);
                if (this.formats[formatName]) {
                    let formatted = mem;
                    try {
                        const valOptions = options && options.formatParams && options.formatParams[options.interpolationkey] || {};
                        const l = valOptions.locale || valOptions.lng || options.locale || options.lng || lng;
                        formatted = this.formats[formatName](mem, l, {
                            ...formatOptions,
                            ...options,
                            ...valOptions
                        });
                    } catch (error) {
                        this.logger.warn(error);
                    }
                    return formatted;
                } else {
                    this.logger.warn(`there was no format function for ${formatName}`);
                }
                return mem;
            }, value);
            return result;
        }
    }

    function removePending(q, name) {
        if (q.pending[name] !== undefined) {
            delete q.pending[name];
            q.pendingCount--;
        }
    }
    class Connector extends EventEmitter {
        constructor(backend, store, services) {
            let options = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : {};
            super();
            this.backend = backend;
            this.store = store;
            this.services = services;
            this.languageUtils = services.languageUtils;
            this.options = options;
            this.logger = baseLogger.create('backendConnector');
            this.waitingReads = [];
            this.maxParallelReads = options.maxParallelReads || 10;
            this.readingCalls = 0;
            this.maxRetries = options.maxRetries >= 0 ? options.maxRetries : 5;
            this.retryTimeout = options.retryTimeout >= 1 ? options.retryTimeout : 350;
            this.state = {};
            this.queue = [];
            if (this.backend && this.backend.init) {
                this.backend.init(services, options.backend, options);
            }
        }
        queueLoad(languages, namespaces, options, callback) {
            const toLoad = {};
            const pending = {};
            const toLoadLanguages = {};
            const toLoadNamespaces = {};
            languages.forEach(lng => {
                let hasAllNamespaces = true;
                namespaces.forEach(ns => {
                    const name = `${lng}|${ns}`;
                    if (!options.reload && this.store.hasResourceBundle(lng, ns)) {
                        this.state[name] = 2;
                    } else if (this.state[name] < 0); else if (this.state[name] === 1) {
                        if (pending[name] === undefined) pending[name] = true;
                    } else {
                        this.state[name] = 1;
                        hasAllNamespaces = false;
                        if (pending[name] === undefined) pending[name] = true;
                        if (toLoad[name] === undefined) toLoad[name] = true;
                        if (toLoadNamespaces[ns] === undefined) toLoadNamespaces[ns] = true;
                    }
                });
                if (!hasAllNamespaces) toLoadLanguages[lng] = true;
            });
            if (Object.keys(toLoad).length || Object.keys(pending).length) {
                this.queue.push({
                    pending,
                    pendingCount: Object.keys(pending).length,
                    loaded: {},
                    errors: [],
                    callback
                });
            }
            return {
                toLoad: Object.keys(toLoad),
                pending: Object.keys(pending),
                toLoadLanguages: Object.keys(toLoadLanguages),
                toLoadNamespaces: Object.keys(toLoadNamespaces)
            };
        }
        loaded(name, err, data) {
            const s = name.split('|');
            const lng = s[0];
            const ns = s[1];
            if (err) this.emit('failedLoading', lng, ns, err);
            if (data) {
                this.store.addResourceBundle(lng, ns, data);
            }
            this.state[name] = err ? -1 : 2;
            const loaded = {};
            this.queue.forEach(q => {
                pushPath(q.loaded, [lng], ns);
                removePending(q, name);
                if (err) q.errors.push(err);
                if (q.pendingCount === 0 && !q.done) {
                    Object.keys(q.loaded).forEach(l => {
                        if (!loaded[l]) loaded[l] = {};
                        const loadedKeys = q.loaded[l];
                        if (loadedKeys.length) {
                            loadedKeys.forEach(n => {
                                if (loaded[l][n] === undefined) loaded[l][n] = true;
                            });
                        }
                    });
                    q.done = true;
                    if (q.errors.length) {
                        q.callback(q.errors);
                    } else {
                        q.callback();
                    }
                }
            });
            this.emit('loaded', loaded);
            this.queue = this.queue.filter(q => !q.done);
        }
        read(lng, ns, fcName) {
            let tried = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 0;
            let wait = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : this.retryTimeout;
            let callback = arguments.length > 5 ? arguments[5] : undefined;
            if (!lng.length) return callback(null, {});
            if (this.readingCalls >= this.maxParallelReads) {
                this.waitingReads.push({
                    lng,
                    ns,
                    fcName,
                    tried,
                    wait,
                    callback
                });
                return;
            }
            this.readingCalls++;
            const resolver = (err, data) => {
                this.readingCalls--;
                if (this.waitingReads.length > 0) {
                    const next = this.waitingReads.shift();
                    this.read(next.lng, next.ns, next.fcName, next.tried, next.wait, next.callback);
                }
                if (err && data && tried < this.maxRetries) {
                    setTimeout(() => {
                        this.read.call(this, lng, ns, fcName, tried + 1, wait * 2, callback);
                    }, wait);
                    return;
                }
                callback(err, data);
            };
            const fc = this.backend[fcName].bind(this.backend);
            if (fc.length === 2) {
                try {
                    const r = fc(lng, ns);
                    if (r && typeof r.then === 'function') {
                        r.then(data => resolver(null, data)).catch(resolver);
                    } else {
                        resolver(null, r);
                    }
                } catch (err) {
                    resolver(err);
                }
                return;
            }
            return fc(lng, ns, resolver);
        }
        prepareLoading(languages, namespaces) {
            let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
            let callback = arguments.length > 3 ? arguments[3] : undefined;
            if (!this.backend) {
                this.logger.warn('No backend was added via i18next.use. Will not load resources.');
                return callback && callback();
            }
            if (typeof languages === 'string') languages = this.languageUtils.toResolveHierarchy(languages);
            if (typeof namespaces === 'string') namespaces = [namespaces];
            const toLoad = this.queueLoad(languages, namespaces, options, callback);
            if (!toLoad.toLoad.length) {
                if (!toLoad.pending.length) callback();
                return null;
            }
            toLoad.toLoad.forEach(name => {
                this.loadOne(name);
            });
        }
        load(languages, namespaces, callback) {
            this.prepareLoading(languages, namespaces, {}, callback);
        }
        reload(languages, namespaces, callback) {
            this.prepareLoading(languages, namespaces, {
                reload: true
            }, callback);
        }
        loadOne(name) {
            let prefix = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : '';
            const s = name.split('|');
            const lng = s[0];
            const ns = s[1];
            this.read(lng, ns, 'read', undefined, undefined, (err, data) => {
                if (err) this.logger.warn(`${prefix}loading namespace ${ns} for language ${lng} failed`, err);
                if (!err && data) this.logger.log(`${prefix}loaded namespace ${ns} for language ${lng}`, data);
                this.loaded(name, err, data);
            });
        }
        saveMissing(languages, namespace, key, fallbackValue, isUpdate) {
            let options = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : {};
            let clb = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : () => { };
            if (this.services.utils && this.services.utils.hasLoadedNamespace && !this.services.utils.hasLoadedNamespace(namespace)) {
                this.logger.warn(`did not save key "${key}" as the namespace "${namespace}" was not yet loaded`, 'This means something IS WRONG in your setup. You access the t function before i18next.init / i18next.loadNamespace / i18next.changeLanguage was done. Wait for the callback or Promise to resolve before accessing it!!!');
                return;
            }
            if (key === undefined || key === null || key === '') return;
            if (this.backend && this.backend.create) {
                const opts = {
                    ...options,
                    isUpdate
                };
                const fc = this.backend.create.bind(this.backend);
                if (fc.length < 6) {
                    try {
                        let r;
                        if (fc.length === 5) {
                            r = fc(languages, namespace, key, fallbackValue, opts);
                        } else {
                            r = fc(languages, namespace, key, fallbackValue);
                        }
                        if (r && typeof r.then === 'function') {
                            r.then(data => clb(null, data)).catch(clb);
                        } else {
                            clb(null, r);
                        }
                    } catch (err) {
                        clb(err);
                    }
                } else {
                    fc(languages, namespace, key, fallbackValue, clb, opts);
                }
            }
            if (!languages || !languages[0]) return;
            this.store.addResource(languages[0], namespace, key, fallbackValue);
        }
    }

    function get() {
        return {
            debug: false,
            initImmediate: true,
            ns: ['translation'],
            defaultNS: ['translation'],
            fallbackLng: ['dev'],
            fallbackNS: false,
            supportedLngs: false,
            nonExplicitSupportedLngs: false,
            load: 'all',
            preload: false,
            simplifyPluralSuffix: true,
            keySeparator: '.',
            nsSeparator: ':',
            pluralSeparator: '_',
            contextSeparator: '_',
            partialBundledLanguages: false,
            saveMissing: false,
            updateMissing: false,
            saveMissingTo: 'fallback',
            saveMissingPlurals: true,
            missingKeyHandler: false,
            missingInterpolationHandler: false,
            postProcess: false,
            postProcessPassResolved: false,
            returnNull: false,
            returnEmptyString: true,
            returnObjects: false,
            joinArrays: false,
            returnedObjectHandler: false,
            parseMissingKeyHandler: false,
            appendNamespaceToMissingKey: false,
            appendNamespaceToCIMode: false,
            overloadTranslationOptionHandler: function handle(args) {
                let ret = {};
                if (typeof args[1] === 'object') ret = args[1];
                if (typeof args[1] === 'string') ret.defaultValue = args[1];
                if (typeof args[2] === 'string') ret.tDescription = args[2];
                if (typeof args[2] === 'object' || typeof args[3] === 'object') {
                    const options = args[3] || args[2];
                    Object.keys(options).forEach(key => {
                        ret[key] = options[key];
                    });
                }
                return ret;
            },
            interpolation: {
                escapeValue: true,
                format: (value, format, lng, options) => value,
                prefix: '{{',
                suffix: '}}',
                formatSeparator: ',',
                unescapePrefix: '-',
                nestingPrefix: '$t(',
                nestingSuffix: ')',
                nestingOptionsSeparator: ',',
                maxReplaces: 1000,
                skipOnVariables: true
            }
        };
    }
    function transformOptions(options) {
        if (typeof options.ns === 'string') options.ns = [options.ns];
        if (typeof options.fallbackLng === 'string') options.fallbackLng = [options.fallbackLng];
        if (typeof options.fallbackNS === 'string') options.fallbackNS = [options.fallbackNS];
        if (options.supportedLngs && options.supportedLngs.indexOf('cimode') < 0) {
            options.supportedLngs = options.supportedLngs.concat(['cimode']);
        }
        return options;
    }

    function noop() { }
    function bindMemberFunctions(inst) {
        const mems = Object.getOwnPropertyNames(Object.getPrototypeOf(inst));
        mems.forEach(mem => {
            if (typeof inst[mem] === 'function') {
                inst[mem] = inst[mem].bind(inst);
            }
        });
    }
    class I18n extends EventEmitter {
        constructor() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            let callback = arguments.length > 1 ? arguments[1] : undefined;
            super();
            this.options = transformOptions(options);
            this.services = {};
            this.logger = baseLogger;
            this.modules = {
                external: []
            };
            bindMemberFunctions(this);
            if (callback && !this.isInitialized && !options.isClone) {
                if (!this.options.initImmediate) {
                    this.init(options, callback);
                    return this;
                }
                setTimeout(() => {
                    this.init(options, callback);
                }, 0);
            }
        }
        init() {
            var _this = this;
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            let callback = arguments.length > 1 ? arguments[1] : undefined;
            if (typeof options === 'function') {
                callback = options;
                options = {};
            }
            if (!options.defaultNS && options.defaultNS !== false && options.ns) {
                if (typeof options.ns === 'string') {
                    options.defaultNS = options.ns;
                } else if (options.ns.indexOf('translation') < 0) {
                    options.defaultNS = options.ns[0];
                }
            }
            const defOpts = get();
            this.options = {
                ...defOpts,
                ...this.options,
                ...transformOptions(options)
            };
            if (this.options.compatibilityAPI !== 'v1') {
                this.options.interpolation = {
                    ...defOpts.interpolation,
                    ...this.options.interpolation
                };
            }
            if (options.keySeparator !== undefined) {
                this.options.userDefinedKeySeparator = options.keySeparator;
            }
            if (options.nsSeparator !== undefined) {
                this.options.userDefinedNsSeparator = options.nsSeparator;
            }
            function createClassOnDemand(ClassOrObject) {
                if (!ClassOrObject) return null;
                if (typeof ClassOrObject === 'function') return new ClassOrObject();
                return ClassOrObject;
            }
            if (!this.options.isClone) {
                if (this.modules.logger) {
                    baseLogger.init(createClassOnDemand(this.modules.logger), this.options);
                } else {
                    baseLogger.init(null, this.options);
                }
                let formatter;
                if (this.modules.formatter) {
                    formatter = this.modules.formatter;
                } else if (typeof Intl !== 'undefined') {
                    formatter = Formatter;
                }
                const lu = new LanguageUtil(this.options);
                this.store = new ResourceStore(this.options.resources, this.options);
                const s = this.services;
                s.logger = baseLogger;
                s.resourceStore = this.store;
                s.languageUtils = lu;
                s.pluralResolver = new PluralResolver(lu, {
                    prepend: this.options.pluralSeparator,
                    compatibilityJSON: this.options.compatibilityJSON,
                    simplifyPluralSuffix: this.options.simplifyPluralSuffix
                });
                if (formatter && (!this.options.interpolation.format || this.options.interpolation.format === defOpts.interpolation.format)) {
                    s.formatter = createClassOnDemand(formatter);
                    s.formatter.init(s, this.options);
                    this.options.interpolation.format = s.formatter.format.bind(s.formatter);
                }
                s.interpolator = new Interpolator(this.options);
                s.utils = {
                    hasLoadedNamespace: this.hasLoadedNamespace.bind(this)
                };
                s.backendConnector = new Connector(createClassOnDemand(this.modules.backend), s.resourceStore, s, this.options);
                s.backendConnector.on('*', function (event) {
                    for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
                        args[_key - 1] = arguments[_key];
                    }
                    _this.emit(event, ...args);
                });
                if (this.modules.languageDetector) {
                    s.languageDetector = createClassOnDemand(this.modules.languageDetector);
                    if (s.languageDetector.init) s.languageDetector.init(s, this.options.detection, this.options);
                }
                if (this.modules.i18nFormat) {
                    s.i18nFormat = createClassOnDemand(this.modules.i18nFormat);
                    if (s.i18nFormat.init) s.i18nFormat.init(this);
                }
                this.translator = new Translator(this.services, this.options);
                this.translator.on('*', function (event) {
                    for (var _len2 = arguments.length, args = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
                        args[_key2 - 1] = arguments[_key2];
                    }
                    _this.emit(event, ...args);
                });
                this.modules.external.forEach(m => {
                    if (m.init) m.init(this);
                });
            }
            this.format = this.options.interpolation.format;
            if (!callback) callback = noop;
            if (this.options.fallbackLng && !this.services.languageDetector && !this.options.lng) {
                const codes = this.services.languageUtils.getFallbackCodes(this.options.fallbackLng);
                if (codes.length > 0 && codes[0] !== 'dev') this.options.lng = codes[0];
            }
            if (!this.services.languageDetector && !this.options.lng) {
                this.logger.warn('init: no languageDetector is used and no lng is defined');
            }
            const storeApi = ['getResource', 'hasResourceBundle', 'getResourceBundle', 'getDataByLanguage'];
            storeApi.forEach(fcName => {
                this[fcName] = function () {
                    return _this.store[fcName](...arguments);
                };
            });
            const storeApiChained = ['addResource', 'addResources', 'addResourceBundle', 'removeResourceBundle'];
            storeApiChained.forEach(fcName => {
                this[fcName] = function () {
                    _this.store[fcName](...arguments);
                    return _this;
                };
            });
            const deferred = defer();
            const load = () => {
                const finish = (err, t) => {
                    if (this.isInitialized && !this.initializedStoreOnce) this.logger.warn('init: i18next is already initialized. You should call init just once!');
                    this.isInitialized = true;
                    if (!this.options.isClone) this.logger.log('initialized', this.options);
                    this.emit('initialized', this.options);
                    deferred.resolve(t);
                    callback(err, t);
                };
                if (this.languages && this.options.compatibilityAPI !== 'v1' && !this.isInitialized) return finish(null, this.t.bind(this));
                this.changeLanguage(this.options.lng, finish);
            };
            if (this.options.resources || !this.options.initImmediate) {
                load();
            } else {
                setTimeout(load, 0);
            }
            return deferred;
        }
        loadResources(language) {
            let callback = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : noop;
            let usedCallback = callback;
            const usedLng = typeof language === 'string' ? language : this.language;
            if (typeof language === 'function') usedCallback = language;
            if (!this.options.resources || this.options.partialBundledLanguages) {
                if (usedLng && usedLng.toLowerCase() === 'cimode') return usedCallback();
                const toLoad = [];
                const append = lng => {
                    if (!lng) return;
                    const lngs = this.services.languageUtils.toResolveHierarchy(lng);
                    lngs.forEach(l => {
                        if (toLoad.indexOf(l) < 0) toLoad.push(l);
                    });
                };
                if (!usedLng) {
                    const fallbacks = this.services.languageUtils.getFallbackCodes(this.options.fallbackLng);
                    fallbacks.forEach(l => append(l));
                } else {
                    append(usedLng);
                }
                if (this.options.preload) {
                    this.options.preload.forEach(l => append(l));
                }
                this.services.backendConnector.load(toLoad, this.options.ns, e => {
                    if (!e && !this.resolvedLanguage && this.language) this.setResolvedLanguage(this.language);
                    usedCallback(e);
                });
            } else {
                usedCallback(null);
            }
        }
        reloadResources(lngs, ns, callback) {
            const deferred = defer();
            if (!lngs) lngs = this.languages;
            if (!ns) ns = this.options.ns;
            if (!callback) callback = noop;
            this.services.backendConnector.reload(lngs, ns, err => {
                deferred.resolve();
                callback(err);
            });
            return deferred;
        }
        use(module) {
            if (!module) throw new Error('You are passing an undefined module! Please check the object you are passing to i18next.use()');
            if (!module.type) throw new Error('You are passing a wrong module! Please check the object you are passing to i18next.use()');
            if (module.type === 'backend') {
                this.modules.backend = module;
            }
            if (module.type === 'logger' || module.log && module.warn && module.error) {
                this.modules.logger = module;
            }
            if (module.type === 'languageDetector') {
                this.modules.languageDetector = module;
            }
            if (module.type === 'i18nFormat') {
                this.modules.i18nFormat = module;
            }
            if (module.type === 'postProcessor') {
                postProcessor.addPostProcessor(module);
            }
            if (module.type === 'formatter') {
                this.modules.formatter = module;
            }
            if (module.type === '3rdParty') {
                this.modules.external.push(module);
            }
            return this;
        }
        setResolvedLanguage(l) {
            if (!l || !this.languages) return;
            if (['cimode', 'dev'].indexOf(l) > -1) return;
            for (let li = 0; li < this.languages.length; li++) {
                const lngInLngs = this.languages[li];
                if (['cimode', 'dev'].indexOf(lngInLngs) > -1) continue;
                if (this.store.hasLanguageSomeTranslations(lngInLngs)) {
                    this.resolvedLanguage = lngInLngs;
                    break;
                }
            }
        }
        changeLanguage(lng, callback) {
            var _this2 = this;
            this.isLanguageChangingTo = lng;
            const deferred = defer();
            this.emit('languageChanging', lng);
            const setLngProps = l => {
                this.language = l;
                this.languages = this.services.languageUtils.toResolveHierarchy(l);
                this.resolvedLanguage = undefined;
                this.setResolvedLanguage(l);
            };
            const done = (err, l) => {
                if (l) {
                    setLngProps(l);
                    this.translator.changeLanguage(l);
                    this.isLanguageChangingTo = undefined;
                    this.emit('languageChanged', l);
                    this.logger.log('languageChanged', l);
                } else {
                    this.isLanguageChangingTo = undefined;
                }
                deferred.resolve(function () {
                    return _this2.t(...arguments);
                });
                if (callback) callback(err, function () {
                    return _this2.t(...arguments);
                });
            };
            const setLng = lngs => {
                if (!lng && !lngs && this.services.languageDetector) lngs = [];
                const l = typeof lngs === 'string' ? lngs : this.services.languageUtils.getBestMatchFromCodes(lngs);
                if (l) {
                    if (!this.language) {
                        setLngProps(l);
                    }
                    if (!this.translator.language) this.translator.changeLanguage(l);
                    if (this.services.languageDetector && this.services.languageDetector.cacheUserLanguage) this.services.languageDetector.cacheUserLanguage(l);
                }
                this.loadResources(l, err => {
                    done(err, l);
                });
            };
            if (!lng && this.services.languageDetector && !this.services.languageDetector.async) {
                setLng(this.services.languageDetector.detect());
            } else if (!lng && this.services.languageDetector && this.services.languageDetector.async) {
                if (this.services.languageDetector.detect.length === 0) {
                    this.services.languageDetector.detect().then(setLng);
                } else {
                    this.services.languageDetector.detect(setLng);
                }
            } else {
                setLng(lng);
            }
            return deferred;
        }
        getFixedT(lng, ns, keyPrefix) {
            var _this3 = this;
            const fixedT = function (key, opts) {
                let options;
                if (typeof opts !== 'object') {
                    for (var _len3 = arguments.length, rest = new Array(_len3 > 2 ? _len3 - 2 : 0), _key3 = 2; _key3 < _len3; _key3++) {
                        rest[_key3 - 2] = arguments[_key3];
                    }
                    options = _this3.options.overloadTranslationOptionHandler([key, opts].concat(rest));
                } else {
                    options = {
                        ...opts
                    };
                }
                options.lng = options.lng || fixedT.lng;
                options.lngs = options.lngs || fixedT.lngs;
                options.ns = options.ns || fixedT.ns;
                options.keyPrefix = options.keyPrefix || keyPrefix || fixedT.keyPrefix;
                const keySeparator = _this3.options.keySeparator || '.';
                let resultKey;
                if (options.keyPrefix && Array.isArray(key)) {
                    resultKey = key.map(k => `${options.keyPrefix}${keySeparator}${k}`);
                } else {
                    resultKey = options.keyPrefix ? `${options.keyPrefix}${keySeparator}${key}` : key;
                }
                return _this3.t(resultKey, options);
            };
            if (typeof lng === 'string') {
                fixedT.lng = lng;
            } else {
                fixedT.lngs = lng;
            }
            fixedT.ns = ns;
            fixedT.keyPrefix = keyPrefix;
            return fixedT;
        }
        t() {
            return this.translator && this.translator.translate(...arguments);
        }
        exists() {
            return this.translator && this.translator.exists(...arguments);
        }
        setDefaultNamespace(ns) {
            this.options.defaultNS = ns;
        }
        hasLoadedNamespace(ns) {
            let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
            if (!this.isInitialized) {
                this.logger.warn('hasLoadedNamespace: i18next was not initialized', this.languages);
                return false;
            }
            if (!this.languages || !this.languages.length) {
                this.logger.warn('hasLoadedNamespace: i18n.languages were undefined or empty', this.languages);
                return false;
            }
            const lng = options.lng || this.resolvedLanguage || this.languages[0];
            const fallbackLng = this.options ? this.options.fallbackLng : false;
            const lastLng = this.languages[this.languages.length - 1];
            if (lng.toLowerCase() === 'cimode') return true;
            const loadNotPending = (l, n) => {
                const loadState = this.services.backendConnector.state[`${l}|${n}`];
                return loadState === -1 || loadState === 2;
            };
            if (options.precheck) {
                const preResult = options.precheck(this, loadNotPending);
                if (preResult !== undefined) return preResult;
            }
            if (this.hasResourceBundle(lng, ns)) return true;
            if (!this.services.backendConnector.backend || this.options.resources && !this.options.partialBundledLanguages) return true;
            if (loadNotPending(lng, ns) && (!fallbackLng || loadNotPending(lastLng, ns))) return true;
            return false;
        }
        loadNamespaces(ns, callback) {
            const deferred = defer();
            if (!this.options.ns) {
                if (callback) callback();
                return Promise.resolve();
            }
            if (typeof ns === 'string') ns = [ns];
            ns.forEach(n => {
                if (this.options.ns.indexOf(n) < 0) this.options.ns.push(n);
            });
            this.loadResources(err => {
                deferred.resolve();
                if (callback) callback(err);
            });
            return deferred;
        }
        loadLanguages(lngs, callback) {
            const deferred = defer();
            if (typeof lngs === 'string') lngs = [lngs];
            const preloaded = this.options.preload || [];
            const newLngs = lngs.filter(lng => preloaded.indexOf(lng) < 0);
            if (!newLngs.length) {
                if (callback) callback();
                return Promise.resolve();
            }
            this.options.preload = preloaded.concat(newLngs);
            this.loadResources(err => {
                deferred.resolve();
                if (callback) callback(err);
            });
            return deferred;
        }
        dir(lng) {
            if (!lng) lng = this.resolvedLanguage || (this.languages && this.languages.length > 0 ? this.languages[0] : this.language);
            if (!lng) return 'rtl';
            const rtlLngs = ['ar', 'shu', 'sqr', 'ssh', 'xaa', 'yhd', 'yud', 'aao', 'abh', 'abv', 'acm', 'acq', 'acw', 'acx', 'acy', 'adf', 'ads', 'aeb', 'aec', 'afb', 'ajp', 'apc', 'apd', 'arb', 'arq', 'ars', 'ary', 'arz', 'auz', 'avl', 'ayh', 'ayl', 'ayn', 'ayp', 'bbz', 'pga', 'he', 'iw', 'ps', 'pbt', 'pbu', 'pst', 'prp', 'prd', 'ug', 'ur', 'ydd', 'yds', 'yih', 'ji', 'yi', 'hbo', 'men', 'xmn', 'fa', 'jpr', 'peo', 'pes', 'prs', 'dv', 'sam', 'ckb'];
            const languageUtils = this.services && this.services.languageUtils || new LanguageUtil(get());
            return rtlLngs.indexOf(languageUtils.getLanguagePartFromCode(lng)) > -1 || lng.toLowerCase().indexOf('-arab') > 1 ? 'rtl' : 'ltr';
        }
        static createInstance() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            let callback = arguments.length > 1 ? arguments[1] : undefined;
            return new I18n(options, callback);
        }
        cloneInstance() {
            let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            let callback = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : noop;
            const forkResourceStore = options.forkResourceStore;
            if (forkResourceStore) delete options.forkResourceStore;
            const mergedOptions = {
                ...this.options,
                ...options,
                ...{
                    isClone: true
                }
            };
            const clone = new I18n(mergedOptions);
            if (options.debug !== undefined || options.prefix !== undefined) {
                clone.logger = clone.logger.clone(options);
            }
            const membersToCopy = ['store', 'services', 'language'];
            membersToCopy.forEach(m => {
                clone[m] = this[m];
            });
            clone.services = {
                ...this.services
            };
            clone.services.utils = {
                hasLoadedNamespace: clone.hasLoadedNamespace.bind(clone)
            };
            if (forkResourceStore) {
                clone.store = new ResourceStore(this.store.data, mergedOptions);
                clone.services.resourceStore = clone.store;
            }
            clone.translator = new Translator(clone.services, mergedOptions);
            clone.translator.on('*', function (event) {
                for (var _len4 = arguments.length, args = new Array(_len4 > 1 ? _len4 - 1 : 0), _key4 = 1; _key4 < _len4; _key4++) {
                    args[_key4 - 1] = arguments[_key4];
                }
                clone.emit(event, ...args);
            });
            clone.init(mergedOptions, callback);
            clone.translator.options = mergedOptions;
            clone.translator.backendConnector.services.utils = {
                hasLoadedNamespace: clone.hasLoadedNamespace.bind(clone)
            };
            return clone;
        }
        toJSON() {
            return {
                options: this.options,
                store: this.store,
                language: this.language,
                languages: this.languages,
                resolvedLanguage: this.resolvedLanguage
            };
        }
    }
    const instance = I18n.createInstance();
    instance.createInstance = I18n.createInstance;

    return instance;

}));