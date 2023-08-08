(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
    typeof define === 'function' && define.amd ? define(factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, global.i18nextBrowserLanguageDetector = factory());
  })(this, (function () { 'use strict';
  
    function _classCallCheck(instance, Constructor) {
      if (!(instance instanceof Constructor)) {
        throw new TypeError("Cannot call a class as a function");
      }
    }
  
    function _typeof(obj) {
      "@babel/helpers - typeof";
  
      return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) {
        return typeof obj;
      } : function (obj) {
        return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
      }, _typeof(obj);
    }
  
    function _toPrimitive(input, hint) {
      if (_typeof(input) !== "object" || input === null) return input;
      var prim = input[Symbol.toPrimitive];
      if (prim !== undefined) {
        var res = prim.call(input, hint || "default");
        if (_typeof(res) !== "object") return res;
        throw new TypeError("@@toPrimitive must return a primitive value.");
      }
      return (hint === "string" ? String : Number)(input);
    }
  
    function _toPropertyKey(arg) {
      var key = _toPrimitive(arg, "string");
      return _typeof(key) === "symbol" ? key : String(key);
    }
  
    function _defineProperties(target, props) {
      for (var i = 0; i < props.length; i++) {
        var descriptor = props[i];
        descriptor.enumerable = descriptor.enumerable || false;
        descriptor.configurable = true;
        if ("value" in descriptor) descriptor.writable = true;
        Object.defineProperty(target, _toPropertyKey(descriptor.key), descriptor);
      }
    }
    function _createClass(Constructor, protoProps, staticProps) {
      if (protoProps) _defineProperties(Constructor.prototype, protoProps);
      if (staticProps) _defineProperties(Constructor, staticProps);
      Object.defineProperty(Constructor, "prototype", {
        writable: false
      });
      return Constructor;
    }
  
    var arr = [];
    var each = arr.forEach;
    var slice = arr.slice;
    function defaults(obj) {
      each.call(slice.call(arguments, 1), function (source) {
        if (source) {
          for (var prop in source) {
            if (obj[prop] === undefined) obj[prop] = source[prop];
          }
        }
      });
      return obj;
    }
  
    // eslint-disable-next-line no-control-regex
    var fieldContentRegExp = /^[\u0009\u0020-\u007e\u0080-\u00ff]+$/;
    var serializeCookie = function serializeCookie(name, val, options) {
      var opt = options || {};
      opt.path = opt.path || '/';
      var value = encodeURIComponent(val);
      var str = "".concat(name, "=").concat(value);
      if (opt.maxAge > 0) {
        var maxAge = opt.maxAge - 0;
        if (Number.isNaN(maxAge)) throw new Error('maxAge should be a Number');
        str += "; Max-Age=".concat(Math.floor(maxAge));
      }
      if (opt.domain) {
        if (!fieldContentRegExp.test(opt.domain)) {
          throw new TypeError('option domain is invalid');
        }
        str += "; Domain=".concat(opt.domain);
      }
      if (opt.path) {
        if (!fieldContentRegExp.test(opt.path)) {
          throw new TypeError('option path is invalid');
        }
        str += "; Path=".concat(opt.path);
      }
      if (opt.expires) {
        if (typeof opt.expires.toUTCString !== 'function') {
          throw new TypeError('option expires is invalid');
        }
        str += "; Expires=".concat(opt.expires.toUTCString());
      }
      if (opt.httpOnly) str += '; HttpOnly';
      if (opt.secure) str += '; Secure';
      if (opt.sameSite) {
        var sameSite = typeof opt.sameSite === 'string' ? opt.sameSite.toLowerCase() : opt.sameSite;
        switch (sameSite) {
          case true:
            str += '; SameSite=Strict';
            break;
          case 'lax':
            str += '; SameSite=Lax';
            break;
          case 'strict':
            str += '; SameSite=Strict';
            break;
          case 'none':
            str += '; SameSite=None';
            break;
          default:
            throw new TypeError('option sameSite is invalid');
        }
      }
      return str;
    };
    var cookie = {
      create: function create(name, value, minutes, domain) {
        var cookieOptions = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : {
          path: '/',
          sameSite: 'strict'
        };
        if (minutes) {
          cookieOptions.expires = new Date();
          cookieOptions.expires.setTime(cookieOptions.expires.getTime() + minutes * 60 * 1000);
        }
        if (domain) cookieOptions.domain = domain;
        document.cookie = serializeCookie(name, encodeURIComponent(value), cookieOptions);
      },
      read: function read(name) {
        var nameEQ = "".concat(name, "=");
        var ca = document.cookie.split(';');
        for (var i = 0; i < ca.length; i++) {
          var c = ca[i];
          while (c.charAt(0) === ' ') {
            c = c.substring(1, c.length);
          }
          if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
        }
        return null;
      },
      remove: function remove(name) {
        this.create(name, '', -1);
      }
    };
    var cookie$1 = {
      name: 'cookie',
      lookup: function lookup(options) {
        var found;
        if (options.lookupCookie && typeof document !== 'undefined') {
          var c = cookie.read(options.lookupCookie);
          if (c) found = c;
        }
        return found;
      },
      cacheUserLanguage: function cacheUserLanguage(lng, options) {
        if (options.lookupCookie && typeof document !== 'undefined') {
          cookie.create(options.lookupCookie, lng, options.cookieMinutes, options.cookieDomain, options.cookieOptions);
        }
      }
    };
  
    var querystring = {
      name: 'querystring',
      lookup: function lookup(options) {
        var found;
        if (typeof window !== 'undefined') {
          var search = window.location.search;
          if (!window.location.search && window.location.hash && window.location.hash.indexOf('?') > -1) {
            search = window.location.hash.substring(window.location.hash.indexOf('?'));
          }
          var query = search.substring(1);
          var params = query.split('&');
          for (var i = 0; i < params.length; i++) {
            var pos = params[i].indexOf('=');
            if (pos > 0) {
              var key = params[i].substring(0, pos);
              if (key === options.lookupQuerystring) {
                found = params[i].substring(pos + 1);
              }
            }
          }
        }
        return found;
      }
    };
  
    var hasLocalStorageSupport = null;
    var localStorageAvailable = function localStorageAvailable() {
      if (hasLocalStorageSupport !== null) return hasLocalStorageSupport;
      try {
        hasLocalStorageSupport = window !== 'undefined' && window.localStorage !== null;
        var testKey = 'i18next.translate.boo';
        window.localStorage.setItem(testKey, 'foo');
        window.localStorage.removeItem(testKey);
      } catch (e) {
        hasLocalStorageSupport = false;
      }
      return hasLocalStorageSupport;
    };
    var localStorage = {
      name: 'localStorage',
      lookup: function lookup(options) {
        var found;
        if (options.lookupLocalStorage && localStorageAvailable()) {
          var lng = window.localStorage.getItem(options.lookupLocalStorage);
          if (lng) found = lng;
        }
        return found;
      },
      cacheUserLanguage: function cacheUserLanguage(lng, options) {
        if (options.lookupLocalStorage && localStorageAvailable()) {
          window.localStorage.setItem(options.lookupLocalStorage, lng);
        }
      }
    };
  
    var hasSessionStorageSupport = null;
    var sessionStorageAvailable = function sessionStorageAvailable() {
      if (hasSessionStorageSupport !== null) return hasSessionStorageSupport;
      try {
        hasSessionStorageSupport = window !== 'undefined' && window.sessionStorage !== null;
        var testKey = 'i18next.translate.boo';
        window.sessionStorage.setItem(testKey, 'foo');
        window.sessionStorage.removeItem(testKey);
      } catch (e) {
        hasSessionStorageSupport = false;
      }
      return hasSessionStorageSupport;
    };
    var sessionStorage = {
      name: 'sessionStorage',
      lookup: function lookup(options) {
        var found;
        if (options.lookupSessionStorage && sessionStorageAvailable()) {
          var lng = window.sessionStorage.getItem(options.lookupSessionStorage);
          if (lng) found = lng;
        }
        return found;
      },
      cacheUserLanguage: function cacheUserLanguage(lng, options) {
        if (options.lookupSessionStorage && sessionStorageAvailable()) {
          window.sessionStorage.setItem(options.lookupSessionStorage, lng);
        }
      }
    };
  
    var navigator$1 = {
      name: 'navigator',
      lookup: function lookup(options) {
        var found = [];
        if (typeof navigator !== 'undefined') {
          if (navigator.languages) {
            // chrome only; not an array, so can't use .push.apply instead of iterating
            for (var i = 0; i < navigator.languages.length; i++) {
              found.push(navigator.languages[i]);
            }
          }
          if (navigator.userLanguage) {
            found.push(navigator.userLanguage);
          }
          if (navigator.language) {
            found.push(navigator.language);
          }
        }
        return found.length > 0 ? found : undefined;
      }
    };
  
    var htmlTag = {
      name: 'htmlTag',
      lookup: function lookup(options) {
        var found;
        var htmlTag = options.htmlTag || (typeof document !== 'undefined' ? document.documentElement : null);
        if (htmlTag && typeof htmlTag.getAttribute === 'function') {
          found = htmlTag.getAttribute('lang');
        }
        return found;
      }
    };
  
    var path = {
      name: 'path',
      lookup: function lookup(options) {
        var found;
        if (typeof window !== 'undefined') {
          var language = window.location.pathname.match(/\/([a-zA-Z-]*)/g);
          if (language instanceof Array) {
            if (typeof options.lookupFromPathIndex === 'number') {
              if (typeof language[options.lookupFromPathIndex] !== 'string') {
                return undefined;
              }
              found = language[options.lookupFromPathIndex].replace('/', '');
            } else {
              found = language[0].replace('/', '');
            }
          }
        }
        return found;
      }
    };
  
    var subdomain = {
      name: 'subdomain',
      lookup: function lookup(options) {
        // If given get the subdomain index else 1
        var lookupFromSubdomainIndex = typeof options.lookupFromSubdomainIndex === 'number' ? options.lookupFromSubdomainIndex + 1 : 1;
        // get all matches if window.location. is existing
        // first item of match is the match itself and the second is the first group macht which sould be the first subdomain match
        // is the hostname no public domain get the or option of localhost
        var language = typeof window !== 'undefined' && window.location && window.location.hostname && window.location.hostname.match(/^(\w{2,5})\.(([a-z0-9-]{1,63}\.[a-z]{2,6})|localhost)/i);
  
        // if there is no match (null) return undefined
        if (!language) return undefined;
        // return the given group match
        return language[lookupFromSubdomainIndex];
      }
    };
  
    function getDefaults() {
      return {
        order: ['querystring', 'cookie', 'localStorage', 'sessionStorage', 'navigator', 'htmlTag'],
        lookupQuerystring: 'lng',
        lookupCookie: 'i18next',
        lookupLocalStorage: 'i18nextLng',
        lookupSessionStorage: 'i18nextLng',
        // cache user language
        caches: ['localStorage'],
        excludeCacheFor: ['cimode'],
        // cookieMinutes: 10,
        // cookieDomain: 'myDomain'
  
        convertDetectedLanguage: function convertDetectedLanguage(l) {
          return l;
        }
      };
    }
    var Browser = /*#__PURE__*/function () {
      function Browser(services) {
        var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
        _classCallCheck(this, Browser);
        this.type = 'languageDetector';
        this.detectors = {};
        this.init(services, options);
      }
      _createClass(Browser, [{
        key: "init",
        value: function init(services) {
          var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
          var i18nOptions = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
          this.services = services || {
            languageUtils: {}
          }; // this way the language detector can be used without i18next
          this.options = defaults(options, this.options || {}, getDefaults());
          if (typeof this.options.convertDetectedLanguage === 'string' && this.options.convertDetectedLanguage.indexOf('15897') > -1) {
            this.options.convertDetectedLanguage = function (l) {
              return l.replace('-', '_');
            };
          }
  
          // backwards compatibility
          if (this.options.lookupFromUrlIndex) this.options.lookupFromPathIndex = this.options.lookupFromUrlIndex;
          this.i18nOptions = i18nOptions;
          this.addDetector(cookie$1);
          this.addDetector(querystring);
          this.addDetector(localStorage);
          this.addDetector(sessionStorage);
          this.addDetector(navigator$1);
          this.addDetector(htmlTag);
          this.addDetector(path);
          this.addDetector(subdomain);
        }
      }, {
        key: "addDetector",
        value: function addDetector(detector) {
          this.detectors[detector.name] = detector;
        }
      }, {
        key: "detect",
        value: function detect(detectionOrder) {
          var _this = this;
          if (!detectionOrder) detectionOrder = this.options.order;
          var detected = [];
          detectionOrder.forEach(function (detectorName) {
            if (_this.detectors[detectorName]) {
              var lookup = _this.detectors[detectorName].lookup(_this.options);
              if (lookup && typeof lookup === 'string') lookup = [lookup];
              if (lookup) detected = detected.concat(lookup);
            }
          });
          detected = detected.map(function (d) {
            return _this.options.convertDetectedLanguage(d);
          });
          if (this.services.languageUtils.getBestMatchFromCodes) return detected; // new i18next v19.5.0
          return detected.length > 0 ? detected[0] : null; // a little backward compatibility
        }
      }, {
        key: "cacheUserLanguage",
        value: function cacheUserLanguage(lng, caches) {
          var _this2 = this;
          if (!caches) caches = this.options.caches;
          if (!caches) return;
          if (this.options.excludeCacheFor && this.options.excludeCacheFor.indexOf(lng) > -1) return;
          caches.forEach(function (cacheName) {
            if (_this2.detectors[cacheName]) _this2.detectors[cacheName].cacheUserLanguage(lng, _this2.options);
          });
        }
      }]);
      return Browser;
    }();
    Browser.type = 'languageDetector';
  
    return Browser;
  
  }));