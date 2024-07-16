var feature = require('caniuse-lite/dist/unpacker/feature').default
var region = require('caniuse-lite/dist/unpacker/region').default
var path = require('path')
var fs = require('fs')

var BrowserslistError = require('./error')

var IS_SECTION = /^\s*\[(.+)]\s*$/
var CONFIG_PATTERN = /^browserslist-config-/
var SCOPED_CONFIG__PATTERN = /@[^/]+(?:\/[^/]+)?\/browserslist-config(?:-|$|\/)/
var TIME_TO_UPDATE_CANIUSE = 6 * 30 * 24 * 60 * 60 * 1000
var FORMAT =
  'Browserslist config should be a string or an array ' +
  'of strings with browser queries'

var dataTimeChecked = false
var filenessCache = {}
var configCache = {}
function checkExtend(name) {
  var use = ' Use `dangerousExtend` option to disable.'
  if (!CONFIG_PATTERN.test(name) && !SCOPED_CONFIG__PATTERN.test(name)) {
    throw new BrowserslistError(
      'Browserslist config needs `browserslist-config-` prefix. ' + use
    )
  }
  if (name.replace(/^@[^/]+\//, '').indexOf('.') !== -1) {
    throw new BrowserslistError(
      '`.` not allowed in Browserslist config name. ' + use
    )
  }
  if (name.indexOf('node_modules') !== -1) {
    throw new BrowserslistError(
      '`node_modules` not allowed in Browserslist config.' + use
    )
  }
}

function isFile(file) {
  if (file in filenessCache) {
    return filenessCache[file]
  }
  var result = fs.existsSync(file) && fs.statSync(file).isFile()
  if (!process.env.BROWSERSLIST_DISABLE_CACHE) {
    filenessCache[file] = result
  }
  return result
}

function eachParent(file, callback) {
  var dir = isFile(file) ? path.dirname(file) : file
  var loc = path.resolve(dir)
  do {
    if (!pathInRoot(loc)) break;
    var result = callback(loc)
    if (typeof result !== 'undefined') return result
  } while (loc !== (loc = path.dirname(loc)))
  return undefined
}

function pathInRoot(p) {
  if (!process.env.BROWSERSLIST_ROOT_PATH) return true
  var rootPath = path.resolve(process.env.BROWSERSLIST_ROOT_PATH);
  if (path.relative(rootPath, p).substring(0,2) === '..') {
    return false;
  }
  return true
}

function check(section) {
  if (Array.isArray(section)) {
    for (var i = 0; i < section.length; i++) {
      if (typeof section[i] !== 'string') {
        throw new BrowserslistError(FORMAT)
      }
    }
  } else if (typeof section !== 'string') {
    throw new BrowserslistError(FORMAT)
  }
}

function pickEnv(config, opts) {
  if (typeof config !== 'object') return config

  var name
  if (typeof opts.env === 'string') {
    name = opts.env
  } else if (process.env.BROWSERSLIST_ENV) {
    name = process.env.BROWSERSLIST_ENV
  } else if (process.env.NODE_ENV) {
    name = process.env.NODE_ENV
  } else {
    name = 'production'
  }

  if (opts.throwOnMissing) {
    if (name && name !== 'defaults' && !config[name]) {
      throw new BrowserslistError(
        'Missing config for Browserslist environment `' + name + '`'
      )
    }
  }

  return config[name] || config.defaults
}

function parsePackage(file) {
  var config = JSON.parse(
    fs
      .readFileSync(file)
      .toString()
      .replace(/^\uFEFF/m, '')
  )
  if (config.browserlist && !config.browserslist) {
    throw new BrowserslistError(
      '`browserlist` key instead of `browserslist` in ' + file
    )
  }
  var list = config.browserslist
  if (Array.isArray(list) || typeof list === 'string') {
    list = { defaults: list }
  }
  for (var i in list) {
    check(list[i])
  }

  return list
}

function latestReleaseTime(agents) {
  var latest = 0
  for (var name in agents) {
    var dates = agents[name].releaseDate || {}
    for (var key in dates) {
      if (latest < dates[key]) {
        latest = dates[key]
      }
    }
  }
  return latest * 1000
}

function normalizeStats(data, stats) {
  if (!data) {
    data = {}
  }
  if (stats && 'dataByBrowser' in stats) {
    stats = stats.dataByBrowser
  }

  if (typeof stats !== 'object') return undefined

  var normalized = {}
  for (var i in stats) {
    var versions = Object.keys(stats[i])
    if (versions.length === 1 && data[i] && data[i].versions.length === 1) {
      var normal = data[i].versions[0]
      normalized[i] = {}
      normalized[i][normal] = stats[i][versions[0]]
    } else {
      normalized[i] = stats[i]
    }
  }

  return normalized
}

function normalizeUsageData(usageData, data) {
  for (var browser in usageData) {
    var browserUsage = usageData[browser]
    // eslint-disable-next-line max-len
    // https://github.com/browserslist/browserslist/issues/431#issuecomment-565230615
    // caniuse-db returns { 0: "percentage" } for `and_*` regional stats
    if ('0' in browserUsage) {
      var versions = data[browser].versions
      browserUsage[versions[versions.length - 1]] = browserUsage[0]
      delete browserUsage[0]
    }
  }
}

module.exports = {
  loadQueries: function loadQueries(ctx, name) {
    if (!ctx.dangerousExtend && !process.env.BROWSERSLIST_DANGEROUS_EXTEND) {
      checkExtend(name)
    }
    var queries = require(require.resolve(name, { paths: ['.', ctx.path] }))
    if (queries) {
      if (Array.isArray(queries)) {
        return queries
      } else if (typeof queries === 'object') {
        if (!queries.defaults) queries.defaults = []
        return pickEnv(queries, ctx, name)
      }
    }
    throw new BrowserslistError(
      '`' +
        name +
        '` config exports not an array of queries' +
        ' or an object of envs'
    )
  },

  loadStat: function loadStat(ctx, name, data) {
    if (!ctx.dangerousExtend && !process.env.BROWSERSLIST_DANGEROUS_EXTEND) {
      checkExtend(name)
    }
    var stats = require(require.resolve(
      path.join(name, 'browserslist-stats.json'),
      { paths: ['.'] }
    ))
    return normalizeStats(data, stats)
  },

  getStat: function getStat(opts, data) {
    var stats
    if (opts.stats) {
      stats = opts.stats
    } else if (process.env.BROWSERSLIST_STATS) {
      stats = process.env.BROWSERSLIST_STATS
    } else if (opts.path && path.resolve && fs.existsSync) {
      stats = eachParent(opts.path, function (dir) {
        var file = path.join(dir, 'browserslist-stats.json')
        return isFile(file) ? file : undefined
      })
    }
    if (typeof stats === 'string') {
      try {
        stats = JSON.parse(fs.readFileSync(stats))
      } catch (e) {
        throw new BrowserslistError("Can't read " + stats)
      }
    }
    return normalizeStats(data, stats)
  },

  loadConfig: function loadConfig(opts) {
    if (process.env.BROWSERSLIST) {
      return process.env.BROWSERSLIST
    } else if (opts.config || process.env.BROWSERSLIST_CONFIG) {
      var file = opts.config || process.env.BROWSERSLIST_CONFIG
      if (path.basename(file) === 'package.json') {
        return pickEnv(parsePackage(file), opts)
      } else {
        return pickEnv(module.exports.readConfig(file), opts)
      }
    } else if (opts.path) {
      return pickEnv(module.exports.findConfig(opts.path), opts)
    } else {
      return undefined
    }
  },

  loadCountry: function loadCountry(usage, country, data) {
    var code = country.replace(/[^\w-]/g, '')
    if (!usage[code]) {
      var compressed
      try {
        compressed = require('caniuse-lite/data/regions/' + code + '.js')
      } catch (e) {
        throw new BrowserslistError('Unknown region name `' + code + '`.')
      }
      var usageData = region(compressed)
      normalizeUsageData(usageData, data)
      usage[country] = {}
      for (var i in usageData) {
        for (var j in usageData[i]) {
          usage[country][i + ' ' + j] = usageData[i][j]
        }
      }
    }
  },

  loadFeature: function loadFeature(features, name) {
    name = name.replace(/[^\w-]/g, '')
    if (features[name]) return
    var compressed
    try {
      compressed = require('caniuse-lite/data/features/' + name + '.js')
    } catch (e) {
      throw new BrowserslistError('Unknown feature name `' + name + '`.')
    }
    var stats = feature(compressed).stats
    features[name] = {}
    for (var i in stats) {
      features[name][i] = {}
      for (var j in stats[i]) {
        features[name][i][j] = stats[i][j]
      }
    }
  },

  parseConfig: function parseConfig(string) {
    var result = { defaults: [] }
    var sections = ['defaults']

    string
      .toString()
      .replace(/#[^\n]*/g, '')
      .split(/\n|,/)
      .map(function (line) {
        return line.trim()
      })
      .filter(function (line) {
        return line !== ''
      })
      .forEach(function (line) {
        if (IS_SECTION.test(line)) {
          sections = line.match(IS_SECTION)[1].trim().split(' ')
          sections.forEach(function (section) {
            if (result[section]) {
              throw new BrowserslistError(
                'Duplicate section ' + section + ' in Browserslist config'
              )
            }
            result[section] = []
          })
        } else {
          sections.forEach(function (section) {
            result[section].push(line)
          })
        }
      })

    return result
  },

  readConfig: function readConfig(file) {
    if (!isFile(file)) {
      throw new BrowserslistError("Can't read " + file + ' config')
    }
    return module.exports.parseConfig(fs.readFileSync(file))
  },

  findConfig: function findConfig(from) {
    from = path.resolve(from)

    var passed = []
    var resolved = eachParent(from, function (dir) {
      if (dir in configCache) {
        return configCache[dir]
      }

      passed.push(dir)

      var config = path.join(dir, 'browserslist')
      var pkg = path.join(dir, 'package.json')
      var rc = path.join(dir, '.browserslistrc')

      var pkgBrowserslist
      if (isFile(pkg)) {
        try {
          pkgBrowserslist = parsePackage(pkg)
        } catch (e) {
          if (e.name === 'BrowserslistError') throw e
          console.warn(
            '[Browserslist] Could not parse ' + pkg + '. Ignoring it.'
          )
        }
      }

      if (isFile(config) && pkgBrowserslist) {
        throw new BrowserslistError(
          dir + ' contains both browserslist and package.json with browsers'
        )
      } else if (isFile(rc) && pkgBrowserslist) {
        throw new BrowserslistError(
          dir + ' contains both .browserslistrc and package.json with browsers'
        )
      } else if (isFile(config) && isFile(rc)) {
        throw new BrowserslistError(
          dir + ' contains both .browserslistrc and browserslist'
        )
      } else if (isFile(config)) {
        return module.exports.readConfig(config)
      } else if (isFile(rc)) {
        return module.exports.readConfig(rc)
      } else {
        return pkgBrowserslist
      }
    })
    if (!process.env.BROWSERSLIST_DISABLE_CACHE) {
      passed.forEach(function (dir) {
        configCache[dir] = resolved
      })
    }
    return resolved
  },

  clearCaches: function clearCaches() {
    dataTimeChecked = false
    filenessCache = {}
    configCache = {}

    this.cache = {}
  },

  oldDataWarning: function oldDataWarning(agentsObj) {
    if (dataTimeChecked) return
    dataTimeChecked = true
    if (process.env.BROWSERSLIST_IGNORE_OLD_DATA) return

    var latest = latestReleaseTime(agentsObj)
    var halfYearAgo = Date.now() - TIME_TO_UPDATE_CANIUSE

    if (latest !== 0 && latest < halfYearAgo) {
      console.warn(
        'Browserslist: caniuse-lite is outdated. Please run:\n' +
          '  npx update-browserslist-db@latest\n' +
          '  Why you should do it regularly: ' +
          'https://github.com/browserslist/update-db#readme'
      )
    }
  },

  currentNode: function currentNode() {
    return 'node ' + process.versions.node
  },

  env: process.env
}
