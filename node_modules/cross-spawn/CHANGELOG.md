# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### [7.0.3](https://github.com/moxystudio/node-cross-spawn/compare/v7.0.2...v7.0.3) (2020-05-25)


### Bug Fixes

* detect path key based on correct environment ([#133](https://github.com/moxystudio/node-cross-spawn/issues/133)) ([159e7e9](https://github.com/moxystudio/node-cross-spawn/commit/159e7e9785e57451cba034ae51719f97135074ae))

### [7.0.2](https://github.com/moxystudio/node-cross-spawn/compare/v7.0.1...v7.0.2) (2020-04-04)


### Bug Fixes

* fix worker threads in Node >=11.10.0 ([#132](https://github.com/moxystudio/node-cross-spawn/issues/132)) ([6c5b4f0](https://github.com/moxystudio/node-cross-spawn/commit/6c5b4f015814a6c4f6b33230dfd1a860aedc0aaf))

### [7.0.1](https://github.com/moxystudio/node-cross-spawn/compare/v7.0.0...v7.0.1) (2019-10-07)


### Bug Fixes

* **core:** support worker threads ([#127](https://github.com/moxystudio/node-cross-spawn/issues/127)) ([cfd49c9](https://github.com/moxystudio/node-cross-spawn/commit/cfd49c9))

## [7.0.0](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.5...v7.0.0) (2019-09-03)


### ⚠ BREAKING CHANGES

* drop support for Node.js < 8

* drop support for versions below Node.js 8 ([#125](https://github.com/moxystudio/node-cross-spawn/issues/125)) ([16feb53](https://github.com/moxystudio/node-cross-spawn/commit/16feb53))

<a name="6.0.5"></a>
## [6.0.5](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.4...v6.0.5) (2018-03-02)


### Bug Fixes

* avoid using deprecated Buffer constructor ([#94](https://github.com/moxystudio/node-cross-spawn/issues/94)) ([d5770df](https://github.com/moxystudio/node-cross-spawn/commit/d5770df)), closes [/nodejs.org/api/deprecations.html#deprecations_dep0005](https://github.com//nodejs.org/api/deprecations.html/issues/deprecations_dep0005)



<a name="6.0.4"></a>
## [6.0.4](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.3...v6.0.4) (2018-01-31)


### Bug Fixes

* fix paths being incorrectly normalized on unix ([06ee3c6](https://github.com/moxystudio/node-cross-spawn/commit/06ee3c6)), closes [#90](https://github.com/moxystudio/node-cross-spawn/issues/90)



<a name="6.0.3"></a>
## [6.0.3](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.2...v6.0.3) (2018-01-23)



<a name="6.0.2"></a>
## [6.0.2](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.1...v6.0.2) (2018-01-23)



<a name="6.0.1"></a>
## [6.0.1](https://github.com/moxystudio/node-cross-spawn/compare/v6.0.0...v6.0.1) (2018-01-23)



<a name="6.0.0"></a>
# [6.0.0](https://github.com/moxystudio/node-cross-spawn/compare/5.1.0...6.0.0) (2018-01-23)


### Bug Fixes

* fix certain arguments not being correctly escaped or causing batch syntax error ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10)), closes [#82](https://github.com/moxystudio/node-cross-spawn/issues/82) [#51](https://github.com/moxystudio/node-cross-spawn/issues/51)
* fix commands as posix relatixe paths not working correctly, e.g.: `./my-command` ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10))
* fix `options` argument being mutated ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10))
* fix commands resolution when PATH was actually Path ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10))


### Features

* improve compliance with node's ENOENT errors ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10))
* improve detection of node's shell option support ([900cf10](https://github.com/moxystudio/node-cross-spawn/commit/900cf10))


### Chores

* upgrade tooling
* upgrate project to es6 (node v4)


### BREAKING CHANGES

* remove support for older nodejs versions, only `node >= 4` is supported


<a name="5.1.0"></a>
## [5.1.0](https://github.com/moxystudio/node-cross-spawn/compare/5.0.1...5.1.0) (2017-02-26)


### Bug Fixes

* fix `options.shell` support for NodeJS [v4.8](https://github.com/nodejs/node/blob/master/doc/changelogs/CHANGELOG_V4.md#4.8.0)


<a name="5.0.1"></a>
## [5.0.1](https://github.com/moxystudio/node-cross-spawn/compare/5.0.0...5.0.1) (2016-11-04)


### Bug Fixes

* fix `options.shell` support for NodeJS v7


<a name="5.0.0"></a>
# [5.0.0](https://github.com/moxystudio/node-cross-spawn/compare/4.0.2...5.0.0) (2016-10-30)


## Features

* add support for `options.shell`
* improve parsing of shebangs by using [`shebang-command`](https://github.com/kevva/shebang-command) module


## Chores

* refactor some code to make it more clear
* update README caveats
