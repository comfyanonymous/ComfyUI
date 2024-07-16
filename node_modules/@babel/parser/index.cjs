try {
  module.exports = require("./lib/index.cjs");
} catch {
  module.exports = require("./lib/index.js");
}
