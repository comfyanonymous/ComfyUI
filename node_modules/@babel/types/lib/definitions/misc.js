"use strict";

var _utils = require("./utils.js");
var _placeholders = require("./placeholders.js");
const defineType = (0, _utils.defineAliasedType)("Miscellaneous");
{
  defineType("Noop", {
    visitor: []
  });
}
defineType("Placeholder", {
  visitor: [],
  builder: ["expectedNode", "name"],
  fields: {
    name: {
      validate: (0, _utils.assertNodeType)("Identifier")
    },
    expectedNode: {
      validate: (0, _utils.assertOneOf)(..._placeholders.PLACEHOLDERS)
    }
  }
});
defineType("V8IntrinsicIdentifier", {
  builder: ["name"],
  fields: {
    name: {
      validate: (0, _utils.assertValueType)("string")
    }
  }
});

//# sourceMappingURL=misc.js.map
