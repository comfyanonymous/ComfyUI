"use strict";

var _utils = require("./utils.js");
(0, _utils.default)("ArgumentPlaceholder", {});
(0, _utils.default)("BindExpression", {
  visitor: ["object", "callee"],
  aliases: ["Expression"],
  fields: !process.env.BABEL_TYPES_8_BREAKING ? {
    object: {
      validate: Object.assign(() => {}, {
        oneOfNodeTypes: ["Expression"]
      })
    },
    callee: {
      validate: Object.assign(() => {}, {
        oneOfNodeTypes: ["Expression"]
      })
    }
  } : {
    object: {
      validate: (0, _utils.assertNodeType)("Expression")
    },
    callee: {
      validate: (0, _utils.assertNodeType)("Expression")
    }
  }
});
(0, _utils.default)("ImportAttribute", {
  visitor: ["key", "value"],
  fields: {
    key: {
      validate: (0, _utils.assertNodeType)("Identifier", "StringLiteral")
    },
    value: {
      validate: (0, _utils.assertNodeType)("StringLiteral")
    }
  }
});
(0, _utils.default)("Decorator", {
  visitor: ["expression"],
  fields: {
    expression: {
      validate: (0, _utils.assertNodeType)("Expression")
    }
  }
});
(0, _utils.default)("DoExpression", {
  visitor: ["body"],
  builder: ["body", "async"],
  aliases: ["Expression"],
  fields: {
    body: {
      validate: (0, _utils.assertNodeType)("BlockStatement")
    },
    async: {
      validate: (0, _utils.assertValueType)("boolean"),
      default: false
    }
  }
});
(0, _utils.default)("ExportDefaultSpecifier", {
  visitor: ["exported"],
  aliases: ["ModuleSpecifier"],
  fields: {
    exported: {
      validate: (0, _utils.assertNodeType)("Identifier")
    }
  }
});
(0, _utils.default)("RecordExpression", {
  visitor: ["properties"],
  aliases: ["Expression"],
  fields: {
    properties: {
      validate: (0, _utils.chain)((0, _utils.assertValueType)("array"), (0, _utils.assertEach)((0, _utils.assertNodeType)("ObjectProperty", "SpreadElement")))
    }
  }
});
(0, _utils.default)("TupleExpression", {
  fields: {
    elements: {
      validate: (0, _utils.chain)((0, _utils.assertValueType)("array"), (0, _utils.assertEach)((0, _utils.assertNodeType)("Expression", "SpreadElement"))),
      default: []
    }
  },
  visitor: ["elements"],
  aliases: ["Expression"]
});
(0, _utils.default)("DecimalLiteral", {
  builder: ["value"],
  fields: {
    value: {
      validate: (0, _utils.assertValueType)("string")
    }
  },
  aliases: ["Expression", "Pureish", "Literal", "Immutable"]
});
(0, _utils.default)("ModuleExpression", {
  visitor: ["body"],
  fields: {
    body: {
      validate: (0, _utils.assertNodeType)("Program")
    }
  },
  aliases: ["Expression"]
});
(0, _utils.default)("TopicReference", {
  aliases: ["Expression"]
});
(0, _utils.default)("PipelineTopicExpression", {
  builder: ["expression"],
  visitor: ["expression"],
  fields: {
    expression: {
      validate: (0, _utils.assertNodeType)("Expression")
    }
  },
  aliases: ["Expression"]
});
(0, _utils.default)("PipelineBareFunction", {
  builder: ["callee"],
  visitor: ["callee"],
  fields: {
    callee: {
      validate: (0, _utils.assertNodeType)("Expression")
    }
  },
  aliases: ["Expression"]
});
(0, _utils.default)("PipelinePrimaryTopicReference", {
  aliases: ["Expression"]
});

//# sourceMappingURL=experimental.js.map
