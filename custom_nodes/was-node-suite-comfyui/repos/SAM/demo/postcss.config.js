// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

const tailwindcss = require("tailwindcss");
module.exports = {
  plugins: ["postcss-preset-env", 'tailwindcss/nesting', tailwindcss],
};
