import AsyncGenerator from "./AsyncGenerator.js";
export default function _wrapAsyncGenerator(fn) {
  return function () {
    return new AsyncGenerator(fn.apply(this, arguments));
  };
}