import getPrototypeOf from "./getPrototypeOf.js";
import isNativeReflectConstruct from "./isNativeReflectConstruct.js";
import possibleConstructorReturn from "./possibleConstructorReturn.js";
export default function _callSuper(t, o, e) {
  return o = getPrototypeOf(o), possibleConstructorReturn(t, isNativeReflectConstruct() ? Reflect.construct(o, e || [], getPrototypeOf(t).constructor) : o.apply(t, e));
}