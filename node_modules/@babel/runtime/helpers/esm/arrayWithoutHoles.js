import arrayLikeToArray from "./arrayLikeToArray.js";
export default function _arrayWithoutHoles(arr) {
  if (Array.isArray(arr)) return arrayLikeToArray(arr);
}