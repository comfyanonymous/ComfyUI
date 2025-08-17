
const FUNCTIONAL = {
    seed_input_regex : "seed|随机种",
    prompt_regex     : "(_|\\b)pos(itive|_|\\b)|^prompt|正面",
    negative_regex   : "(_|\\b)neg(ative|_|\\b)|负面",
}

const DISPLAY = {
    title : "title",
    input : "input",
    group : "group",
    Group : "Group",
    Color : "Colour",
    Priority : "Priority",
    prompt_regexes_text : "Prompt Everywhere",
    "No restrictions" : "No restrictions",
    "Send only within group" : "Send only within group", 
    "Send only not within group" : "Send only not within group",
    "Send only to same color" : "Send only to same color", 
    "Send only to different color" : "Send only to different color",
}

export const GROUP_RESTRICTION_OPTIONS = [i18n("No restrictions"), i18n("Send only within group"), i18n("Send only not within group")]
export const COLOR_RESTRICTION_OPTIONS = [i18n("No restrictions"), i18n("Send only to same color"), i18n("Send only to different color")]

const toTitleCase = (phrase) => {
  return phrase
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export function i18n(v, extras) {
    var r = DISPLAY[v] || v
    if (extras?.titlecase) r = toTitleCase(r)
    return r
}

export function default_regex(v) {
    return FUNCTIONAL[v] 
}
