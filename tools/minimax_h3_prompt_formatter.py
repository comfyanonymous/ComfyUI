#!/home/op/miniconda3/bin/python3
"""Rewrite user intent into the prompt contract expected by MiniMax H3.

This is deliberately an operator-side tool: ComfyUI never makes an internet or
LLM request while executing a graph.  The caller receives a validated prompt
before it is allowed to queue the graph.
"""

import argparse
import base64
import functools
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_ENDPOINT = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.3"
DEFAULT_TIMEOUT = 1800
DEFAULT_REASONING_EFFORT = "low"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
XAI_API_KEY_ENV = "XAI_API_KEY"
ZAI_API_KEY_ENVS = ("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY")
COMFY_INPUT = Path(__file__).resolve().parents[1] / "input"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
GUIDE_ROOT = Path(__file__).resolve().parents[1] / "docs" / "minimax-h3"
GUIDE_FILES = {
    "base": (
        "VIDEO_PROMPT_WRITING_GUIDE_base_en.md",
        "2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc",
    ),
    "reference": (
        "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
        "1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7",
    ),
}
DIRECTOR_FILE = (
    "MINIMAX_H3_PROMPT_DIRECTOR.md",
    "1ba57f010bf890992868cc54eeef79dbbec5174c3399f467d4b579f7c5b39eb2",
)
DIRECTOR_PATH_ENV = "MINIMAX_H3_PROMPT_DIRECTOR_PATH"
DIRECTOR_SHA256_ENV = "MINIMAX_H3_PROMPT_DIRECTOR_SHA256"
MAX_PROMPT_CHARS = 7000
BASE_HEADERS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
I2VA_ALIGNMENT = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
REFERENCE_HEADERS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
QUOTED_TEXT = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
REF2VA_MULTISHOT_ROLE = (
    "<Picture 1> supplies recurring identity, facial appearance, hair, clothing, "
    "and visual style only; it is not the opening frame and does not control pose, "
    "composition, camera, location, objects, or action."
)
MAX_SPOKEN_WORDS_PER_SEGMENT = 30
LAST_RESPONSE_PROVENANCE = {}
MAX_FORMAT_ATTEMPTS = 3


class PromptFormatError(RuntimeError):
    """The rewriter returned a prompt that does not satisfy the queue contract."""


UNREQUESTED_SANITIZATION_PHRASES = (
    "safely",
    "harmlessly",
    "unharmed",
    "no one is hurt",
    "nobody is hurt",
    "without injury",
    "without injuries",
    "non-graphic",
    "no blood or gore",
)

INTENT_ANCHORS = (
    (
        re.compile(r"\b(dead|lifeless|corpse|carcass|deceased|muert[oa]s?|cad[aá]ver)\b", re.IGNORECASE),
        re.compile(r"\b(dead|lifeless|corpse|carcass|deceased)\b", re.IGNORECASE),
        "death state",
    ),
    (
        re.compile(r"\b(explod(?:e|es|ed|ing)|explosion|detonat(?:e|es|ed|ing)|explot(?:a|ar|ando|ó)|explosi[oó]n)\b", re.IGNORECASE),
        re.compile(r"\b(explod(?:e|es|ed|ing)|explosion|detonat(?:e|es|ed|ing)|detonation)\b", re.IGNORECASE),
        "explosion",
    ),
)

DEATH_IDIOM = re.compile(
    r"\bdead\s+(?:quiet|silent|silence|calm|tired|center|serious|set)\b",
    re.IGNORECASE,
)


def quoted_spans(text):
    """Return user-supplied quoted phrases which must remain byte-for-byte intact."""
    return [next(group for group in match.groups() if group is not None) for match in QUOTED_TEXT.finditer(text)]


def base_contract(image_aware=False):
    alignment = f"{I2VA_ALIGNMENT}\n\n" if image_aware else ""
    return f"""Return only this H3 base prompt contract, with these headers spelled and ordered exactly:
{alignment}{BASE_HEADERS[0]}
<a coherent visual-and-diegetic-audio narrative>

{BASE_HEADERS[1]}
<1-4 sentences of ambient sound only; no music>

{BASE_HEADERS[2]}
<1-3 sentences of non-diegetic music, or 'none'>"""


def reference_contract():
    return f"""Return only this official H3 full-reference contract, with these headers spelled and ordered exactly:
{REFERENCE_HEADERS[0]}
<define <Subject N> and <Picture 1>; state that <Picture 1> is the first frame of [Shot 1]>

{REFERENCE_HEADERS[1]}
<[keyframe completion] followed by one short paragraph>

{REFERENCE_HEADERS[2]}
<one line per reference label with the official relationship marker>

{REFERENCE_HEADERS[3]}
<English playback-order description using [Shot N], stable references and speakers; preserve dialogue language>

{REFERENCE_HEADERS[4]}
<ambient and physical sounds only>

{REFERENCE_HEADERS[5]}
<audience-only music, or N/A>"""


@functools.lru_cache(maxsize=None)
def load_guide(kind):
    try:
        filename, expected_sha256 = GUIDE_FILES[kind]
    except KeyError as error:
        raise PromptFormatError(f"Unknown MiniMax H3 guide kind: {kind}") from error
    path = GUIDE_ROOT / filename
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PromptFormatError(f"Required MiniMax H3 prompt guide is unavailable: {path}") from error
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PromptFormatError(
            f"MiniMax H3 prompt guide hash mismatch for {path}: {actual_sha256}"
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromptFormatError(f"MiniMax H3 prompt guide is not valid UTF-8: {path}") from error


@functools.lru_cache(maxsize=1)
def load_director():
    override = os.environ.get(DIRECTOR_PATH_ENV, "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        expected_sha256 = os.environ.get(DIRECTOR_SHA256_ENV, "").strip()
        if not expected_sha256:
            raise PromptFormatError(
                f"{DIRECTOR_SHA256_ENV} is required when {DIRECTOR_PATH_ENV} is set"
            )
    else:
        filename, expected_sha256 = DIRECTOR_FILE
        path = GUIDE_ROOT / filename
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PromptFormatError(f"Required MiniMax H3 Prompt Director is unavailable: {path}") from error
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PromptFormatError(
            f"MiniMax H3 Prompt Director hash mismatch for {path}: {actual_sha256}"
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromptFormatError(f"MiniMax H3 Prompt Director is not valid UTF-8: {path}") from error


def guide_context(reference):
    sections = [
        "LOCAL MASTER MINIMAX H3 PROMPT DIRECTOR:\n" + load_director(),
        "OFFICIAL MINIMAX H3 BASE GUIDE:\n" + load_guide("base"),
    ]
    if reference:
        sections.append("OFFICIAL MINIMAX H3 FULL-REFERENCE GUIDE:\n" + load_guide("reference"))
    return "\n\n".join(sections)


def system_prompt(
    mode,
    reference,
    shots,
    image_aware=False,
    reference_asset=False,
):
    common = f"""You are MINIMAX H3 PROMPT DIRECTOR, a dedicated audiovisual prompt compiler rather than a general chatbot. Compile the user's intent into a directly queueable H3 prompt. Do not add commentary, markdown, explanations, assumptions, alternatives, or a title. Preserve every quoted phrase explicitly supplied as dialogue, lyrics, or visible text exactly, including punctuation; do not turn figurative quotation marks, labels, thoughts, or source headings into invented speech or visible text. Put spoken dialogue in <d>[Language]...</d>, using the dialogue's actual language name, and use stable speaker identifiers such as (S1). Do not invent visible text, logos, watermarks, or subtitles unless explicitly requested. Describe camera movement naturally, including direction, speed, and amplitude where relevant. Keep reference labels consistent and do not contradict supplied reference material. Plan and validate internally; never expose reasoning. Keep the finished prompt at or below {MAX_PROMPT_CHARS} characters.

CONTENT FIDELITY IS MANDATORY. Preserve the user's requested facts, subject states, actions, causal sequence, intensity, damage, injuries, death, destruction, tone, and consequences exactly. Do not sanitize, soften, euphemize, moralize, de-escalate, make safer, or substitute a less intense event. Never convert dead into living, injured into unharmed, violent impact into a gentle landing, an explosion into a safe fracture, or destruction into harmless slapstick. Do not add safety qualifiers, disclaimers, survivals, recovery, calm behavior, lack of injury, or absence of blood/gore unless the user explicitly requested them. Formatting may add concrete cinematic and audiovisual detail only when it is compatible with every requested fact. If exact preservation is impossible, do not invent a sanitized alternative.

Treat the following downloaded MiniMaxAI/MiniMax-H3 guides as authoritative. Follow their shot, camera, dialogue, audio, keyframe, and reference rules. The exact output contract after the guides selects the applicable format.

{guide_context(reference)}"""
    if reference and mode == "multishot":
        if image_aware:
            common += """

The actual <Picture 1> is attached to the user message. Inspect it directly at high
detail. Use only visibly supported recurring identity, facial appearance, hair,
clothing, and visual style. Do not carry its pose, composition, camera, location,
objects, or action into the generated scene unless the user's text independently
requires them. It is persistent REF2VA identity/style conditioning, never an opening
keyframe.
"""
        elif reference_asset:
            common += """

The selected <Picture 1> has been validated locally and will be supplied directly to
the ComfyUI REF2VA workflow, but this text-only formatter cannot inspect its pixels.
Do not claim to see it and do not invent pixel-derived appearance, pose, composition,
camera, location, objects, or action. Treat it only as persistent identity/style
conditioning; the workflow is the pixel authority.
"""
        common += """
Every multishot segment must include this exact sentence once:
""" + REF2VA_MULTISHOT_ROLE
    elif image_aware and reference:
        common += """

The actual <Picture 1> is attached to the user message. Inspect it directly at high detail. Accurately describe every visible person and their non-sensitive physical appearance, hair, clothing, accessories, pose, expression, and position; all important objects and their spatial relationships; the composition, camera angle, lens/framing, lighting, colors, materials, and environment. Do not guess names, identity, ethnicity, private traits, hidden objects, or facts that are not visually supported. Integrate this visual description concretely into subject_definitions, retention_analysis, and detailed_description so the generated motion begins from the real image. The image never overrides the user's requested future action or outcome."""
    elif image_aware:
        common += f"""

The actual <Picture 1> is attached to the user message and is the first frame of [Shot 1]. Inspect it directly at high detail. Accurately describe every visible person and their non-sensitive physical appearance, hair, clothing, accessories, pose, expression, and position; all important objects and their spatial relationships; the composition, camera angle, lens/framing, lighting, colors, materials, and environment. Do not guess names, identity, ethnicity, private traits, hidden objects, or facts that are not visually supported. Begin the output with this exact line: {I2VA_ALIGNMENT} Integrate the visual description into integrated_multimodal_description so motion develops forward from the real image. The image never overrides the user's requested future action or outcome."""
    if mode == "base":
        contract = reference_contract() if reference else base_contract(image_aware=image_aware)
        return f"""{common}

{contract}

Give a concrete, temporally coherent scene: visual composition, subject, setting, action, camera behavior, and diegetic sound. Keep sound events synchronized to their causes. Put dialogue exactly where it is spoken. Do not put music in overall_soundscape."""
    if mode == "multishot":
        return f"""{common}

MULTISHOT OUTPUT OVERRIDE: The guide examples above provide audiovisual advice,
not the output envelope for this local sampler. Do not emit any base/reference
headers, including `integrated_multimodal_description:`, `overall_soundscape:`,
`non_diegetic_music:`, `subject_definitions:`, `summary:`,
`retention_analysis:`, or `detailed_description:`. The first non-whitespace text
must be `[Shot 1]`. Put all visual action, camera, and synchronized sound directly
in that segment's prose. A reference-role sentence belongs inside each segment,
never before its `[Shot N]` label.

Return exactly {shots} self-contained H3 multishot narrative segment(s), separated only by a line containing --- (three hyphens). This is the local H3MultishotSampler execution contract: segment N maps to exactly one generated sampler shot and must contain exactly one [Shot N] label; never add an internal cut, another shot label, or a cut timestamp inside a segment. Do not use the base-prompt headers or JSON. Each segment must be a coherent visual-and-diegetic-audio narrative with composition, subject, setting, action, camera behavior, and synchronized sound. Keep character identity and any reference image anchor stable across segments. There must be no preamble or epilogue.

When narration or voiceover is requested, the visible characters must physically act
out the events and must never mouth or recite the narration. Use a disembodied
off-screen narrator with no on-screen body and a stable speaker ID distinct from all
visible characters. Immediately before every narrated <d> block use the exact phrase
`says in an off-screen voiceover:`. Immediately after that block, state that the lips
of every visible character remain completely closed. On-screen dialogue is allowed
only when the source explicitly assigns spoken words to that character.
When a quoted span is governed by `says`, `said`, `asks`, `shouts`, `dice`, `dijo`,
`dicho`, `decir`, or another explicit speech verb, assign it to that visible character's own
`<d>` block and do not duplicate it in narrator voiceover. Never use `mouth`, `mouths`,
`mouthing`, `lip-sync`, `recite`, or `reciting` in the formatted output. Never describe
a visible character as talking, speaking, whispering, shouting, singing, or otherwise
vocalizing without an adjacent source-verbatim `<d>` block.

When the request contains an `AUTHORITATIVE SOURCE PARAGRAPH` section, treat its
unquoted prose as visual staging direction rather than a transcript. Show that prose
through performance, blocking, props, reactions, camera, and synchronized action
sound; never copy or paraphrase it into `<d>`. Only verbatim quoted spans from that
paragraph that are explicitly governed by a speech verb (or its complete short heading
when explicitly required) may be spoken. Never invent narration or voiceover when the
authoritative paragraph does not explicitly request an off-screen narrator or voiceover.
Keep every segment at or below {MAX_SPOKEN_WORDS_PER_SEGMENT} total spoken words.

When a quoted source passage contains multiple sentences and cannot fit naturally in one segment, distribute its verbatim sentences across consecutive segments. Split only between complete sentences, keep the same stable speaker ID, and never split one sentence across the --- sampler boundary."""
    raise ValueError(f"Unknown mode: {mode}")


def extract_completion(response):
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise PromptFormatError("LLM response did not contain choices[0].message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise PromptFormatError("LLM returned an empty prompt")
    return content.strip()


def request_headers(endpoint):
    """Build request headers without persisting or exposing provider credentials."""
    headers = {"Content-Type": "application/json"}
    hostname = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    if hostname == "api.deepseek.com":
        api_key = os.environ.get(DEEPSEEK_API_KEY_ENV, "").strip()
        if not api_key:
            raise PromptFormatError(
                f"{DEEPSEEK_API_KEY_ENV} is required for the DeepSeek prompt formatter"
            )
        headers["Authorization"] = f"Bearer {api_key}"
    elif hostname == "api.x.ai":
        api_key = os.environ.get(XAI_API_KEY_ENV, "").strip()
        if not api_key:
            raise PromptFormatError(
                f"{XAI_API_KEY_ENV} is required for the xAI prompt formatter"
            )
        headers["Authorization"] = f"Bearer {api_key}"
    elif hostname == "api.z.ai":
        api_key = next(
            (
                os.environ.get(variable, "").strip()
                for variable in ZAI_API_KEY_ENVS
                if os.environ.get(variable, "").strip()
            ),
            "",
        )
        if not api_key:
            raise PromptFormatError(
                "ZAI_API_KEY (or GLM_API_KEY/Z_AI_API_KEY) is required for the "
                "Z.AI prompt formatter"
            )
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_local_qwen_proxy(endpoint):
    parsed = urllib.parse.urlparse(endpoint)
    return (parsed.hostname or "").lower() in {
        "127.0.0.1",
        "localhost",
        "100.85.200.51",
    } and parsed.port == 12434


@functools.lru_cache(maxsize=32)
def model_supports_images(endpoint, model):
    """Read the local proxy inventory instead of pretending a text model saw pixels."""
    if not _is_local_qwen_proxy(endpoint):
        return True
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/models",
        headers=request_headers(endpoint),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            inventory = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise PromptFormatError(f"Could not inspect local formatter model capabilities: {error}") from error
    models = inventory.get("data") if isinstance(inventory, dict) else None
    if not isinstance(models, list):
        raise PromptFormatError("Local formatter proxy returned an invalid model inventory")
    selected = next(
        (item for item in models if isinstance(item, dict) and item.get("id") == model),
        None,
    )
    if selected is None:
        raise PromptFormatError(f"Formatter model is not available from the local proxy: {model}")
    modalities = selected.get("modalities")
    inputs = modalities.get("input") if isinstance(modalities, dict) else None
    return isinstance(inputs, list) and "image" in inputs


def redact_api_key(text):
    redacted = text
    for variable in (DEEPSEEK_API_KEY_ENV, XAI_API_KEY_ENV, *ZAI_API_KEY_ENVS):
        api_key = os.environ.get(variable, "").strip()
        if api_key:
            redacted = redacted.replace(api_key, "<redacted>")
    return redacted


def user_rewrite_message(raw_prompt, correction=None):
    rules = (
        "Rewrite the idea only into the required MiniMax H3 movie-scene prompt contract. "
        "Preserve every requested fact, subject state, action, causal sequence, intensity, "
        "damage, injury, death, destruction, tone, dialogue, and consequence exactly. "
        "Do not sanitize, soften, moralize, de-escalate, add safety language, or substitute "
        "a safer or less intense event. Add only compatible cinematic, camera, timing, and "
        "synchronized audio detail, and follow the supplied official MiniMax H3 rules."
    )
    message = (
        "Compile this raw creative intent into the exact MiniMax H3 contract selected by the "
        f'system prompt: "{raw_prompt}"\n\nMandatory fidelity rules: {rules}'
    )
    if correction:
        message += f"\n\nCorrect a previous invalid draft without changing the idea: {correction}"
    return message


def resolve_reference_image(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        input_candidate = (COMFY_INPUT / path).resolve()
        path = cwd_candidate if cwd_candidate.is_file() else input_candidate
    else:
        path = path.resolve()
    if not path.is_file():
        raise PromptFormatError(f"Reference image is not a readable file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise PromptFormatError(f"Reference image is empty: {path}")
    if size > MAX_IMAGE_BYTES:
        raise PromptFormatError(
            f"Reference image exceeds the formatter's {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit: {path}"
        )
    prefix = path.read_bytes()[:12]
    for signature, mime_type in IMAGE_SIGNATURES:
        if prefix.startswith(signature):
            return path, mime_type
    raise PromptFormatError("Reference image must be a PNG or JPEG detected by file signature")


def multimodal_user_content(raw_prompt, correction, reference_image=None):
    text = user_rewrite_message(raw_prompt, correction)
    if reference_image is None:
        return text
    path, mime_type = resolve_reference_image(reference_image)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            },
        },
    ]


def call_llm(endpoint, model, raw_prompt, mode, reference, shots, timeout, correction=None,
             max_tokens=8192, reference_image=None):
    global LAST_RESPONSE_PROVENANCE
    reference_asset = reference_image is not None
    attached_reference = reference_image
    if reference_asset:
        resolve_reference_image(reference_image)
        if not model_supports_images(endpoint, model):
            attached_reference = None
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt(
                    mode,
                    reference,
                    shots,
                    image_aware=attached_reference is not None,
                    reference_asset=reference_asset,
                ),
            },
            {
                "role": "user",
                "content": multimodal_user_content(
                    raw_prompt, correction, attached_reference
                ),
            },
        ],
        "max_tokens": max_tokens,
    }
    hostname = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    if hostname == "api.x.ai":
        payload["reasoning_effort"] = DEFAULT_REASONING_EFFORT
    elif hostname == "api.z.ai":
        pass
    elif _is_local_qwen_proxy(endpoint) and model.endswith("-general"):
        pass
    else:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "max"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(endpoint),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            LAST_RESPONSE_PROVENANCE = {
                "provider_response_id": payload.get("id"),
                "provider_model": payload.get("model"),
                "provider_created": payload.get("created"),
                "provider_system_fingerprint": payload.get("system_fingerprint"),
            }
            return extract_completion(payload)
    except urllib.error.HTTPError as error:
        detail = redact_api_key(error.read().decode(errors="replace"))
        raise PromptFormatError(f"LLM returned HTTP {error.code}: {detail}") from error
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise PromptFormatError(f"LLM request failed: {error}") from error


def validate_headers(prompt, headers, contract_name):
    lines = prompt.splitlines()
    header_indexes = []
    for header in headers:
        matches = [index for index, line in enumerate(lines) if line.strip().startswith(header)]
        if len(matches) != 1:
            raise PromptFormatError(
                f"Formatted {contract_name} prompt must contain exactly one {header!r} header"
            )
        header_indexes.append(matches[0])
    if header_indexes != sorted(header_indexes) or len(set(header_indexes)) != len(header_indexes):
        raise PromptFormatError(f"Formatted {contract_name} prompt has headers in the wrong order")


def validate_base(prompt, reference, source, image_aware=False):
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PromptFormatError(
            f"Formatted prompt exceeds the {MAX_PROMPT_CHARS}-character director limit"
        )
    if reference:
        validate_headers(prompt, REFERENCE_HEADERS, "reference")
        if "<Picture 1>" not in prompt or "[Shot 1]" not in prompt:
            raise PromptFormatError(
                "Formatted reference prompt must bind <Picture 1> to [Shot 1]"
            )
    else:
        validate_headers(prompt, BASE_HEADERS, "base")
        if image_aware:
            first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
            if first_line != I2VA_ALIGNMENT:
                raise PromptFormatError(
                    "Formatted I2VA prompt must begin with the exact Picture 1 alignment instruction"
                )
    validate_content_fidelity(prompt, source)
    validate_quoted_dialogue(prompt, source)


def validate_voiceover_contract(prompt, source):
    has_voiceover = False
    for match in re.finditer(r"<d>\[[^\]]+\].*?</d>", prompt, flags=re.DOTALL):
        prefix = prompt[max(0, match.start() - 320):match.start()]
        if not re.search(
            r"(?:off-screen\s+(?:narrator|voice)|voice-?over|voiceover)",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        has_voiceover = True
        if not re.search(
            r"says in an off-screen voiceover:\s*$", prefix, flags=re.IGNORECASE
        ):
            raise PromptFormatError(
                "Voiceover dialogue must use the exact phrase "
                "'says in an off-screen voiceover' immediately before <d>"
            )
        narrator_context = prefix[-220:]
        if re.search(r"\bnarrator\b", narrator_context, flags=re.IGNORECASE) and not re.search(
            r"(?:disembodied|unseen|no on-screen body)",
            narrator_context,
            flags=re.IGNORECASE,
        ):
            raise PromptFormatError(
                "An off-screen narrator must be explicitly disembodied and have no "
                "on-screen body"
            )
        suffix = prompt[match.end():match.end() + 220]
        if not re.match(
            r"\s*(?:while\s+)?[^.!?\n]{0,150}\blips?\b"
            r"[^.!?\n]{0,100}\b(?:remain|stays?|are kept)\b"
            r"[^.!?\n]{0,40}\bclosed\b",
            suffix,
            flags=re.IGNORECASE,
        ):
            raise PromptFormatError(
                "Every voiceover <d> block must be followed immediately by an "
                "explicit statement that visible lips remain completely closed"
            )
    if has_voiceover and re.search(
        r"\b(?:mouths?|mouthing|lip[- ]?sync(?:s|ed|ing)?|recit(?:e|es|ed|ing))\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        raise PromptFormatError(
            "A voiceover shot must not describe any visible character mouthing, "
            "lip-syncing, or reciting the narration"
        )
    authoritative = authoritative_scene_brief(source) or authoritative_paragraph(source)
    narration_source = authoritative if authoritative is not None else source
    if has_voiceover and not re.search(
        r"\b(?:narrat(?:e|es|ed|ing|ion|or)|voice-?over|off-screen voice)\b",
        narration_source,
        flags=re.IGNORECASE,
    ):
        raise PromptFormatError(
            "The source does not request narration or voiceover; stage its events "
            "through visible character action and use only explicitly attributed dialogue"
        )


def dialogue_payloads(prompt):
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"<d>\[[^\]]+\]\s*(.*?)</d>", prompt, flags=re.DOTALL
        )
    ]


def authoritative_paragraph(source):
    match = re.search(
        r"AUTHORITATIVE SOURCE PARAGRAPH \d+ OF \d+\n(.*?)\n\nFORMATTER JOB",
        source,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def authoritative_scene_brief(source):
    match = re.search(
        r"Creative Brief — source-faithful scene \d+ of \d+:\n(.*?)"
        r"\n\nLOCAL H3 MULTISHOT EXECUTION ADAPTER",
        source,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def required_dialogue_quotes(source):
    authoritative = authoritative_scene_brief(source) or authoritative_paragraph(source)
    if authoritative is None:
        return quoted_spans(source)
    required = []
    speech = re.compile(
        r"\b(?:says?|said|asks?|asked|shouts?|shouted|whispers?|whispered|"
        r"murmurs?|murmured|dice|dijo|dicho|decir|pregunta|preguntó|grita|gritó|"
        r"susurra|susurró|murmura|murmuró)\b",
        flags=re.IGNORECASE,
    )
    for match in QUOTED_TEXT.finditer(authoritative):
        quote = next(group for group in match.groups() if group is not None)
        before = authoritative[max(0, match.start() - 120):match.start()]
        after = authoritative[match.end():match.end() + 120]
        line_prefix = before.rsplit("\n", 1)[-1]
        sentence_before = re.split(r"[.!?…]\s*", before)[-1]
        sentence_after = re.split(r"[.!?…]", after, maxsplit=1)[0]
        attribution_bridge = re.search(
            r"\b(?:says?|said|asks?|asked|shouts?|shouted|whispers?|whispered|"
            r"murmurs?|murmured|dice|dijo|dicho|decir|pregunta|preguntó|grita|gritó|"
            r"susurra|susurró|murmura|murmuró)\b[^.!?…]{0,16}[.!?…]\s*$",
            before,
            flags=re.IGNORECASE,
        )
        explicit_dialogue = (
            bool(speech.search(sentence_before))
            or bool(speech.search(sentence_after))
            or bool(attribution_bridge)
            or bool(re.search(r"(?:^|\s)—\s*$", line_prefix))
        )
        if explicit_dialogue:
            required.append(quote)
    return required


def is_short_heading(text):
    words = text.split()
    return bool(words) and len(words) <= 12 and not re.search(r"[.!?…]", text)


def validate_paragraph_spoken_source(prompt, source):
    paragraph = authoritative_scene_brief(source) or authoritative_paragraph(source)
    if paragraph is None:
        return
    payloads = dialogue_payloads(prompt)
    heading = paragraph if is_short_heading(paragraph) else None
    quotes = required_dialogue_quotes(source)
    for payload in payloads:
        if heading and payload == heading:
            continue
        if any(payload in quote for quote in quotes):
            continue
        raise PromptFormatError(
            "Authoritative paragraph prose is visual staging, not dialogue: every "
            "<d> payload must be a verbatim span of source-quoted speech"
        )


def validate_vocal_actions_have_dialogue(segment, index):
    pattern = re.compile(
        r"\b(?:says?|speak(?:s|ing|spoke)?|talk(?:s|ed|ing)?|asks?|repl(?:y|ies|ied)|"
        r"shout(?:s|ed|ing)?|whisper(?:s|ed|ing)?|murmur(?:s|ed|ing)?|"
        r"chant(?:s|ed|ing)?|sing(?:s|ing|sang)?)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(segment):
        sentence_end = re.search(r"[.!?\n]", segment[match.end():])
        end = (
            match.end() + sentence_end.start() + 1
            if sentence_end is not None
            else len(segment)
        )
        if "<d>" not in segment[match.start():end]:
            sentence_start = max(
                segment.rfind(".", 0, match.start()),
                segment.rfind("!", 0, match.start()),
                segment.rfind("?", 0, match.start()),
                segment.rfind("\n", 0, match.start()),
            )
            sentence = segment[sentence_start + 1:end].strip()
            action_prefix = segment[sentence_start + 1:match.start()]
            if re.search(
                r"\b(?:leather|fabric|cloth|wind|leaves|paper|parchment|water|"
                r"flames?|fire|rain|floorboards?|metal|stone)\s+$",
                action_prefix,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(
                r"\b(?:no\s+one|nobody|neither\s+person|does\s+not|do\s+not|"
                r"did\s+not|never)\s*$",
                action_prefix,
                flags=re.IGNORECASE,
            ):
                continue
            raise PromptFormatError(
                f"Multishot segment {index} contains an untagged vocal action "
                f"{match.group(0)!r} in {sentence!r}; visible speech must have an "
                "adjacent <d> block"
            )


def validate_multishot(prompt, shots, source, reference=False, image_aware=False):
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PromptFormatError(
            f"Formatted prompt exceeds the {MAX_PROMPT_CHARS}-character director limit"
        )
    if any(header in prompt for header in BASE_HEADERS):
        raise PromptFormatError("Formatted multishot prompt incorrectly used the base prompt headers")
    segments = [segment.strip() for segment in re.split(r"^---$", prompt, flags=re.MULTILINE)]
    if len(segments) != shots or any(not segment for segment in segments):
        raise PromptFormatError(f"Formatted multishot prompt must contain exactly {shots} non-empty segment(s)")
    for index, segment in enumerate(segments, start=1):
        shot_labels = re.findall(r"\[Shot \d+\]", segment)
        expected = f"[Shot {index}]"
        if shot_labels != [expected]:
            raise PromptFormatError(
                f"Multishot segment {index} must contain exactly one {expected} label and no internal cuts"
            )
        if image_aware and reference and "<Picture 1>" not in segment:
            raise PromptFormatError(
                f"REF2VA multishot segment {index} must bind <Picture 1>"
            )
        if image_aware and reference and REF2VA_MULTISHOT_ROLE not in segment:
            raise PromptFormatError(
                f"REF2VA multishot segment {index} must preserve Picture 1 as an "
                "identity/style reference rather than an opening keyframe"
            )
        spoken_words = sum(
            len(re.findall(r"[\wÀ-ÖØ-öø-ÿ’'-]+", payload, flags=re.UNICODE))
            for payload in dialogue_payloads(segment)
        )
        if spoken_words > MAX_SPOKEN_WORDS_PER_SEGMENT:
            raise PromptFormatError(
                f"Multishot segment {index} has {spoken_words} spoken words; "
                f"maximum is {MAX_SPOKEN_WORDS_PER_SEGMENT}"
            )
        validate_vocal_actions_have_dialogue(segment, index)
    validate_paragraph_spoken_source(prompt, source)
    validate_voiceover_contract(prompt, source)
    validate_content_fidelity(prompt, source)
    validate_quoted_dialogue(prompt, source)


def validate_content_fidelity(prompt, source):
    """Reject common semantic sanitization introduced by the formatter."""
    source_folded = source.casefold()
    prompt_folded = prompt.casefold()
    literal_source = DEATH_IDIOM.sub("", source)
    introduced = [
        phrase for phrase in UNREQUESTED_SANITIZATION_PHRASES
        if phrase not in source_folded and phrase in prompt_folded
    ]
    if introduced:
        raise PromptFormatError(
            "Formatted prompt introduced unrequested sanitization: "
            + ", ".join(repr(phrase) for phrase in introduced)
        )

    for source_pattern, output_pattern, label in INTENT_ANCHORS:
        if source_pattern.search(literal_source) and not output_pattern.search(prompt):
            raise PromptFormatError(
                f"Formatted prompt dropped the user's explicit {label}"
            )

    if re.search(r"\b(dead|lifeless|corpse|carcass|deceased)\b", literal_source, re.IGNORECASE):
        if re.search(r"\b(living|alive)\b", prompt, re.IGNORECASE):
            raise PromptFormatError(
                "Formatted prompt contradicted the requested death state with living/alive"
            )

    scene_brief = authoritative_scene_brief(source)
    if scene_brief is not None:
        prior_match = re.search(
            r"PRIOR FORMATTER-APPROVED TERMINAL DESCRIPTION\n.*?\n"
            r"(.*?)\n\nINHERITED CONTINUITY",
            source,
            flags=re.DOTALL,
        )
        grounding = scene_brief + ("\n" + prior_match.group(1) if prior_match else "")
        gore = re.compile(
            r"\b(?:dead|death|corpse|headless|decapitat\w*|blood|bloody|gore|"
            r"brain matter|wounds?|severed|muert[oa]s?|muerte|cadáver(?:es)?|"
            r"decapit\w*|sangre|heridas?)\b",
            flags=re.IGNORECASE,
        )
        if has_positive_guarded_detail(prompt, gore.pattern) and not gore.search(grounding):
            raise PromptFormatError(
                "Formatted prompt invented unsupported death, injury, or gore"
            )
        if (
            "Do not fire it" in source or "never a gunshot" in source
        ) and re.search(
            r"\b(?:gunshots?|muzzle flash|bullets?|fires?|firing|shoots?|shooting)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            raise PromptFormatError("Formatted prompt invented a forbidden weapon discharge")
        if "MINOR AGE INVARIANT:" in source:
            if re.search(
                r"\b(?:young man|adult man|grown man|middle-aged man)\b",
                prompt,
                flags=re.IGNORECASE,
            ):
                raise PromptFormatError("Formatted prompt aged up a minor")
            if re.search(
                r"\b(?:womanizer|seduct\w*|sexual\w*|flirt\w*|lust\w*|"
                r"desir(?:e|es|ed|ing))\b|"
                r"(?:girl|student).{0,80}(?:greenhouse|falls? in step)|"
                r"(?:greenhouse).{0,80}(?:girl|student)",
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                raise PromptFormatError("Formatted prompt sexualized a minor")

        guarded_details = (
            (
                "The source supplies no speaker identity",
                r"\b(?:kitchen|envelope|letter|table|adult(?:s)?)\b",
                "invented an unsupported first-scene person, place, or prop",
            ),
            (
                "Do not add a vacant lot",
                r"\b(?:vacant lot|street|cars?|fence)\b",
                "invented an unsupported discovery location",
            ),
            (
                "never a cigarette",
                r"\bcigarettes?\b",
                "replaced the source faso with a cigarette",
            ),
            (
                "faso means cannabis or marijuana",
                r"\b(?:tobacco|cigarettes?|cigars?)\b",
                "replaced cannabis faso with tobacco",
            ),
            (
                "without the prior Hufflepuff student",
                r"\b(?:Hufflepuff|sleeper|sleeping student|shoulder contact)\b",
                "carried a prior secondary subject across a hard cut",
            ),
        )
        for marker, pattern, error in guarded_details:
            if marker in source and has_positive_guarded_detail(prompt, pattern):
                raise PromptFormatError(f"Formatted prompt {error}")


def has_positive_guarded_detail(text, pattern):
    detail = re.compile(pattern, flags=re.IGNORECASE)
    negation = re.compile(
        r"\b(?:no|not|never|without|neither|nor|does not|do not|is not|are not)\b"
        r"[^.!?]{0,40}$",
        flags=re.IGNORECASE,
    )
    for match in detail.finditer(text):
        sentence_prefix = re.split(
            r"[.!?]", text[max(0, match.start() - 100):match.start()]
        )[-1]
        if negation.search(sentence_prefix):
            continue
        return True
    return False


def validate_quoted_dialogue(prompt, source):
    dialogue_blocks = [
        _plain_dialogue(block)
        for block in re.findall(r"<d>\[[^\]\n]+\](.*?)</d>", prompt, flags=re.DOTALL)
    ]
    combined_dialogue = _plain_dialogue(" ".join(dialogue_blocks))
    for quote in required_dialogue_quotes(source):
        normalized_quote = _plain_dialogue(quote)
        if normalized_quote not in combined_dialogue:
            raise PromptFormatError(f"Formatted prompt changed or dropped quoted user text: {quote!r}")
        sentences = [
            _plain_dialogue(sentence)
            for sentence in re.split(r"(?<=[.!?…])\s+", quote)
            if sentence.strip()
        ]
        if not all(
            any(sentence in dialogue for dialogue in dialogue_blocks)
            for sentence in sentences
        ):
            raise PromptFormatError(
                "Formatted prompt must preserve every quoted sentence inside one "
                f"language-tagged <d> block and split only at a sentence boundary: {quote!r}"
            )


def _plain_dialogue(value):
    without_tags = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", without_tags).strip()


def normalize_structural_contract(prompt):
    """Canonicalize non-dialogue voiceover closure wording required by H3."""
    parts = re.split(r"(<d>\[[^\]]+\].*?</d>)", prompt, flags=re.DOTALL)
    ownership = (
        r"(?P<owner>(?:Harry's|his|her|their|the boy's|the child's|"
        r"the visible character's)\s+)"
    )
    closure = re.compile(
        ownership
        + r"mouth\s+(?:remains?|stays?|is\s+kept|are\s+kept)\s+"
        r"(?:completely\s+)?closed\b",
        flags=re.IGNORECASE,
    )
    keeps_closed = re.compile(
        r"(?P<subject>Harry|he|she|the boy|the child)\s+keeps\s+"
        r"(?:his|her|their)\s+mouth\s+(?:completely\s+)?closed\b",
        flags=re.IGNORECASE,
    )
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(
            r"\bclosed[- ]mouth(?:ed)?\b",
            "closed-lipped",
            parts[index],
            flags=re.IGNORECASE,
        )
        parts[index] = closure.sub(
            lambda match: match.group("owner") + "lips remain completely closed",
            parts[index],
        )
        parts[index] = keeps_closed.sub(
            lambda match: match.group("subject")
            + " keeps their lips completely closed",
            parts[index],
        )
    return "".join(parts)


def format_prompt(endpoint, model, raw_prompt, mode, reference=False, shots=1, timeout=180,
                  max_tokens=8192, reference_image=None):
    if not raw_prompt or not raw_prompt.strip():
        raise PromptFormatError("A non-empty user prompt is required")
    if shots < 1:
        raise PromptFormatError("shots must be positive")
    if reference_image is not None and mode == "multishot" and not reference:
        raise PromptFormatError(
            "A multishot reference image requires reference=True"
        )
    correction = None
    for attempt in range(MAX_FORMAT_ATTEMPTS):
        result = call_llm(
            endpoint, model, raw_prompt.strip(), mode, reference, shots, timeout,
            correction, max_tokens=max_tokens, reference_image=reference_image,
        )
        result = normalize_structural_contract(result)
        try:
            if mode == "base":
                validate_base(result, reference, raw_prompt, image_aware=reference_image is not None)
            else:
                validate_multishot(
                    result,
                    shots,
                    raw_prompt,
                    reference=reference,
                    image_aware=reference_image is not None,
                )
            return result
        except PromptFormatError as error:
            correction = str(error)
            if attempt == MAX_FORMAT_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "multishot"), required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--stdin", action="store_true", help="Read the raw prompt from standard input")
    parser.add_argument("--reference", action="store_true", help="Require the <Picture 1> reference alignment contract")
    parser.add_argument(
        "--reference-image",
        help=(
            "Workflow reference image; attached only when the selected formatter "
            "model advertises image input"
        ),
    )
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--json", action="store_true", help="Emit a JSON object for the queue helper")
    args = parser.parse_args()
    if bool(args.prompt) == args.stdin:
        parser.error("provide exactly one of --prompt or --stdin")
    if args.reference_image and args.mode == "multishot" and not args.reference:
        parser.error("multishot --reference-image requires --reference")
    return args


def main():
    args = parse_args()
    raw_prompt = sys.stdin.read() if args.stdin else args.prompt
    try:
        formatted = format_prompt(
            args.endpoint,
            args.model,
            raw_prompt,
            args.mode,
            reference=args.reference,
            shots=args.shots,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            reference_image=args.reference_image,
        )
    except PromptFormatError as error:
        raise SystemExit(f"MiniMax H3 prompt formatting failed: {error}") from error
    if args.json:
        print(json.dumps({
            "mode": args.mode,
            "model": args.model,
            "formatted_prompt": formatted,
            "response_provenance": LAST_RESPONSE_PROVENANCE,
        }))
    else:
        print(formatted)


if __name__ == "__main__":
    main()
