#!/usr/bin/env python3
"""Format each authoritative THE OCEAN shot sentence with Z.AI GLM-5.3."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

H3_REPO = Path("/home/op/ai/minimax-h3")
sys.path.insert(0, str(H3_REPO))

from minimax_h3_story.formatter import SubprocessParagraphFormatter  # noqa: E402

ROOT = Path("/home/op/ai/ComfyUI")
SOURCE = ROOT / "docs/prompts/THE-OCEAN_minimax-h3-prompts.md"
FORMATTER_SCRIPT = ROOT / "tools/minimax_h3_prompt_formatter.py"
OUTPUT = ROOT / "logs/the-ocean-glm53-sentence-prompts-20260816"
CACHE = OUTPUT / "cache"
MANIFEST = OUTPUT / "manifest.json"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"
MODEL = "glm-5.3"
MAX_WORKERS = 3
MAX_ATTEMPTS = 3
STYLE = (
    "Cinematic music video; atmospheric progressive-techno visual rhythm; warm gold "
    "and deep sapphire-blue grade with Egyptian gold accents; anamorphic flares, "
    "volumetric god rays, shimmering particles, film grain, shallow depth of field, "
    "slow motion. The render is silent: no dialogue, narrator, captions, logos, "
    "watermarks, diegetic sound, or generated music."
)

_lock = threading.Lock()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(
            r".+?(?:[.!?…](?:[\"”’])?(?=\s|$)|$)",
            text.strip(),
            flags=re.DOTALL,
        )
        if match.group(0).strip()
    ]


def parse_source(markdown: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pattern = re.compile(
        r"^## SCENE (\d+) — ([^|\n]+) \| ([^|\n]+) \| ([^\n]+)$"
        r".*?^```text\n(.*?)^```$",
        flags=re.MULTILINE | re.DOTALL,
    )
    scenes: list[dict[str, object]] = []
    units: list[dict[str, object]] = []
    global_index = 0
    for match in pattern.finditer(markdown):
        scene = int(match.group(1))
        duration_label = match.group(4).strip()
        nominal_seconds = 30 if duration_label.startswith("30 s") else 15
        if scene == 12:
            nominal_seconds = 3
        body = match.group(5).strip()
        visual = re.search(
            r"integrated_multimodal_description:\s*(.*?)\n\noverall_soundscape:",
            body,
            flags=re.DOTALL,
        )
        if not visual:
            raise RuntimeError(f"scene {scene} has no integrated visual description")
        text = visual.group(1).strip()
        labels = list(re.finditer(r"\[Shot (\d+)\]", text))
        expected_shots = 3 if duration_label.startswith("30 s") else 1
        if [int(item.group(1)) for item in labels] != list(range(1, expected_shots + 1)):
            raise RuntimeError(f"scene {scene} has invalid shot labels")
        scene_units: list[list[int]] = []
        for index, label in enumerate(labels):
            end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
            source = text[label.end() : end].strip()
            source = re.sub(r"^At\s+00:\d{2}\.\d{3},\s*", "", source)
            source = re.sub(r"^the (?:camera|shot) cuts? to\s+", "", source, flags=re.I)
            if not source:
                raise RuntimeError(f"scene {scene} shot {index + 1} is empty")
            sentences = split_sentences(source)
            sentence_units: list[int] = []
            for sentence_index, sentence in enumerate(sentences, 1):
                global_index += 1
                unit = {
                    "unit": global_index,
                    "scene": scene,
                    "shot": index + 1,
                    "sentence_in_shot": sentence_index,
                    "sentences_in_shot": len(sentences),
                    "source": sentence,
                    "source_sha256": sha256_text(sentence),
                }
                units.append(unit)
                sentence_units.append(global_index)
            scene_units.append(sentence_units)
        scenes.append(
            {
                "scene": scene,
                "title": match.group(2).strip(),
                "timeline": match.group(3).strip(),
                "duration_label": duration_label,
                "nominal_seconds": nominal_seconds,
                "render_seconds": 30 if expected_shots == 3 else 15,
                "shots": expected_shots,
                "units": scene_units,
            }
        )
    if [item["scene"] for item in scenes] != list(range(1, 13)):
        raise RuntimeError("expected THE OCEAN scenes 1 through 12")
    if len(units) <= 24:
        raise RuntimeError(
            "strict sentence parsing must produce more source sentences than the 24 shot units"
        )
    return scenes, units


def archive_payload(scenes, units, records, status):
    return {
        "schema_version": 2,
        "status": status,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "formatter": {
            "backend": "zai-openai-compatible",
            "endpoint": ENDPOINT,
            "model": MODEL,
            "script": str(FORMATTER_SCRIPT),
            "script_sha256": hashlib.sha256(FORMATTER_SCRIPT.read_bytes()).hexdigest(),
            "granularity": "one independent call per grammatical source sentence",
            "max_workers": MAX_WORKERS,
        },
        "scene_count": len(scenes),
        "sentence_count": len(units),
        "scenes": scenes,
        "records": sorted(records.values(), key=lambda item: item["unit"]),
    }


def write_archive(scenes, units, records, status):
    payload = archive_payload(scenes, units, records, status)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(MANIFEST)


def locate_cache_record(unit, formatted):
    matches = []
    for path in CACHE.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provenance = payload.get("response_provenance") or {}
        if (
            payload.get("source_sha256") == unit["source_sha256"]
            and payload.get("formatted_prompt") == formatted
            and payload.get("model") == MODEL
            and provenance.get("provider_model") == MODEL
            and provenance.get("provider_response_id")
        ):
            matches.append((path, provenance))
    if len(matches) != 1:
        raise RuntimeError(
            f"unit {unit['unit']} has {len(matches)} provider-auditable cache records; expected 1"
        )
    path, provenance = matches[0]
    return {
        "cache_record": str(path),
        "cache_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "response_provenance": provenance,
    }


def format_unit(unit, units):
    formatter = SubprocessParagraphFormatter(
        script_path=FORMATTER_SCRIPT,
        python_binary=Path("/home/op/miniconda3/bin/python3"),
        endpoint=ENDPOINT,
        model=MODEL,
        backend="script",
        timeout_seconds=540,
        cache_dir=CACHE,
        granularity="paragraph",
    )
    prior = units[unit["unit"] - 2]["source"] if unit["unit"] > 1 else ""
    error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            output = formatter.format_paragraph(
                paragraph=unit["source"],
                paragraph_index=unit["unit"],
                paragraph_count=len(units),
                shots=1,
                language="English",
                style=STYLE,
                continuity_context=prior,
                planning_prompts=[],
                must_include=[
                    "Preserve Facundo and Keri's established character designs and the ordered Ocean story event.",
                    "Keep the shot silent; all sound and music are N/A for later track mixing.",
                ],
                must_avoid=["dialogue", "narration", "captions", "logos", "watermarks"],
            )[0]
            cache_provenance = locate_cache_record(unit, output)
            return {
                **unit,
                **cache_provenance,
                "status": "PASS",
                "attempt": attempt,
                "formatted": output,
                "formatted_sha256": sha256_text(output),
                "model": MODEL,
                "endpoint": ENDPOINT,
            }
        except Exception as exc:  # bounded retry; archive final failure
            error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)
    raise RuntimeError(f"unit {unit['unit']} failed after {MAX_ATTEMPTS} attempts: {error}")


def restore_records(units):
    if not MANIFEST.is_file():
        return {}
    try:
        prior = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {unit["unit"]: unit for unit in units}
    restored = {}
    for record in prior.get("records", []):
        unit = expected.get(record.get("unit"))
        cache_path = Path(record.get("cache_record", ""))
        if (
            unit
            and record.get("status") == "PASS"
            and record.get("model") == MODEL
            and record.get("source_sha256") == unit["source_sha256"]
            and record.get("response_provenance", {}).get("provider_model") == MODEL
            and record.get("response_provenance", {}).get("provider_response_id")
            and cache_path.is_file()
            and hashlib.sha256(cache_path.read_bytes()).hexdigest()
            == record.get("cache_record_sha256")
        ):
            restored[int(unit["unit"])] = record
    return restored


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenes, units = parse_source(SOURCE.read_text(encoding="utf-8"))
    records: dict[int, dict[str, object]] = restore_records(units)
    write_archive(scenes, units, records, "RUNNING")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(format_unit, unit, units): unit
            for unit in units
            if unit["unit"] not in records
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                record = future.result()
            except Exception as error:
                with _lock:
                    records[unit["unit"]] = {
                        **unit,
                        "status": "FAILED",
                        "error": f"{type(error).__name__}: {error}",
                        "model": MODEL,
                        "endpoint": ENDPOINT,
                    }
                    write_archive(scenes, units, records, "FAILED")
                for pending in futures:
                    pending.cancel()
                raise
            with _lock:
                records[record["unit"]] = record
                write_archive(scenes, units, records, "RUNNING")
                print(
                    f"GLM53_SENTENCE_PASS unit={record['unit']}/{len(units)} "
                    f"scene={record['scene']} shot={record['shot']} "
                    f"sentence={record['sentence_in_shot']}/{record['sentences_in_shot']}",
                    flush=True,
                )
    if len(records) != len(units):
        raise RuntimeError(
            f"formatter completed only {len(records)} of {len(units)} sentences"
        )
    by_unit = records
    for scene in scenes:
        parts = []
        for shot, sentence_units in enumerate(scene["units"], 1):
            bodies = [
                re.sub(r"^\[Shot\s+1\]\s*", "", str(by_unit[index]["formatted"]).strip())
                for index in sentence_units
            ]
            parts.append(f"[Shot {shot}] " + " ".join(bodies))
        output = OUTPUT / f"scene-{scene['scene']:02d}-glm-5.3-sentence.txt"
        output.write_text("\n---\n".join(parts) + "\n", encoding="utf-8")
        scene["prompt"] = str(output)
        scene["prompt_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        scene["model"] = MODEL
        scene["formatter_granularity"] = "sentence"
    write_archive(scenes, units, records, "DONE")
    print(f"PROMPT_GATE=PASS scenes={len(scenes)} sentences={len(records)} model={MODEL}")


if __name__ == "__main__":
    main()
