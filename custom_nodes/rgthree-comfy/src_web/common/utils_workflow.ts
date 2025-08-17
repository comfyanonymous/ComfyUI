import type {ISerialisedGraph} from "@comfyorg/frontend";
import type {ComfyApiFormat} from "typings/comfy.js";

import {getResolver} from "./shared_utils.js";
import {getPngMetadata, getWebpMetadata} from "./comfyui_shim.js";

/**
 * Parses the workflow JSON and do any necessary cleanup.
 */
function parseWorkflowJson(stringJson?: string) {
  stringJson = stringJson || "null";
  // Starting around August 2024 the serialized JSON started to get messy and contained `NaN` (for
  // an is_changed property, specifically). NaN is not parseable, so we'll get those on out of there
  // and cleanup anything else we need.
  stringJson = stringJson.replace(/:\s*NaN/g, ": null");
  return JSON.parse(stringJson);
}

export async function tryToGetWorkflowDataFromEvent(
  e: DragEvent,
): Promise<{workflow: ISerialisedGraph | null; prompt: ComfyApiFormat | null}> {
  let work;
  for (const file of e.dataTransfer?.files || []) {
    const data = await tryToGetWorkflowDataFromFile(file);
    if (data.workflow || data.prompt) {
      return data;
    }
  }
  const validTypes = ["text/uri-list", "text/x-moz-url"];
  const match = (e.dataTransfer?.types || []).find((t) => validTypes.find((v) => t === v));
  if (match) {
    const uri = e.dataTransfer!.getData(match)?.split("\n")?.[0];
    if (uri) {
      return tryToGetWorkflowDataFromFile(await (await fetch(uri)).blob());
    }
  }
  return {workflow: null, prompt: null};
}

export async function tryToGetWorkflowDataFromFile(
  file: File | Blob,
): Promise<{workflow: ISerialisedGraph | null; prompt: ComfyApiFormat | null}> {
  if (file.type === "image/png") {
    const pngInfo = await getPngMetadata(file);
    return {
      workflow: parseWorkflowJson(pngInfo?.workflow),
      prompt: parseWorkflowJson(pngInfo?.prompt),
    };
  }

  if (file.type === "image/webp") {
    const pngInfo = await getWebpMetadata(file);
    // Support loading workflows from that webp custom node.
    const workflow = parseWorkflowJson(pngInfo?.workflow || pngInfo?.Workflow || "null");
    const prompt = parseWorkflowJson(pngInfo?.prompt || pngInfo?.Prompt || "null");
    return {workflow, prompt};
  }

  if (file.type === "application/json" || (file as File).name?.endsWith(".json")) {
    const resolver = getResolver<{workflow: any; prompt: any}>();
    const reader = new FileReader();
    reader.onload = async () => {
      const json = parseWorkflowJson(reader.result as string);
      const isApiJson = Object.values(json).every((v: any) => v.class_type);
      const prompt = isApiJson ? json : null;
      const workflow = !isApiJson && !json?.templates ? json : null;
      return {workflow, prompt};
    };
    return resolver.promise;
  }
  return {workflow: null, prompt: null};
}
