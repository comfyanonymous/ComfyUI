import json, time, urllib.request, uuid, sys

HOST = "http://127.0.0.1:8188"
WF = r"C:\Users\Owner\ComfyUIDirectML\workflows\animatediff_txt2vid.json"

# 1) Load UI workflow
wf = json.load(open(WF))
nodes = {n["id"]: n for n in wf["nodes"]}
linkmap = {l[0]: (l[1], l[2]) for l in wf["links"]}

# 2) Pull API schemas so we can fill defaults the UI export omitted
allinfo = json.loads(urllib.request.urlopen(HOST + "/object_info", timeout=30).read())

def api_schema(class_type):
    info = allinfo.get(class_type, {})
    req = info.get("input", {}).get("required", {})
    opt = info.get("input", {}).get("optional", {})
    valid = set(req) | set(opt)
    defaults = {}
    for k, v in {**req, **opt}.items():
        meta = v[1] if (isinstance(v, list) and len(v) > 1) else {}
        defaults[k] = meta.get("default", None)
    return valid, defaults

# 3) Build API prompt, merging only inputs valid for the INSTALLED node version
prompt = {}
for nid, n in nodes.items():
    ctype = n["type"]
    valid, defaults = api_schema(ctype)
    inputs = {}
    # seed from API defaults for required slots
    for k, dv in defaults.items():
        if k in valid:
            inputs[k] = dv
    # apply UI widget overrides / links, but only if the slot exists on this node
    for inp in n.get("inputs", []):
        name = inp["name"]
        if name not in valid:
            continue  # skip version-drifted keys (e.g. deprecation_warning)
        if inp.get("link") is not None:
            src_node, src_idx = linkmap[inp["link"]]
            inputs[name] = [str(src_node), src_idx]
        else:
            w = inp.get("widget")
            if w is not None:
                inputs[name] = w["value"]
    # drop hidden-only injected keys
    for hidden in ("extra_pnginfo", "prompt", "unique_id"):
        inputs.pop(hidden, None)
    # Explicit fixes for node slots whose API "default" is None/invalid
    if ctype == "ADE_AnimateDiffSamplingSettings" and inputs.get("seed_gen") is None:
        inputs["seed_gen"] = "comfy"
    if ctype == "VHS_VideoCombine" and inputs.get("loop_count") is None:
        inputs["loop_count"] = 0
    # The installed AnimateDiff maps this node to a LEGACY class whose
    # load_mm_and_inject_params() rejects 'deprecation_warning' (even as None).
    if ctype == "ADE_AnimateDiffLoaderWithContext":
        inputs.pop("deprecation_warning", None)
    # SHRINK to fit torch-directml's ~1GB VRAM cap (AMD 6800 XT shows only 1024MB)
    if ctype == "EmptyLatentImage":
        inputs["width"] = 256
        inputs["height"] = 256
        inputs["batch_size"] = 8
    if ctype == "KSampler":
        inputs["steps"] = 12
        inputs["cfg"] = 7.0
        inputs["sampler_name"] = "euler"
        inputs["scheduler"] = "normal"
    if ctype == "VHS_VideoCombine":
        inputs["frame_rate"] = 8
    prompt[str(nid)] = {"class_type": 'ADE_AnimateDiffLoaderWithContext' if ctype == 'ADE_AnimateDiffLoaderWithContext' else ctype, "inputs": inputs}

# 4) Submit
client_id = str(uuid.uuid4())
body = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
req = urllib.request.Request(HOST + "/prompt", data=body,
                              headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    pid = json.loads(resp.read())["prompt_id"]
    print("QUEUED prompt_id:", pid)
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED", e.code, e.read().decode()[:3000])
    sys.exit(1)

# 5) Poll for completion
for _ in range(300):  # up to ~25 min
    time.sleep(5)
    try:
        h = json.loads(urllib.request.urlopen(HOST + "/history", timeout=15).read())
    except Exception as e:
        print("history poll err:", e); continue
    if pid in h:
        entry = h[pid]
        st = entry.get("status", {})
        if "outputs" in entry and entry["outputs"]:
            print("=== DONE ===")
            for nid, out in entry["outputs"].items():
                for key in ("videos", "gifs"):
                    for item in out.get(key, []):
                        print(key.upper(), item.get("filename"), item.get("type"), item.get("subfolder"))
            sys.exit(0)
        if st.get("status_str") == "error":
            print("=== ERROR ===")
            print(json.dumps(st, indent=2)[:4000])
            sys.exit(2)
print("TIMEOUT: still rendering after 25 min")
