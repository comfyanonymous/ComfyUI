import cv2
import torch
import numpy as np
import base64
import os
import io
from PIL import Image
from openai import OpenAI
import json
import datetime
import subprocess

class VideoFrameSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "sample_interval_sec": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "max_dimension": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "timestamps", "total_duration")
    FUNCTION = "sample_frames"
    CATEGORY = "SoccerHighlights"
    
    def sample_frames(self, video_path, sample_interval_sec, max_dimension):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        interval_frames = int(fps * sample_interval_sec)
        
        frames = []
        timestamps = []
        
        current_frame = 0
        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to reduce memory usage (target max_dimension)
            h, w, _ = frame.shape
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor (H, W, C) -> (1, H, W, C)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            
            frames.append(frame_tensor)
            timestamps.append(current_frame / fps)
            
            current_frame += interval_frames
            
        cap.release()
        
        if not frames:
            raise ValueError("No frames sampled from video")
            
        # Stack all frames into a batch tensor (B, H, W, C)
        images_batch = torch.stack(frames)
        
        return (images_batch, timestamps, duration)

class LLMFrameScorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
                "model": ("STRING", {"default": "gpt-4o", "multiline": False}),
                "prompt": ("STRING", {"default": "Analyze this soccer frame. Score it from 0 to 10 based on excitement or goal likelihood. Return ONLY a JSON object with keys 'score' (float) and 'reason' (string).", "multiline": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("scores", "reasoning")
    FUNCTION = "score_frames"
    CATEGORY = "SoccerHighlights"

    def score_frames(self, images, api_key, base_url, model, prompt):
        client = OpenAI(api_key=api_key, base_url=base_url)
        scores = []
        reasons = []

        # Process each image in the batch
        # Note: Depending on batch size and rate limits, this might need batching or async.
        # For simplicity, we process sequentially here.
        
        total = images.shape[0]
        
        print(f"Scoring {total} frames...")

        for i in range(total):
            # Convert tensor to PIL Image for base64 encoding
            img_tensor = images[i] # (H, W, C)
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}",
                                        "detail": "low" # Use low detail to save tokens/cost
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=100,
                    response_format={ "type": "json_object" }
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                scores.append(float(result.get("score", 0.0)))
                reasons.append(result.get("reason", ""))
                print(f"Frame {i+1}/{total}: Score {scores[-1]}")
                
            except Exception as e:
                print(f"Error scoring frame {i}: {e}")
                scores.append(0.0)
                reasons.append(f"Error: {str(e)}")

        return (scores, reasons)

class HighlightScriptGenerator:

    @staticmethod
    def _parse_reasoning_entry(reasoning_entry):
        reason_text = str(reasoning_entry).strip()
        team_bias = "neutral"
        swing = "neutral"
        try:
            data = json.loads(reason_text)
            if isinstance(data, dict):
                reason_text = str(data.get("reason", reason_text)).strip()
                team_bias = str(data.get("team_bias", team_bias)).strip() or "neutral"
                swing = str(data.get("swing", swing)).strip() or "neutral"
        except Exception:
            pass
        return {
            "reason": reason_text,
            "team_bias": team_bias,
            "swing": swing,
        }

    @staticmethod
    def _effective_score(score, team_bias, swing):
        bonus = 0.0
        if team_bias == "active_team_positive":
            bonus += 0.75
        elif team_bias == "opponent_positive":
            bonus -= 0.75

        if swing == "negative_to_positive":
            bonus += 1.0
        elif swing == "positive_to_negative":
            bonus -= 0.5

        return max(0.0, min(10.0, float(score) + bonus))

    @staticmethod
    def _find_best_ffmpeg():
        """PATH上の全ffmpegを検索し、最新バージョンのパスを返す"""
        import subprocess, re
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        best_path = "ffmpeg"
        best_ver = (0, 0, 0)

        for d in path_dirs:
            exe = os.path.join(d, "ffmpeg.exe")
            if not os.path.isfile(exe):
                continue
            try:
                result = subprocess.run(
                    [exe, "-version"], capture_output=True, text=True, timeout=5
                )
                m = re.search(r'ffmpeg version (\d+)\.(\d+)\.?(\d*)', result.stdout)
                if m:
                    ver = tuple(int(x) if x else 0 for x in m.groups())
                    if ver > best_ver:
                        best_ver = ver
                        best_path = exe
            except Exception:
                continue

        ver_str = ".".join(str(v) for v in best_ver) if best_ver != (0, 0, 0) else "unknown"
        print(f"[FFmpeg] Using: {best_path} (version {ver_str})")
        return best_path, best_ver

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "timestamps": ("FLOAT", {"forceInput": True}),
                "scores": ("FLOAT", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "padding_sec": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 1.0}),
                "max_clip_sec": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 600.0, "step": 5.0}),
                "min_gap_sec": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 120.0, "step": 5.0}),
                "min_consecutive_frames": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "output_mode": (["video_clips", "images_only", "both"], {"default": "video_clips"}),
                "auto_execute": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reasoning": ("STRING", {"forceInput": True}),
                "player_info": ("STRING", {"forceInput": True}),
                "active_team": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("status",)
    FUNCTION = "generate_script"
    CATEGORY = "SoccerHighlights"
    OUTPUT_NODE = True
    
    def generate_script(self, video_path, timestamps, scores, threshold, padding_sec,
                        max_clip_sec, min_gap_sec, min_consecutive_frames,
                        output_mode, auto_execute, reasoning=None, player_info=None, active_team=None):
        
        # Ensure inputs are lists
        if not isinstance(timestamps, list):
            timestamps = [timestamps]
        if not isinstance(scores, list):
            scores = [scores]
            
        output_dir = "output"
        video_filename = os.path.basename(video_path)
        video_name_only = os.path.splitext(video_filename)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{video_name_only}"
        timestamp_dir = os.path.join(output_dir, folder_name)
        os.makedirs(timestamp_dir, exist_ok=True)

        reasoning_meta = []
        effective_scores = [float(s) for s in scores]
        if reasoning:
            if not isinstance(reasoning, list):
                reasoning = [reasoning]
            if len(reasoning) == len(timestamps):
                reasoning_meta = [self._parse_reasoning_entry(r) for r in reasoning]
                effective_scores = [
                    self._effective_score(score, meta["team_bias"], meta["swing"])
                    for score, meta in zip(scores, reasoning_meta)
                ]
        
        # Save scores to CSV
        scores_csv_path = os.path.join(timestamp_dir, "scores.csv")
        try:
            with open(scores_csv_path, "w", encoding="utf-8") as f:
                has_reasoning = len(reasoning_meta) == len(timestamps)
                
                if has_reasoning:
                    f.write("Timestamp,Score,EffectiveScore,TeamBias,Swing,Reasoning\n")
                    for t, s, eff, meta in zip(timestamps, scores, effective_scores, reasoning_meta):
                        clean_r = meta["reason"].replace("\n", " ").replace(",", ";").strip()
                        clean_bias = meta["team_bias"].replace(",", ";").strip()
                        clean_swing = meta["swing"].replace(",", ";").strip()
                        f.write(f"{t:.2f},{s:.2f},{eff:.2f},{clean_bias},{clean_swing},{clean_r}\n")
                else:
                    f.write("Timestamp,Score\n")
                    for t, s in zip(timestamps, scores):
                        f.write(f"{t:.2f},{s:.2f}\n")
        except Exception as e:
            print(f"Error saving scores CSV: {e}")

        # Save player_info CSV
        has_player_info = False
        player_info_list = None
        active_team_list = None
        if player_info:
            if not isinstance(player_info, list):
                player_info = [player_info]
            if len(player_info) == len(timestamps):
                player_info_list = player_info
                has_player_info = True
        if active_team:
            if not isinstance(active_team, list):
                active_team = [active_team]
            if len(active_team) == len(timestamps):
                active_team_list = active_team

        if has_player_info:
            player_csv_path = os.path.join(timestamp_dir, "player_info.csv")
            try:
                with open(player_csv_path, "w", encoding="utf-8") as f:
                    f.write("Timestamp,Score,ActiveTeam,JerseyNumbers_A,JerseyNumbers_B,PlayerCount_A,PlayerCount_B,KeyAction\n")
                    for idx, (t, s, pi) in enumerate(zip(timestamps, scores, player_info_list)):
                        team = active_team_list[idx] if active_team_list else "unknown"
                        try:
                            info = json.loads(pi)
                            jn_a = str(info.get("jersey_numbers", {}).get("A", []))
                            jn_b = str(info.get("jersey_numbers", {}).get("B", []))
                            pc_a = info.get("player_count", {}).get("A", 0)
                            pc_b = info.get("player_count", {}).get("B", 0)
                            action = str(info.get("key_action", "")).replace(",", ";").replace("\n", " ")
                        except Exception:
                            jn_a, jn_b, pc_a, pc_b, action = "", "", 0, 0, pi.replace(",", ";")[:80]
                        f.write(f"{t:.2f},{s:.2f},{team},{jn_a},{jn_b},{pc_a},{pc_b},{action}\n")
            except Exception as e:
                print(f"Error saving player_info CSV: {e}")

        # Filter indices above threshold
        high_score_indices = []
        run = 0
        for i, s in enumerate(effective_scores):
            if s >= threshold:
                run += 1
                if run >= min_consecutive_frames:
                    for back in range(run - 1, -1, -1):
                        candidate = i - back
                        if candidate not in high_score_indices:
                            high_score_indices.append(candidate)
            else:
                run = 0
        high_score_indices.sort()

        if not high_score_indices:
            max_score = max(effective_scores) if effective_scores else 0.0
            report_path = os.path.join(timestamp_dir, "report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"No highlights found.\n")
                f.write(f"Video: {video_path}\nThreshold: {threshold}\nMax Effective Score: {max_score}\n")
            return (f"No highlights found. Max score: {max_score:.2f}.",)

        # Merge intervals
        raw_intervals = []
        for idx in high_score_indices:
            t = timestamps[idx]
            raw_intervals.append((max(0, t - padding_sec), t + padding_sec))

        raw_intervals.sort()
        merged_intervals = []
        if raw_intervals:
            current_start, current_end = raw_intervals[0]
            for next_start, next_end in raw_intervals[1:]:
                if next_start - current_end < min_gap_sec:
                    current_end = max(current_end, next_end)
                else:
                    merged_intervals.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_intervals.append((current_start, current_end))

        # Split long clips
        final_intervals = []
        for start, end in merged_intervals:
            while end - start > max_clip_sec:
                final_intervals.append((start, start + max_clip_sec))
                start = start + max_clip_sec
            if end - start > 0:
                final_intervals.append((start, end))
        merged_intervals = final_intervals

        # Generate Batch Script
        ffmpeg_bin, ffmpeg_ver = HighlightScriptGenerator._find_best_ffmpeg()
        lines = [
            "@echo off",
            "setlocal enabledelayedexpansion",
            f"set \"VIDEO_PATH={video_path}\"",
            f"set \"FFMPEG_BIN={ffmpeg_bin}\"",
            f"set \"DEFAULT_MODE={output_mode}\"",
            "cd /d %~dp0",
            "",
            "rem Check command line argument for mode override",
            "set \"MODE=%~1\"",
            "if \"!MODE!\"==\"\" set \"MODE=%DEFAULT_MODE%\"",
            "",
            "echo [Highlight Script] Mode: !MODE!",
            "if \"!MODE!\"==\"images_only\" goto IMAGES",
            "if \"!MODE!\"==\"video_clips\" goto VIDEOS",
            "if \"!MODE!\"==\"both\" (",
            "  call :IMAGES",
            "  call :VIDEOS",
            "  goto END",
            ")",
            "echo Unknown mode: !MODE!",
            "goto END",
            "",
            ":IMAGES",
            "echo --- Extracting Peak Images ---",
            "if not exist preview_images mkdir preview_images"
        ]

        # Prepare per-interval data (peaks and clips)
        img_cmds = []
        vid_cmds = []
        concat_list_filename = "concat_list.txt"
        
        # 1. Images for verification (Filtered by threshold if images_only/both is selected)
        if output_mode in ["images_only", "both"]:
            # Extract scored points >= threshold for accuracy verification
            for i, (t, s) in enumerate(zip(timestamps, scores)):
                if s >= threshold:
                    img_name = f"preview_images/frame_{i:04d}_Time-{t:.1f}_Score-{s:.1f}.jpg"
                    img_cmds.append(f'"!FFMPEG_BIN!" -y -ss {t:.2f} -i "!VIDEO_PATH!" -frames:v 1 -q:v 2 "{img_name}"')
        else:
            # Just peaks for intervals if video_clips only (standard behavior previously)
            # Actually, if it's video_clips only, we don't need img_cmds unless the user switches mode.
            # To allow switching in batch, let's always include the peak images at least.
            for i, (start, end) in enumerate(merged_intervals):
                peak_time = start
                max_s = -1.0
                for t, s in zip(timestamps, scores):
                    if start <= t <= end:
                        if s > max_s:
                            max_s = s
                            peak_time = t
                img_name = f"preview_images/peak_{i:03d}_Score-{max_s:.1f}.jpg"
                img_cmds.append(f'"!FFMPEG_BIN!" -y -ss {peak_time:.2f} -i "!VIDEO_PATH!" -frames:v 1 -q:v 2 "{img_name}"')

        # 2. Video Clips
        for i, (start, end) in enumerate(merged_intervals):
            # Find max score for filename
            max_s = -1.0
            for t, s in zip(timestamps, scores):
                if start <= t <= end:
                    max_s = max(max_s, s)
            
            duration = end - start
            clip_name = f"output_clips/clip_{i:03d}_Score-{max_s:.1f}.mp4"
            if torch.cuda.is_available():
                enc = "-c:v h264_nvenc -preset p6 -tune hq -rc constqp -qp 20" if ffmpeg_ver >= (5,0,0) else "-c:v h264_nvenc -preset hq -cq 20"
            else:
                enc = "-c:v libx264 -crf 18 -preset fast"
            vid_cmds.append(f'"!FFMPEG_BIN!" -y -ss {start:.2f} -i "!VIDEO_PATH!" -t {duration:.2f} {enc} -c:a aac "{clip_name}"')
            vid_cmds.append(f"echo file '{clip_name}' >> {concat_list_filename}")

        lines += img_cmds
        lines += [
            "echo Images saved to preview_images/",
            "if \"%~1\"==\"\" if \"!MODE!\"==\"images_only\" pause",
            "exit /b",
            "",
            ":VIDEOS",
            "echo --- Processing Video Clips ---",
            "if not exist output_clips mkdir output_clips",
            f"if exist {concat_list_filename} del {concat_list_filename}"
        ]
        lines += vid_cmds
        lines += [
            f'"!FFMPEG_BIN!" -y -f concat -safe 0 -i {concat_list_filename} -c copy "highlight_video.mp4"',
            "echo Total highlight video saved to highlight_video.mp4",
            "if \"%~1\"==\"\" if \"!MODE!\"==\"video_clips\" pause",
            "exit /b",
            "",
            ":END",
            "if \"%~1\"==\"\" pause"
        ]

        script_content = "\n".join(lines)
        bat_file_path = os.path.join(timestamp_dir, "highlight_script.bat")
        with open(bat_file_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        status_msg = f"Saved batch file to: {os.path.abspath(bat_file_path)}"
        if auto_execute:
            try:
                subprocess.Popen(f'start "" "{os.path.abspath(bat_file_path)}"', shell=True)
                status_msg += "\nAuto-execution started."
            except Exception as e:
                status_msg += f"\nAuto-execution failed: {e}"

        return (status_msg,)


# ---------------------------------------------------------------------------
# LocalVLMScorerTemporal - Approach 2: multi-frame grid for scene understanding
# ---------------------------------------------------------------------------
class LocalVLMScorerTemporal:
    """前後フレームをグリッド合成してVLMに送り、時系列の動きを考慮した興奮度を算出する"""

    _model = None
    _tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": (
                        "Soccer match scene analysis. Score the excitement of the MIDDLE frame only (0-10):\n"
                        "0-2: Dead time — throw-in setup, players walking, non-play camera angle\n"
                        "3-4: Normal possession, routine pass, build-up play\n"
                        "5-6: Pressing, counter-attack, corner kick preparation\n"
                        "7-8: Shot on goal, tackle, dangerous chance, near miss\n"
                        "9-10: GOAL SCORED, penalty decision, red card, game-changing moment\n"
                        "IMPORTANT: A goalpost visible in the BACKGROUND does NOT increase the score.\n"
                        "Score based on PLAYER ACTION in the scene, not field geometry.\n"
                        "Reply with JSON only: {\"score\": float, \"reason\": \"brief description\"}"
                    ),
                    "multiline": True
                }),
                "context_frames": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("scores", "reasoning")
    FUNCTION = "score_frames"
    CATEGORY = "SoccerHighlights"

    @staticmethod
    def _make_horizontal_grid(pil_images, target_height=256):
        """複数PIL画像を同じ高さにリサイズして横並び結合"""
        resized = []
        for img in pil_images:
            ratio = target_height / img.height
            new_w = max(1, int(img.width * ratio))
            resized.append(img.resize((new_w, target_height), Image.LANCZOS))
        total_w = sum(r.width for r in resized)
        grid = Image.new("RGB", (total_w, target_height), color=(0, 0, 0))
        x = 0
        for r in resized:
            grid.paste(r, (x, 0))
            x += r.width
        return grid

    def score_frames(self, images, prompt, context_frames):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Reuse LocalVLMScorer cache if already loaded
        if LocalVLMScorerTemporal._model is None:
            if LocalVLMScorer._model is not None:
                LocalVLMScorerTemporal._model = LocalVLMScorer._model
                LocalVLMScorerTemporal._tokenizer = LocalVLMScorer._tokenizer
                print("LocalVLMScorerTemporal: reusing LocalVLMScorer model cache.")
            else:
                print("Loading Moondream2 model for LocalVLMScorerTemporal...")
                model_id = "vikhyatk/moondream2"
                try:
                    LocalVLMScorerTemporal._model = AutoModelForCausalLM.from_pretrained(
                        model_id, trust_remote_code=True,
                        torch_dtype=torch.float16
                    ).to("cuda")
                    LocalVLMScorerTemporal._tokenizer = AutoTokenizer.from_pretrained(model_id)
                    print("Moondream2 loaded for temporal scorer.")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    total = images.shape[0]
                    return ([0.0] * total, [f"Error loading model: {e}"] * total)

        model = LocalVLMScorerTemporal._model
        tokenizer = LocalVLMScorerTemporal._tokenizer

        scores = []
        reasons = []
        total = images.shape[0]
        ctx = context_frames
        print(f"Scoring {total} frames with LocalVLMScorerTemporal (context={ctx})...")

        import re

        for i in range(total):
            # Gather context indices
            indices = list(range(max(0, i - ctx), min(total, i + ctx + 1)))
            pil_frames = []
            for j in indices:
                img_np = (images[j].numpy() * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(img_np))

            # Build grid (single frame if context_frames==0)
            if len(pil_frames) == 1:
                grid_img = pil_frames[0]
            else:
                grid_img = LocalVLMScorerTemporal._make_horizontal_grid(pil_frames)

            try:
                enc = model.encode_image(grid_img)
                answer = model.answer_question(enc, prompt, tokenizer)

                # Try JSON parse first
                score = 0.0
                reason = answer
                try:
                    result = json.loads(answer)
                    score = float(result.get("score", 0.0))
                    reason = result.get("reason", answer)
                except Exception:
                    # Fallback: regex extract number
                    match = re.search(r'\b(10|[0-9](?:\.[0-9])?)\b', answer)
                    if match:
                        score = float(match.group(1))

                scores.append(score)
                reasons.append(reason)
                print(f"  Frame {i+1}/{total} [ctx {len(indices)} frames]: score={score:.1f}")

            except Exception as e:
                print(f"  Error scoring frame {i}: {e}")
                scores.append(0.0)
                reasons.append(f"Error: {e}")

        return (scores, reasons)


# ---------------------------------------------------------------------------
# PlayerInfoExtractor - jersey number recognition + auto team calibration
# ---------------------------------------------------------------------------
class PlayerInfoExtractor:
    """冒頭フレームのK-meansでチームカラーを自動学習し、全フレームの選手情報・背番号・主体チームを抽出"""

    _team_colors_cache = {}  # video_path -> (center_a_bgr, center_b_bgr, name_a, name_b)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "timestamps": ("FLOAT", {"forceInput": True}),
                "video_path": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "calibration_sec": ("INT", {"default": 30, "min": 5, "max": 120, "step": 5}),
                "context_frames": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("player_info", "active_team", "team_a_color", "team_b_color")
    FUNCTION = "extract_player_info"
    CATEGORY = "SoccerHighlights"

    @staticmethod
    def _bgr_to_color_name(bgr):
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v < 60:
            return "black"
        if s < 50:
            return "white" if v > 180 else "gray"
        if h < 10 or h > 170:
            return "red"
        if h < 25:
            return "orange"
        if h < 35:
            return "yellow"
        if h < 85:
            return "green"
        if h < 130:
            return "blue"
        if h < 150:
            return "purple"
        return "pink"

    def _calibrate_team_colors(self, video_path, calibration_sec, sample_interval_sec=5):
        """冒頭フレームから芝を除外してK-means(k=2)でチームカラーを自動検出"""
        if video_path in PlayerInfoExtractor._team_colors_cache:
            print(f"[TeamCalibration] Using cached team colors for: {video_path}")
            return PlayerInfoExtractor._team_colors_cache[video_path]

        print(f"[TeamCalibration] Calibrating team colors from first {calibration_sec}s of video...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        all_pixels = []

        for t_sec in range(0, calibration_sec, sample_interval_sec):
            frame_idx = int(t_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Resize for speed
            frame_small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            # Mask out green (grass): H 35-85, S>40, V>40
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            # Mask out very dark pixels (shadows, markings)
            dark_mask = hsv[:, :, 2] < 50
            combined_mask = (green_mask > 0) | dark_mask
            # Collect non-masked BGR pixels (subsampled)
            non_grass = frame_small[~combined_mask]
            if len(non_grass) > 0:
                step = max(1, len(non_grass) // 200)
                all_pixels.extend(non_grass[::step].tolist())

        cap.release()

        if len(all_pixels) < 10:
            print("[TeamCalibration] Not enough pixels. Defaulting to blue/red.")
            return None, None, "blue", "red"

        arr = np.array(all_pixels, dtype=np.float32)
        center_a, center_b, name_a, name_b = self._run_kmeans(arr)
        result = (center_a, center_b, name_a, name_b)
        PlayerInfoExtractor._team_colors_cache[video_path] = result
        print(f"[TeamCalibration] Team A: {name_a}, Team B: {name_b}")
        return result

    def _run_kmeans(self, arr):
        """scikit-learn が使えればK-means、なければnumpyフォールバック"""
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, n_init=5, random_state=0)
            km.fit(arr)
            counts = np.bincount(km.labels_)
            team_a_idx = int(np.argmax(counts))  # 多数派 = Team A
            team_b_idx = 1 - team_a_idx
            center_a = km.cluster_centers_[team_a_idx]
            center_b = km.cluster_centers_[team_b_idx]
        except ImportError:
            print("[TeamCalibration] scikit-learn not found. Using numpy fallback.")
            # Simple fallback: pick two random samples far apart
            idx1 = 0
            dists = np.linalg.norm(arr - arr[idx1], axis=1)
            idx2 = int(np.argmax(dists))
            center_a = arr[idx1]
            center_b = arr[idx2]

        name_a = self._bgr_to_color_name(center_a)
        name_b = self._bgr_to_color_name(center_b)
        # If both resolved to same color, add differentiation
        if name_a == name_b:
            name_b = name_b + "_2"
        return center_a, center_b, name_a, name_b

    def extract_player_info(self, images, timestamps, video_path, calibration_sec, context_frames):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Reuse model cache from LocalVLMScorer / LocalVLMScorerTemporal
        model = LocalVLMScorer._model or LocalVLMScorerTemporal._model
        tokenizer = LocalVLMScorer._tokenizer or LocalVLMScorerTemporal._tokenizer

        if model is None:
            print("Loading Moondream2 model for PlayerInfoExtractor...")
            model_id = "vikhyatk/moondream2"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                # Store in LocalVLMScorer cache for sharing
                LocalVLMScorer._model = model
                LocalVLMScorer._tokenizer = tokenizer
            except Exception as e:
                total = images.shape[0]
                err = f"Error loading model: {e}"
                return ([err] * total, ["unknown"] * total, "unknown", "unknown")

        if not isinstance(timestamps, list):
            timestamps = [timestamps]

        # Calibrate team colors
        center_a, center_b, name_a, name_b = self._calibrate_team_colors(
            video_path, calibration_sec
        )

        total = images.shape[0]
        ctx = context_frames
        print(f"[PlayerInfoExtractor] Processing {total} frames. Team A={name_a}, Team B={name_b}")

        import re

        PROMPT_TEMPLATE = (
            "You are analyzing a soccer match.\n"
            "Team A wears {name_a} jerseys (the focal/home team). "
            "Team B wears {name_b} jerseys.\n"
            "Frames are shown in chronological order (left to right).\n"
            "List all visible jersey numbers and identify each player's team.\n"
            "Return ONLY a JSON object:\n"
            "{{\"jersey_numbers\": {{\"A\": [list], \"B\": [list]}}, "
            "\"player_count\": {{\"A\": int, \"B\": int}}, "
            "\"key_action\": \"brief description e.g. #9 (Team A) shooting\", "
            "\"active_team\": \"A or B or both or unknown\"}}"
        )
        prompt = PROMPT_TEMPLATE.format(name_a=name_a, name_b=name_b)

        player_info_out = []
        active_team_out = []
        FALLBACK = json.dumps({
            "jersey_numbers": {"A": [], "B": []},
            "player_count": {"A": 0, "B": 0},
            "key_action": "parse error",
            "active_team": "unknown"
        })

        for i in range(total):
            indices = list(range(max(0, i - ctx), min(total, i + ctx + 1)))
            pil_frames = []
            for j in indices:
                img_np = (images[j].numpy() * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(img_np))

            if len(pil_frames) == 1:
                grid_img = pil_frames[0]
            else:
                grid_img = LocalVLMScorerTemporal._make_horizontal_grid(pil_frames)

            try:
                enc = model.encode_image(grid_img)
                answer = model.answer_question(enc, prompt, tokenizer)

                # Try to parse JSON from answer
                info_json = FALLBACK
                active = "unknown"
                try:
                    # Strip markdown fences if present
                    clean = re.sub(r'```[a-z]*', '', answer).strip().strip('`')
                    info = json.loads(clean)
                    active = info.get("active_team", "unknown")
                    info_json = json.dumps(info, ensure_ascii=False)
                except Exception:
                    # Fallback: try to extract active_team from text
                    m = re.search(r'active_team["\s:]+([ABab]+(?:oth)?)', answer)
                    if m:
                        active = m.group(1).capitalize()
                    info_json = json.dumps({"raw": answer, "active_team": active})

                player_info_out.append(info_json)
                active_team_out.append(active)
                print(f"  Frame {i+1}/{total}: active_team={active}")

            except Exception as e:
                print(f"  Error processing frame {i}: {e}")
                player_info_out.append(FALLBACK)
                active_team_out.append("unknown")

        return (player_info_out, active_team_out, name_a, name_b)


class AudioVolumeScorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "timestamps": ("FLOAT", {"forceInput": True}),
                "window_sec": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("scores",)
    FUNCTION = "score_audio"
    CATEGORY = "SoccerHighlights"

    def score_audio(self, video_path, timestamps, window_sec):
        import librosa
        import numpy as np

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Ensure timestamps is a list
        if not isinstance(timestamps, list):
            timestamps = [timestamps]

        print(f"Loading audio from {video_path}...")
        try:
            # Load audio (mono, resample to 22050Hz for speed)
            y, sr = librosa.load(video_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Return zero scores if audio loading fails
            return ([0.0] * len(timestamps),)

        scores = []
        
        # Calculate RMS energy for the whole track to normalize
        # Hop length for RMS calculation
        hop_length = 512
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # Frame times associated with RMS values
        times = librosa.times_like(rmse, sr=sr, hop_length=hop_length)
        
        max_rms = np.max(rmse) if np.max(rmse) > 0 else 1.0
        
        print(f"Scoring {len(timestamps)} timestamps based on audio volume...")
        
        for t in timestamps:
            if t < 0 or t > duration:
                scores.append(0.0)
                continue
                
            # Find indices in RMS array corresponding to the window [t - window/2, t + window/2]
            t_start = max(0, t - window_sec / 2)
            t_end = min(duration, t + window_sec / 2)
            
            # Find index range in 'times' array
            # We can use searchsorted efficiently because times is sorted
            idx_start = np.searchsorted(times, t_start)
            idx_end = np.searchsorted(times, t_end)
            
            if idx_start >= idx_end:
                local_rms = 0.0
            else:
                # Get max RMS in this window (or mean?) 
                # Max is better for detecting sudden cheers/whistles
                local_rms = np.max(rmse[idx_start:idx_end])
            
            # Normalize to 0-10
            score = (local_rms / max_rms) * 10.0
            scores.append(float(score))
            
        return (scores,)

class LocalVLMScorer:
    # Cache model to avoid reloading
    _model = None
    _tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image and rate the excitement level from 0 to 10.", "multiline": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("scores", "reasoning")
    FUNCTION = "score_frames"
    CATEGORY = "SoccerHighlights"

    def score_frames(self, images, prompt):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model only once
        if LocalVLMScorer._model is None:
            print("Loading Moondream2 model (latest)...")
            model_id = "vikhyatk/moondream2"
            # Removed revision to use latest
            try:
                LocalVLMScorer._model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to("cuda") # fp16 for faster inference
                LocalVLMScorer._tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"Moondream2 model loaded. Type: {type(LocalVLMScorer._model)}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return ([0.0]*len(images), [f"Error loading model: {e}"]*len(images))

        scores = []
        reasons = []
        
        total = images.shape[0]
        print(f"Scoring {total} frames with Local VLM...")

        for i in range(total):
            # Convert tensor to PIL Image
            img_tensor = images[i]
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            try:
                # Encode image
                enc_image = LocalVLMScorer._model.encode_image(img_pil)
                # Generate answer
                answer = LocalVLMScorer._model.answer_question(enc_image, prompt, LocalVLMScorer._tokenizer)
                
                # Extract score from answer (heuristic)
                # We expect the prompt to ask for a number or we parse it.
                # Let's try to find a number in the output.
                import re
                match = re.search(r'\b(10|[0-9](\.[0-9])?)\b', answer)
                if match:
                    score = float(match.group(1))
                else:
                    score = 0.0 # Could not find score
                
                scores.append(score)
                reasons.append(answer)
                print(f"Frame {i+1}/{total}: Score {score} - {answer[:50]}...")
                
            except Exception as e:
                print(f"Error scoring frame {i}: {e}")
                scores.append(0.0)
                reasons.append(f"Error: {e}")

        return (scores, reasons)

class ScoreCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scores_A": ("FLOAT", {"forceInput": True}),
                "scores_B": ("FLOAT", {"forceInput": True}),
                "weight_A": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "weight_B": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("combined_scores",)
    FUNCTION = "combine_scores"
    CATEGORY = "SoccerHighlights"

    def combine_scores(self, scores_A, scores_B, weight_A, weight_B, normalize=True):
        if not isinstance(scores_A, list): scores_A = [scores_A]
        if not isinstance(scores_B, list): scores_B = [scores_B]

        length = min(len(scores_A), len(scores_B))
        combined = []
        total_weight = (weight_A + weight_B) if (weight_A + weight_B) > 0 else 1.0

        for i in range(length):
            s_a = scores_A[i]
            s_b = scores_B[i]
            if normalize:
                # Weighted average → keeps result in 0-10 range
                final = (s_a * weight_A + s_b * weight_B) / total_weight
            else:
                # Raw weighted sum (legacy behaviour)
                final = s_a * weight_A + s_b * weight_B
            combined.append(round(final, 4))

        return (combined,)

class VideoDirectoryLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:\\path\\to\\videos_or_video.mp4", "multiline": False}),
                "extensions": ("STRING", {"default": "mp4,mkv,avi,mov", "multiline": False}),
                "select_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}), # -1 for all (batch), >=0 for specific
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_videos"
    CATEGORY = "SoccerHighlights"
    
    # IS_CHANGED ensures the node runs even if inputs haven't changed, 
    # but here we only want it to update if the directory content changes or user forces it.
    # Actually, if we return a list, ComfyUI handles the loop.
    # To force re-check of directory, we can use a dummy input or just rely on user refresh.
    
    def load_videos(self, directory_path, extensions, select_index):
        if not os.path.exists(directory_path):
            print(f"[VideoDirectoryLoader] Path not found: {directory_path}")
            return ([],)

        # ファイルパスが渡された場合→そのファイルだけを返す
        if os.path.isfile(directory_path):
            print(f"[VideoDirectoryLoader] Single file mode: {directory_path}")
            return ([directory_path],)

        # フォルダパスの場合→拡張子フィルタリングして一括処理
        ext_list = [e.strip().lower() for e in extensions.split(",")]
        files = []
        for f in os.listdir(directory_path):
            if f.startswith(".") or f.startswith("._"):
                continue
            if any(f.lower().endswith(ext) for ext in ext_list):
                files.append(os.path.join(directory_path, f))
        files.sort()

        if not files:
            return ([""],)

        if select_index >= 0:
            if select_index < len(files):
                return ([files[select_index]],)
            else:
                return ([files[-1]],)

        print(f"[VideoDirectoryLoader] Folder mode: {len(files)} files found")
        return (files,)

# ---------------------------------------------------------------------------
# GeminiVideoScorer - Gemini File API で動画を解析しハイライトタイムスタンプを返す
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_PROMPT = """
You are analyzing a soccer match video clip.

Your goal is to score all distinct moments across the clip while preferentially surfacing positive moments for the team we care about.

STEP 1: Infer the likely active team
- Examine the footage from the start of the clip until kickoff.
- Identify which team is shown most prominently during that pre-kickoff period.
- Treat that team as the likely active team only if the visual evidence is reasonably strong.
- If the evidence is weak or ambiguous, do not force the inference. In that case, score scenes neutrally.

STEP 2: Score all distinct moments across the clip
- Report ONLY events you can actually see. Do NOT invent or hallucinate events.
- Timestamps must be within the clip duration.
- Each entry must describe a distinct moment. Do not repeat the same phase of play many times.
- Include both uneventful and exciting periods so the clip is covered by scored moments.
- If several adjacent moments are effectively the same phase of play, you may merge them into one representative entry.
- For goals, shots, and dangerous attacks, set the timestamp at the START of the build-up or decisive action, not only the final touch.

TEAM-PREFERENCE RULES:
- If the likely active team is identifiable, assign relatively higher scores to positive moments for that team.
- Positive active-team moments include attacking progress, line-breaking passes, dribbles that beat pressure, dangerous chances, shots, goals, strong recoveries, blocks, and saves by that team.
- Assign relatively lower scores to positive moments for the opponent, unless they are objectively major match events.
- Major match events must still receive high scores even if they favor the opponent.
- If a sequence clearly shifts from the active team being under pressure to the active team creating danger, raise the score for that turning-point moment.
- If a sequence shifts the other way, you may still include it, but usually score it lower than an equally strong positive moment for the active team.
- Do not bias the output so strongly that obvious important events are ignored.

LABELING RULES:
- For each entry, include:
  - "team_bias": one of "active_team_positive", "opponent_positive", or "neutral"
  - "swing": one of "negative_to_positive", "positive_to_negative", "stable", or "neutral"
- Use "negative_to_positive" when the active team moves from a disadvantaged or defensive situation into a favorable attacking or momentum-gaining situation.
- Use "positive_to_negative" when the active team loses advantage and the situation turns in the opponent's favor.
- Use "stable" when the phase does not meaningfully flip momentum.
- Use "neutral" when the direction is unclear or the moment is dead time.

Return ONLY a JSON object with this exact structure:
{
  "highlights": [
    {"time_sec": 12.0, "score": 2.0, "team_bias": "neutral", "swing": "neutral", "reason": "Players resetting shape before kickoff"},
    {"time_sec": 125.5, "score": 8.5, "team_bias": "active_team_positive", "swing": "negative_to_positive", "reason": "Active team escapes pressure and creates a clear shooting chance from the right side"},
    {"time_sec": 342.0, "score": 9.5, "team_bias": "active_team_positive", "swing": "stable", "reason": "Active team goal scored from close range after a fast combination"}
  ]
}

Scoring guide:
0-2: Dead time, waiting, setup
3-4: Routine possession, low-threat circulation
5-6: Promising build-up, pressing, set-piece setup
7-8: Dangerous chance, shot on target, strong defensive action, near miss
9-10: Goal, penalty, red card, decisive save, major highlight

Return scored moments across the entire clip, not only highlights above a threshold. Be precise with timestamps and specific in your reasons.
"""


class GeminiVideoScorer:
    """Gemini File API で動画を解析しハイライトタイムスタンプを返す"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False,
                                          "forceInput": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gemini-2.5-flash", "multiline": False}),
                # --- セグメント分割 ---
                "segment_duration_min": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
                # --- ローカル圧縮モード（推奨） ---
                "use_local_compress": ("BOOLEAN", {"default": True}),
                "local_compress_resolution": (["360p", "480p", "720p"], {"default": "480p"}),
                # --- YouTube ダウンロードモード ---
                "use_yt_dlp": ("BOOLEAN", {"default": False}),
                "youtube_url": ("STRING", {"default": "", "multiline": False}),
                "yt_dlp_resolution": (["360p", "480p", "720p"], {"default": "480p"}),
                # --- プロンプト（最後に配置）---
                "prompt": ("STRING", {"default": DEFAULT_GEMINI_PROMPT, "multiline": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("timestamps", "scores", "reasoning")
    FUNCTION = "score_video"
    CATEGORY = "SoccerHighlights"

    def score_video(self, video_path, api_key, model, prompt,
                    segment_duration_min,
                    use_local_compress, local_compress_resolution,
                    use_yt_dlp, youtube_url, yt_dlp_resolution):
        import tempfile
        import time
        import shutil
        import subprocess
        import glob as _glob
        import re

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai がインストールされていません。\n"
                "pip install google-genai でインストールしてください。"
            )

        # API キー解決
        _key = api_key.strip()
        if not _key or _key.upper().startswith("YOUR_"):
            _key = ""
        resolved_key = _key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Gemini API キーが設定されていません。\n"
                "ノードの api_key に直接入力するか、環境変数 GEMINI_API_KEY を設定してください。\n"
                "環境変数を設定した場合は ComfyUI の起動前に設定してください。"
            )

        client = genai.Client(api_key=resolved_key)

        # ウィジェット値の型ズレ対策
        use_local_compress = bool(use_local_compress)
        use_yt_dlp = isinstance(use_yt_dlp, bool) and use_yt_dlp

        # ffmpeg パスを探す
        _nvenc_candidates = _glob.glob(
            r"C:\Users\jando\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin\ffmpeg.exe",
            recursive=True
        ) + _glob.glob(r"C:\Program Files\ffmpeg*\bin\ffmpeg.exe")
        ffmpeg_bin = (_nvenc_candidates[0] if _nvenc_candidates else None) or shutil.which("ffmpeg") or "ffmpeg"
        print(f"[GeminiVideoScorer] Using FFmpeg: {ffmpeg_bin}")

        # ------------------------------------------------------------------
        # Helper: 動画長を秒で取得
        # ------------------------------------------------------------------
        def _get_duration_sec(path):
            try:
                r = subprocess.run(
                    [ffmpeg_bin.replace("ffmpeg.exe", "ffprobe.exe") if "ffmpeg.exe" in ffmpeg_bin else "ffprobe",
                     "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", path],
                    capture_output=True, text=True
                )
                return float(r.stdout.strip())
            except Exception:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                return frames / fps

        # ------------------------------------------------------------------
        # Helper: FFmpeg で圧縮（GPU 優先 → CPU フォールバック）
        # ------------------------------------------------------------------
        def _compress_segment(src, dst, height, ss=None, t=None):
            """src → dst に圧縮。ss/t でセグメント切り出し可能。"""
            base_cmd = [ffmpeg_bin, "-y"]
            if ss is not None:
                base_cmd += ["-ss", str(ss)]
            if t is not None:
                base_cmd += ["-t", str(t)]
            # GPU パイプライン
            cmd_gpu = base_cmd + [
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-i", src,
                "-vf", f"scale_cuda=-2:{height}",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
                "-c:a", "aac", "-b:a", "96k",
                dst
            ]
            result = subprocess.run(cmd_gpu, capture_output=True, text=True)
            if result.returncode != 0:
                print("[GeminiVideoScorer] GPU unavailable, falling back to CPU...")
                cmd_cpu = base_cmd + [
                    "-i", src,
                    "-vf", f"scale=-2:{height}",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "96k",
                    dst
                ]
                result = subprocess.run(cmd_cpu, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg compression failed:\n{result.stderr[-500:]}")

        # ------------------------------------------------------------------
        # Helper: Gemini にファイルをアップロードしてクエリを実行
        # ------------------------------------------------------------------
        def _query_segment(upload_path, seg_prompt):
            file_size_mb = os.path.getsize(upload_path) / 1024 / 1024
            print(f"[GeminiVideoScorer] Uploading {file_size_mb:.1f} MB...")
            video_file = client.files.upload(file=upload_path)
            print("[GeminiVideoScorer] Waiting for file processing...")
            while video_file.state.name == "PROCESSING":
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)
            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {video_file.state}")
            print(f"[GeminiVideoScorer] Querying {model}...")
            response = client.models.generate_content(
                model=model,
                contents=[video_file, seg_prompt],
            )
            raw_text = response.text
            print(f"[GeminiVideoScorer] Response (first 500 chars): {raw_text[:500]}")
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
            return raw_text

        # ------------------------------------------------------------------
        # Helper: レスポンスからハイライトをパース
        # ------------------------------------------------------------------
        def _parse_highlights(raw_text, offset_sec=0.0):
            clean = re.sub(r"```[a-z]*", "", raw_text).strip().strip("`")
            highlights = []
            try:
                data = json.loads(clean)
                highlights = data.get("highlights", [])
            except Exception:
                m = re.search(r'\{.*"highlights".*\}', clean, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group(0))
                        highlights = data.get("highlights", [])
                    except Exception:
                        pass
            # タイムスタンプにオフセットを加算
            for h in highlights:
                h["time_sec"] = float(h.get("time_sec", 0)) + offset_sec
            return highlights

        # ==================================================================
        # メイン処理: ソース動画を決定
        # ==================================================================
        source_path = None
        temp_files = []  # 後処理で削除

        try:
            if use_yt_dlp and youtube_url.strip():
                try:
                    import yt_dlp
                except ImportError:
                    raise ImportError("yt-dlp がインストールされていません。pip install yt-dlp")
                height = int(yt_dlp_resolution.replace("p", ""))
                tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tf.close()
                temp_files.append(tf.name)
                print(f"[GeminiVideoScorer] Downloading {yt_dlp_resolution} from YouTube...")
                ydl_opts = {
                    "format": f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
                    "outtmpl": tf.name.replace(".mp4", "") + ".%(ext)s",
                    "merge_output_format": "mp4",
                    "quiet": True,
                    "no_warnings": True,
                    "cookiesfrombrowser": ("chrome",),
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                except Exception as e_cookie:
                    print(f"[GeminiVideoScorer] Cookie download failed ({e_cookie}), retrying...")
                    ydl_opts_nc = {k: v for k, v in ydl_opts.items() if k != "cookiesfrombrowser"}
                    with yt_dlp.YoutubeDL(ydl_opts_nc) as ydl:
                        ydl.download([youtube_url])
                candidates = _glob.glob(tf.name.replace(".mp4", "") + ".*")
                source_path = candidates[0] if candidates else tf.name
                print(f"[GeminiVideoScorer] Downloaded to {source_path}")
            else:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                source_path = video_path

            # ------------------------------------------------------------------
            # セグメント分割処理
            # ------------------------------------------------------------------
            total_sec = _get_duration_sec(source_path)
            segment_sec = segment_duration_min * 60
            height = int(local_compress_resolution.replace("p", "")) if use_local_compress else None

            num_segments = max(1, int(total_sec / segment_sec) + (1 if total_sec % segment_sec > 0 else 0))
            print(f"[GeminiVideoScorer] Video duration: {total_sec:.0f}s → {num_segments} segment(s) of {segment_duration_min}min")
            if num_segments > 10:
                print(f"[GeminiVideoScorer] WARNING: {num_segments} segments will use {num_segments} API requests (RPD=20).")

            all_highlights = []

            for seg_idx in range(num_segments):
                seg_start = seg_idx * segment_sec
                seg_duration = min(segment_sec, total_sec - seg_start)
                seg_end = seg_start + seg_duration
                print(f"[GeminiVideoScorer] Segment {seg_idx+1}/{num_segments}: {seg_start:.0f}s – {seg_end:.0f}s")

                # セグメントファイルを作成
                if use_local_compress:
                    tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    tf.close()
                    seg_path = tf.name
                    temp_files.append(seg_path)
                    _compress_segment(
                        src=source_path, dst=seg_path, height=height,
                        ss=seg_start, t=seg_duration
                    )
                    compressed_mb = os.path.getsize(seg_path) / 1024 / 1024
                    print(f"[GeminiVideoScorer] Segment compressed: {compressed_mb:.1f}MB")
                else:
                    # 圧縮なし: そのまま全体を送る（1セグメントのみ想定）
                    seg_path = source_path

                # セグメント用プロンプト（時刻オフセットを明示）
                seg_prompt = (
                    f"This video clip starts at {seg_start:.0f}s and ends at {seg_end:.0f}s of the full match.\n"
                    f"Report timestamps as seconds from the START OF THIS CLIP (0 to {seg_duration:.0f}).\n\n"
                    + prompt
                )

                try:
                    raw = _query_segment(seg_path, seg_prompt)
                    seg_highlights = _parse_highlights(raw, offset_sec=seg_start)
                    print(f"[GeminiVideoScorer] Segment {seg_idx+1}: {len(seg_highlights)} highlights found.")
                    for h in seg_highlights:
                        print(f"  {h['time_sec']:7.1f}s  score={h.get('score',0):.1f}  {h.get('reason','')}")
                    all_highlights.extend(seg_highlights)
                except Exception as e:
                    print(f"[GeminiVideoScorer] Segment {seg_idx+1} failed: {e}")

            if not all_highlights:
                print("[GeminiVideoScorer] WARNING: No highlights found in any segment.")
                return ([], [], [])

            # タイムスタンプ順にソート
            all_highlights.sort(key=lambda h: h.get("time_sec", 0))
            timestamps = [float(h.get("time_sec", 0)) for h in all_highlights]
            scores     = [float(h.get("score", 5.0)) for h in all_highlights]
            reasons    = [
                json.dumps({
                    "reason": str(h.get("reason", "")),
                    "team_bias": str(h.get("team_bias", "neutral")),
                    "swing": str(h.get("swing", "neutral")),
                }, ensure_ascii=False)
                for h in all_highlights
            ]

            print(f"[GeminiVideoScorer] Total highlights: {len(all_highlights)}")
            return (timestamps, scores, reasons)

        finally:
            # 一時ファイルを削除
            for tf_path in temp_files:
                for f in _glob.glob(tf_path.replace(".mp4", "") + ".*") + [tf_path]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass



NODE_CLASS_MAPPINGS = {
    "VideoFrameSampler": VideoFrameSampler,
    "LLMFrameScorer": LLMFrameScorer,
    "HighlightScriptGenerator": HighlightScriptGenerator,
    "AudioVolumeScorer": AudioVolumeScorer,
    "LocalVLMScorer": LocalVLMScorer,
    "LocalVLMScorerTemporal": LocalVLMScorerTemporal,
    "PlayerInfoExtractor": PlayerInfoExtractor,
    "ScoreCombiner": ScoreCombiner,
    "VideoDirectoryLoader": VideoDirectoryLoader,
    "GeminiVideoScorer": GeminiVideoScorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameSampler": "Video Frame Sampler",
    "LLMFrameScorer": "LLM Frame Scorer (API)",
    "HighlightScriptGenerator": "Highlight Script Generator",
    "AudioVolumeScorer": "Audio Volume Scorer",
    "LocalVLMScorer": "Local VLM Scorer (Moondream2)",
    "LocalVLMScorerTemporal": "Local VLM Scorer Temporal (Moondream2)",
    "PlayerInfoExtractor": "Player Info Extractor",
    "ScoreCombiner": "Score Combiner",
    "VideoDirectoryLoader": "Video Directory Loader",
    "GeminiVideoScorer": "Gemini Video Scorer"
}


