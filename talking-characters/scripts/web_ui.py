"""
Web UI for Talking Characters
Flask-based interface for generating talking character videos
"""

import os
import sys
from pathlib import Path
from threading import Thread
import json
import logging

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_generator import TalkingCharacterGenerator
from config import (
    CHARACTERS, DIALOGUE_SCENES, AUDIO_DIR, VIDEO_DIR,
    TTS_ENGINE, PROJECT_ROOT, DEBUG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, template_folder=PROJECT_ROOT / "templates")
CORS(app)

# Generation status
generation_status = {
    "active": False,
    "character": None,
    "progress": 0,
    "message": "",
    "error": None,
    "output_file": None
}


# Routes
@app.route("/")
def index():
    """Main page"""
    return render_template("index.html", characters=list(CHARACTERS.keys()))


@app.route("/api/characters", methods=["GET"])
def get_characters():
    """Get available characters"""
    chars = []
    for char_key, char_config in CHARACTERS.items():
        chars.append({
            "id": char_key,
            "name": char_config["name"],
            "workflow": char_config["workflow"]
        })
    return jsonify(chars)


@app.route("/api/dialogue-scenes", methods=["GET"])
def get_dialogue_scenes():
    """Get available dialogue scenes"""
    return jsonify(DIALOGUE_SCENES)


@app.route("/api/generate", methods=["POST"])
def generate_video():
    """Generate video"""
    global generation_status

    if generation_status["active"]:
        return jsonify({"error": "Generation already in progress"}), 409

    data = request.json
    character = data.get("character")
    text = data.get("text")
    tts_engine = data.get("tts_engine", TTS_ENGINE)
    output_name = data.get("output_name", f"{character}_video")

    if not character or not text:
        return jsonify({"error": "Missing character or text"}), 400

    if character not in CHARACTERS:
        return jsonify({"error": f"Unknown character: {character}"}), 400

    # Start generation in background
    Thread(
        target=_generate_background,
        args=(character, text, tts_engine, output_name),
        daemon=True
    ).start()

    return jsonify({"status": "generating", "message": "Video generation started"})


def _generate_background(character: str, text: str, tts_engine: str, output_name: str):
    """Background video generation"""
    global generation_status

    try:
        generation_status["active"] = True
        generation_status["character"] = character
        generation_status["progress"] = 0
        generation_status["error"] = None
        generation_status["output_file"] = None

        logger.info(f"🎬 Starting generation: {character}")

        generator = TalkingCharacterGenerator(
            character=character,
            dialogue_text=text,
            tts_engine=tts_engine,
            output_name=output_name
        )

        generation_status["message"] = "Generating image with Flux..."
        generation_status["progress"] = 20

        # Generate
        if generator.generate():
            generation_status["message"] = "Generation complete!"
            generation_status["progress"] = 100
            generation_status["output_file"] = f"{output_name}_final.mp4"
            logger.info(f"✅ Generation complete: {output_name}")
        else:
            generation_status["error"] = "Generation failed"
            logger.error(f"❌ Generation failed: {output_name}")

    except Exception as e:
        generation_status["error"] = str(e)
        logger.error(f"❌ Error: {e}")

    finally:
        generation_status["active"] = False


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get generation status"""
    return jsonify(generation_status)


@app.route("/api/videos", methods=["GET"])
def list_videos():
    """List generated videos"""
    videos = []

    if VIDEO_DIR.exists():
        for video_file in VIDEO_DIR.glob("*_final.mp4"):
            videos.append({
                "name": video_file.stem.replace("_final", ""),
                "filename": video_file.name,
                "size": video_file.stat().st_size,
                "date": video_file.stat().st_mtime
            })

    return jsonify(sorted(videos, key=lambda x: x["date"], reverse=True))


@app.route("/api/download/<filename>", methods=["GET"])
def download_video(filename: str):
    """Download video file"""
    filepath = VIDEO_DIR / filename

    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True)


@app.route("/api/delete/<filename>", methods=["DELETE"])
def delete_video(filename: str):
    """Delete video file"""
    filepath = VIDEO_DIR / filename

    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        filepath.unlink()
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check"""
    return jsonify({
        "status": "ok",
        "version": "0.1.0",
        "project_root": str(PROJECT_ROOT)
    })


if __name__ == "__main__":
    logger.info("🚀 Starting Talking Characters Web UI...")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"🎬 Video output: {VIDEO_DIR}")
    logger.info(f"🔊 Audio output: {AUDIO_DIR}")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=DEBUG,
        threaded=True
    )
