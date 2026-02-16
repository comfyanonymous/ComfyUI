"""
Example usage of the Talking Characters system
Demonstrates different scenarios and configurations
"""

import sys
from pathlib import Path

# Add parent directory to imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_generator import TalkingCharacterGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_jessica():
    """Generate single Jessica video"""
    logger.info("\n📹 Example 1: Single Jessica Video")

    generator = TalkingCharacterGenerator(
        character="jessica",
        dialogue_text="Cześć kochanie, czekam na ciebie. Chcę się z tobą bawić.",
        tts_engine="piper",
        output_name="jessica_seduction_01"
    )

    generator.generate()


def example_single_gigi():
    """Generate single Gigi video"""
    logger.info("\n📹 Example 2: Single Gigi Video")

    generator = TalkingCharacterGenerator(
        character="gigi",
        dialogue_text="Cześć, jestem Gigi. Jestem gotowa dla ciebie.",
        tts_engine="piper",
        output_name="gigi_introduction_01"
    )

    generator.generate()


def example_jessica_dialogue_sequence():
    """Generate sequence of Jessica videos telling a story"""
    logger.info("\n📹 Example 3: Jessica Dialogue Sequence")

    dialogue_sequence = [
        {
            "text": "Cześć kochanie, czekam na ciebie.",
            "output": "jessica_scene_01_hello"
        },
        {
            "text": "Chcę się z tobą bawić. Jestem dla ciebie gotowa.",
            "output": "jessica_scene_02_ready"
        },
        {
            "text": "Mmmm, to jest takie dobre. Nie przestawaj.",
            "output": "jessica_scene_03_pleasure"
        },
        {
            "text": "Jeszcze, jeszcze bardziej! Oooh!",
            "output": "jessica_scene_04_intense"
        }
    ]

    for i, scene in enumerate(dialogue_sequence, 1):
        logger.info(f"\nGenerating scene {i}/{len(dialogue_sequence)}")

        generator = TalkingCharacterGenerator(
            character="jessica",
            dialogue_text=scene["text"],
            tts_engine="piper",
            output_name=scene["output"]
        )

        if generator.generate():
            logger.info(f"✅ Scene {i} completed")
        else:
            logger.error(f"❌ Scene {i} failed")


def example_gigi_dialogue_sequence():
    """Generate sequence of Gigi videos"""
    logger.info("\n📹 Example 4: Gigi Dialogue Sequence")

    dialogue_sequence = [
        {
            "text": "Cześć, jestem Gigi. Jestem gotowa dla ciebie.",
            "output": "gigi_scene_01_intro"
        },
        {
            "text": "Dotknij mnie. Chcę czuć twoją dotyk.",
            "output": "gigi_scene_02_touch"
        },
        {
            "text": "Tak, tak! Dokładnie tam! Mmmm...",
            "output": "gigi_scene_03_pleasure"
        }
    ]

    for i, scene in enumerate(dialogue_sequence, 1):
        logger.info(f"\nGenerating scene {i}/{len(dialogue_sequence)}")

        generator = TalkingCharacterGenerator(
            character="gigi",
            dialogue_text=scene["text"],
            tts_engine="piper",
            output_name=scene["output"]
        )

        if generator.generate():
            logger.info(f"✅ Scene {i} completed")
        else:
            logger.error(f"❌ Scene {i} failed")


def example_different_tts_engines():
    """Compare different TTS engines with Jessica"""
    logger.info("\n📹 Example 5: Different TTS Engines")

    tts_engines = ["piper", "bark", "gtts"]
    text = "Cześć, to jest test różnych TTS silników."

    for engine in tts_engines:
        logger.info(f"\n🔊 Testing {engine} engine...")

        try:
            generator = TalkingCharacterGenerator(
                character="jessica",
                dialogue_text=text,
                tts_engine=engine,
                output_name=f"jessica_tts_{engine}_test"
            )

            if generator.generate():
                logger.info(f"✅ {engine} successful")
            else:
                logger.warning(f"⚠️  {engine} failed")
        except Exception as e:
            logger.error(f"❌ {engine} error: {e}")


def example_batch_production():
    """Production-ready batch generation"""
    logger.info("\n📹 Example 6: Batch Production")

    # Define production content
    production_scenes = [
        {
            "character": "jessica",
            "dialogue": [
                "Cześć kochanie, czekam na ciebie.",
                "Chcę się z tobą bawić.",
                "Tak, jak to lubię. Jeszcze!",
                "Ohhh! To wspaniałe!"
            ]
        },
        {
            "character": "gigi",
            "dialogue": [
                "Cześć, jestem Gigi.",
                "Jestem dla ciebie gotowa.",
                "Mmmm, tak dobrze!",
                "Nie przestawaj, proszę!"
            ]
        }
    ]

    total_videos = sum(len(s["dialogue"]) for s in production_scenes)
    completed = 0

    for scene_group in production_scenes:
        character = scene_group["character"]

        for line_num, dialogue in enumerate(scene_group["dialogue"], 1):
            completed += 1
            logger.info(f"\n[{completed}/{total_videos}] {character.upper()} - Line {line_num}")

            generator = TalkingCharacterGenerator(
                character=character,
                dialogue_text=dialogue,
                tts_engine="piper",
                output_name=f"{character}_production_line_{line_num:02d}"
            )

            if not generator.generate():
                logger.warning(f"⚠️  Line {line_num} failed, continuing...")

    logger.info(f"\n✅ Batch complete: {completed}/{total_videos} videos generated")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TALKING CHARACTERS - EXAMPLE USAGE")
    logger.info("=" * 70)

    # Choose which example to run:
    # example_single_jessica()          # Simple single video
    # example_single_gigi()             # Different character
    # example_jessica_dialogue_sequence()  # Story sequence
    # example_gigi_dialogue_sequence()    # Story sequence
    # example_different_tts_engines()     # Compare TTS quality
    example_batch_production()            # Production workflow

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXAMPLES COMPLETED")
    logger.info("=" * 70)
