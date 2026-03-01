
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "custom_nodes", "ComfyUI_SoccerHighlights"))

try:
    from custom_nodes.ComfyUI_SoccerHighlights.nodes import AudioVolumeScorer, LocalVLMScorer, ScoreCombiner
    print(" Import Successful: New nodes imported correctly.")
except Exception as e:
    print(f" Import Failed: {e}")
