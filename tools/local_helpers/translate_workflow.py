
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-english.json"

replacements = {
    "节点核心作用：超分修复": "Node Core Function: Super Resolution",
    "节点核心作用：分块处理": "Node Core Function: Tiled Processing",
    "节点核心作用：长视频注意力机制优化": "Node Core Function: Long Video Attention Optimization",
    "模型的话，关闭增益三件套。": "For models, disable the gain trio.",
    "其实差不多，不过加载模型比较慢。": "Similar results, but model loading is slower.",
    "宽": "width",
    "高": "height", 
    "输入": "Input",
    "输出": "Output",
    "图像": "IMAGE",
    "字符串": "STRING",
    "遮罩": "MASK",
    "文件名": "Filename",
    "帧数": "frames",
    "显示帮助": "Show Help",
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，": "Vibrant tones, overexposed, static, blurred details, subtitles, style, artwork, painting, screen, still, grey overall, worst quality, low quality,",
    "女孩脱下身上的外衣，当然，图像的尺寸也要相应的放大。运行时长和": "Girl taking off outer coat. Of course, image size must be scaled up. Runtime and...",
    "杂乱的背景，三条腿，背景人很多，倒着走": "Cluttered background, three legs, crowded background, walking backwards",
    "按照我的撰写公式来描述图像内容，公式是：[主体描述]": "Describe image content using my formula: [Subject Description]",
    "压缩残留": "Compression Artifacts",
    "Anything Everywhere": "Anything Everywhere", # Keep English
    "Note": "Note",
    "WanVideoDecode": "WanVideoDecode"
}

def translate_obj(obj):
    if isinstance(obj, str):
        # Exact match
        if obj in replacements:
            return replacements[obj]
        # Containment match for prompts?
        # Let's simple check if any key is in obj
        for k, v in replacements.items():
            if k in obj and len(k) > 1: # Avoid single char replacement in random places unless absolutely sure
                 obj = obj.replace(k, v)
        return obj
    elif isinstance(obj, list):
        return [translate_obj(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: translate_obj(v) for k, v in obj.items()}
    else:
        return obj

def translate_workflow():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # Walk and translate
    new_workflow = translate_obj(workflow)
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_workflow, f, indent=4, ensure_ascii=False)
        
    print("Translation Complete.")

if __name__ == "__main__":
    translate_workflow()
