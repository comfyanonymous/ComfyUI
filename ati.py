import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="A little boy and a dog are hiking in the forest. The puppy runs around the boy cheerfully.\n",
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，",
            clip=get_value_at_index(cliploader_38, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="kj-Wan2_1_VAE_fp32.safetensors")

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_140 = unetloader.load_unet(
            unet_name="Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn_fast",
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_154 = loraloadermodelonly.load_lora_model_only(
            lora_name="Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
            strength_model=0.30000000000000004,
            model=get_value_at_index(unetloader_140, 0),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_240 = loadimage.load_image(image="example.png")

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_244 = clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        primitivestringmultiline = NODE_CLASS_MAPPINGS["PrimitiveStringMultiline"]()
        primitivestringmultiline_247 = primitivestringmultiline.execute(
            value="[[{'x':637.5706176757812,'y':357.4817810058594},{'x':637.1891479492188,'y':357.1156005859375},{'x':636.8077392578125,'y':356.7493896484375},{'x':636.42626953125,'y':356.3832092285156},{'x':636.0448608398438,'y':356.01702880859375},{'x':635.6633911132812,'y':355.6508483886719},{'x':635.2819213867188,'y':355.2846374511719},{'x':634.9005126953125,'y':354.91845703125},{'x':634.51904296875,'y':354.5522766113281},{'x':634.1375732421875,'y':354.18609619140625},{'x':633.7561645507812,'y':353.81988525390625},{'x':633.3746948242188,'y':353.4537048339844},{'x':632.9932861328125,'y':353.0875244140625},{'x':632.61181640625,'y':352.7213134765625},{'x':632.2305297851562,'y':352.35498046875},{'x':631.849365234375,'y':351.9884948730469},{'x':631.4681396484375,'y':351.6220397949219},{'x':631.0878295898438,'y':351.25469970703125},{'x':630.7080688476562,'y':350.8867492675781},{'x':630.328857421875,'y':350.51824951171875},{'x':629.9509887695312,'y':350.1484069824219},{'x':629.5737915039062,'y':349.7778015136719},{'x':629.197509765625,'y':349.4063415527344},{'x':628.8228759765625,'y':349.0331726074219},{'x':628.4490356445312,'y':348.65924072265625},{'x':628.0768432617188,'y':348.28363037109375},{'x':627.7057495117188,'y':347.9069519042969},{'x':627.3364868164062,'y':347.5285339355469},{'x':626.9689331054688,'y':347.14837646484375},{'x':626.603271484375,'y':346.7663879394531},{'x':626.2395629882812,'y':346.38262939453125},{'x':625.8778076171875,'y':345.9969482421875},{'x':625.518310546875,'y':345.6092224121094},{'x':625.1611328125,'y':345.21929931640625},{'x':624.8068237304688,'y':344.8268127441406},{'x':624.4552612304688,'y':344.4318542480469},{'x':624.1066284179688,'y':344.0342712402344},{'x':623.76123046875,'y':343.6339111328125},{'x':623.4193725585938,'y':343.2305603027344},{'x':623.0819091796875,'y':342.823486328125},{'x':622.7485961914062,'y':342.41302490234375},{'x':622.4198608398438,'y':341.9988708496094},{'x':622.0963745117188,'y':341.5805969238281},{'x':621.7791137695312,'y':341.1575622558594},{'x':621.4686889648438,'y':340.72955322265625},{'x':621.165771484375,'y':340.296142578125},{'x':620.8717041015625,'y':339.8567199707031},{'x':620.5880126953125,'y':339.4104919433594},{'x':620.31640625,'y':338.95684814453125},{'x':620.0592041015625,'y':338.494873046875},{'x':619.8197631835938,'y':338.02349853515625},{'x':619.60205078125,'y':337.5417175292969},{'x':619.4119873046875,'y':337.04840087890625},{'x':619.2584228515625,'y':336.5425720214844},{'x':619.15380859375,'y':336.0245361328125},{'x':619.1163940429688,'y':335.4975891113281},{'x':619.1705932617188,'y':334.9725341796875},{'x':619.3410034179688,'y':334.4734802246094},{'x':619.6347045898438,'y':334.03564453125},{'x':620.0236206054688,'y':333.6789245605469},{'x':620.4705810546875,'y':333.3973388671875},{'x':620.9505004882812,'y':333.17608642578125},{'x':621.4495239257812,'y':333.001708984375},{'x':621.9599609375,'y':332.864013671875},{'x':622.4772338867188,'y':332.7549133300781},{'x':622.9990234375,'y':332.6694030761719},{'x':623.5235595703125,'y':332.6031188964844},{'x':624.0499267578125,'y':332.55303955078125},{'x':624.5774536132812,'y':332.5166931152344},{'x':625.1056518554688,'y':332.4925231933594},{'x':625.6342163085938,'y':332.4786376953125},{'x':626.1629638671875,'y':332.4738464355469},{'x':626.6917114257812,'y':332.4774475097656},{'x':627.2203979492188,'y':332.4881896972656},{'x':627.7488403320312,'y':332.50579833984375},{'x':628.277099609375,'y':332.5293884277344},{'x':628.8050537109375,'y':332.55828857421875},{'x':629.3327026367188,'y':332.59210205078125},{'x':629.860107421875,'y':332.6304931640625},{'x':630.3871459960938,'y':332.6730651855469},{'x':630.913818359375,'y':332.71966552734375},{'x':631.440185546875,'y':332.7701416015625},{'x':631.9661865234375,'y':332.8240051269531},{'x':632.4918823242188,'y':332.8810729980469},{'x':633.0172119140625,'y':332.9410705566406},{'x':633.542236328125,'y':333.00384521484375},{'x':634.0669555664062,'y':333.0692443847656},{'x':634.5913696289062,'y':333.1371154785156},{'x':635.1154174804688,'y':333.20733642578125},{'x':635.6392211914062,'y':333.27978515625},{'x':636.1627197265625,'y':333.3542785644531},{'x':636.6859130859375,'y':333.43084716796875},{'x':637.208740234375,'y':333.5098571777344},{'x':637.7314453125,'y':333.589599609375},{'x':638.2538452148438,'y':333.6716003417969},{'x':638.7760009765625,'y':333.7548522949219},{'x':639.2979736328125,'y':333.8394775390625},{'x':639.8196411132812,'y':333.92578125},{'x':640.3411865234375,'y':334.0127868652344},{'x':640.8624267578125,'y':334.1016845703125},{'x':641.383544921875,'y':334.1914367675781},{'x':641.9043579101562,'y':334.2825622558594},{'x':642.4252319335938,'y':334.37371826171875},{'x':642.9459838867188,'y':334.46527099609375},{'x':643.466552734375,'y':334.5581970214844},{'x':643.987060546875,'y':334.651123046875},{'x':644.507568359375,'y':334.7441711425781},{'x':645.028076171875,'y':334.8374328613281},{'x':645.548583984375,'y':334.9306640625},{'x':646.0690307617188,'y':335.02392578125},{'x':646.5895385742188,'y':335.1171875},{'x':647.1099853515625,'y':335.21044921875},{'x':647.6304931640625,'y':335.3037109375},{'x':648.1509399414062,'y':335.3969421386719},{'x':648.6714477539062,'y':335.4902038574219},{'x':649.19189453125,'y':335.5834655761719},{'x':649.71240234375,'y':335.6767272949219},{'x':650.2328491210938,'y':335.7699890136719},{'x':650.7533569335938,'y':335.86322021484375},{'x':651.2738037109375,'y':335.95648193359375},{'x':651.7943115234375,'y':336.04974365234375}]]"
        )

        wantracktovideo = NODE_CLASS_MAPPINGS["WanTrackToVideo"]()
        wantracktovideo_248 = wantracktovideo.encode(
            tracks=get_value_at_index(primitivestringmultiline_247, 0),
            width=832,
            height=480,
            length=81,
            batch_size=1,
            temperature=220,
            topk=10,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            start_image=get_value_at_index(loadimage_240, 0),
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()

        for q in range(1):
            modelsamplingsd3_48 = modelsamplingsd3.patch(
                shift=5, model=get_value_at_index(loraloadermodelonly_154, 0)
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1,
                sampler_name="uni_pc",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(modelsamplingsd3_48, 0),
                positive=get_value_at_index(wantracktovideo_248, 0),
                negative=get_value_at_index(wantracktovideo_248, 1),
                latent_image=get_value_at_index(wantracktovideo_248, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            createvideo_68 = createvideo.create_video(
                fps=16, images=get_value_at_index(vaedecode_8, 0)
            )

            savevideo_69 = savevideo.save_video(
                filename_prefix="video/ComfyUI",
                format="auto",
                codec="auto",
                video=get_value_at_index(createvideo_68, 0),
            )


if __name__ == "__main__":
    main()
