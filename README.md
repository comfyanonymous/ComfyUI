Skip to content
Navigation Menu
patientx
ComfyUI-Zluda

Type / to search
Code
Issues
23
Pull requests
Discussions
Security
Insights
Settings
ComfyUI-Zluda
/
README.md
in
master

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

Soft wrap
Editing README.md file contents
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
<div align="center">

# ComfyUI-ZLUDA

Windows-only version of ComfyUI which uses ZLUDA to get better performance with AMD GPUs.

</div>

## Table of Contents

- [What's New?](#whats-new)
- [Dependencies](#dependencies)
- [Setup (Windows-Only)](#setup-windows-only)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Credits](#credits)

*** FOR THOSE THAT ARE GETTING TROJAN DETECTIONS IN NCCL.DLL IN ZLUDA FOLDER, in dev's words :  "   nccl.dll is a dummy file, it does nothing. when one of its functions is called, it will just return "not supported" status. nccl.dll, cufftw.dll are dummy. they are introduced only for compatibility (to run applications that reject to start without them, but rarely or never use them).zluda.exe hijacks windows api and injects some dlls. its behavior can be considered as malicious by some antiviruses, but it does not hurt the user.
The antiviruses, including windows defender on my computer, didn't detect them as malicious when I made nightly build. but somehow the nightly build is now detected as virus on my end too.   " SO IGNORE THE WARNING AND MAYBE EXCLUDE THE ZLUDA FOLDER FROM DEFENDER.

*** For those who can't seem to follow this readme structure here is install instructions put together neatly : "https://github.com/patientx/ComfyUI-Zluda/issues/188"

## What's New?

* Updated zluda version for old install method and for old amd devices to 3.9.5 (latest) because the newer amd gpu drivers are having problem with old zluda versions. For newer users this wouldn't be a problem but if you already had comfyui-zluda installed (via the older method) and want to re-install the app , please delete everything in : "C:\Users\yourusername\AppData\Local\ZLUDA\ComputeCache" so new zluda can create its database from scratch. 
* Added "cfz-vae-loader" node to cfz folder, it enables changing vae precision on the fly without using "--fp16-vae" etc. on the starting commandline. This is important because while "wan" works faster with fp16 , flux produces black output if fp16 vae is used. So, start comfy normally and add this node to your wan workflow to change it only with that model type.
* Use update.bat if comfyui.bat or comfyui-n.bat can't update. (as when they are the files that needs to be updated, so delete them, run update.bat) When you run your comfyui(-n).bat afterwards, it now copies correct zluda and uses that. 
* Added two workflows for wan text to video and image to video. They are on CFZ folder. From now on I will try to share other workflows that works good on zluda at least on my setup, it could be a starting point for new users if nothing else. Text-to-video has some notes please read them. 
* Updated included zluda version for the new install method to 3.9.5 nightly aka latest version available. You MUST use latest amd gpu drivers with this setup otherwise there would be problems later,  (drivers >= 25.5.1) """ WIPE THE CACHES , READ BELOW """ I recommend using "patchzluda-n.bat" to install it , it uninstalls torch's and reinstalls and patches them with new zluda.ALSO you need to uninstall hip 6.2.4 completely, delete the folder , and reinstall it and again download and unzip the NEW Hip addon (this is not the same as the one I shared before , this is new for 3.9.5) for this 3.9.5 version (hopefully this won't happen that much) new hip addon for zluda 3.9.5 : (https://drive.google.com/file/d/1Gvg3hxNEj2Vsd2nQgwadrUEY6dYXy0H9/view?usp=sharing)
* WIPING CACHES FOR A CLEAN REINSTALL :: There are three caches to delete if you want to start anew -it is recommended if you want a painless zluda experience : 1-) "C:\Users\yourusername\AppData\Local\ZLUDA\ComputeCache" 2-) "C:\ Users \ yourusername \ .miopen" 3-) "C:\ Users \ yourusername \ .triton" , delete everything in these three directories, zluda and miopen and triton will do everything from the start again but it would be less painful for the future.
* Added "CFZ Cudnn Toggle" node, it is for some of the audio models, not working with cudnn -which is enabled by default on new install method- to use it just connect it before ksampler -latent_image input or any latent input- disable cudnn, THEN after the vae decoding -which most of these problems occur- to re-enable cudnn , add it after vae-decoding, select audio_output and connect it save audio node of course enable cudnn now.This way within that workflow you are disabling cudnn when working with models that are not compatible with it, so instead of completely disabling it in comfy we can do it locally like this.
*  "CFZ Checkpoint Loader"was broken, it might corrupt the models if you load with it and quit halfway, I completely redone it and it now works outside the checkpoint loading so doesn't touch the file and when it does quantize the model, it makes a copy and quantizes it. Please delete the "cfz_checkpoint_loader.py" and use the newly added "cfz_patcher.py" it got three seperate nodes and much safer and better.
* BOTH of these nodes are inside "cfz" folder, to use them copy them into custom_nodes, they would appear next time you open comfy, to find them searh for "cfz" you will see both nodes.
* flash-attention download error fixed, also added sage-attention fix, especially for vae out of memory errors that occurs a lot with sage-attention enabled. NOTE : this doesn't require any special packages or hardware as far as I know so it could work with everything.
* `install-n.bat` now not only installs everything needed for MIOPEN and Flash-Attention use, it also automates installing triton (only supported for python 3.10.x and 3.11.x) and flash-attention. So if you especially have 6000+ gpu , have HIP 6.2.4 and libraries if necessary, try it. But beware, there are lots of errors yet to be unsolved. So it is still not the default installed version.
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
No file chosen
Attach files by dragging & dropping, selecting or pasting them.
Editing ComfyUI-Zluda/README.md at master · patientx/ComfyUI-Zluda
