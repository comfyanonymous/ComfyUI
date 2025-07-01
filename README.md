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

## What's New?

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
* If you want to enable MIOPEN , Triton and Flash-Attention use the install-n.bat . This will install torch 2.7 , with latest nightly zluda and patch the correct files into comfyui-zluda. This features do not work very well and you are on your own if you want to try these. (---- You have to install HIP 6.2.4 - Download and extract HIP Addon inside the folder (information down below on installation section). ---)
* Florance2 is now fixed , (probably some other nodes too) you need to disable "do_sample" meaning change it from True to False, now it would work without needing to edit it's node.
* Added onnxruntime fix so that it now uses cpu-only regardless of node. So now  nodes like pulid, reactor, infiniteyou etc works without problems and can now use codeformer too.
* Added a way to use any zluda you want (to use with HIP versions you want to use such as 6.1 - 6.2) After installing, close the app, run `patchzluda2.bat`. It will ask for url of the zluda build you want to use. You can choose from them here,
 [lshyqqtiger's ZLUDA Fork](https://github.com/lshqqytiger/ZLUDA/releases) then you can use `patchzluda2.bat`, run it paste the link via right click (A correct link would be like this, `https://github.com/lshqqytiger/ZLUDA/releases/download/rel.d60bddbc870827566b3d2d417e00e1d2d8acc026/ZLUDA-windows-rocm6-amd64.zip`)
 After pasting press enter and it would patch that zluda into comfy for you.
* Added a "small flux guide." This aims to use low vram and provides the very basic necessary files needed to get flux generation running. [HERE](fluxguide.md)

> [!IMPORTANT]
> 
> 📢 ***REGARDING KEEPING THE APP UP TO DATE***
>
> Avoid using the update function from the manager, instead use `git pull`, which we
> are doing on every start if `start.bat` is used. (App Already Does It Every Time You Open It, If You Are Using
> `comfyui.bat`, So This Way It Is Always Up To Date With Whatever Is On My GitHub Page)
>
> Only use comfy manager to update the extensions
> (Manager -> Custom Nodes Manager -> Set Filter To Installed -> Click Check Update On The Bottom Of The Window)
> otherwise it breaks the basic installation, and in that case run `install.bat` once again.
> 
> 📢 ***REGARDING RX 480-580 AND SIMILAR GPUS***
>
> After a while we need to keep updating certain packages to keep up with the original app and certain requirements of
> some models etc. So, torch is changed over time , but this gave gpu's prior to rdna some negative performance. 
> So for these gpu's please use `fixforrx580.bat` if you are having memory problems too much slowdown etc. This
> is not mandatory and it won't just make your gpu faster then before but it would be less problematic then using latest torch
> that we use with other gpu's.
> 
> 📢 ***RANDOM TIPS & TRICKS REGARDING USING AMD GPU'S***
> 
> * The generation speed is slower than nvidia counterparts we all know and accept it, but most of the time the out of memory situation 
> at the end with VAE decoding is worse. To overcome this use "--cpu-vae". Add it to commandline_args on comfyui.bat.
> You can now decode using your system memory (the more the memory the better) and your cpu power. This might be slower but at least it works.

## Dependencies

If coming from the very start, you need :

1. **Git**: Download from https://git-scm.com/download/win.
   During installation don't forget to check the box for "Use Git from the Windows Command line and also from
   3rd-party-software" to add Git to your system's PATH.
2. **Python** ([3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)) version "3.11" . (I no longer am suggesting 3.10 as default since a lot of newer stuff is only coming out in 3.11 and up. 3.10 still works but you may encounter a package that specifically needs 3.11 , so for new installs I suggest using 3.11)  Install the latest release from python.org. **Don't Use
   Windows Store Version**. If you have that installed, uninstall and please install from python.org. During
   installation remember to check the box for "Add Python to PATH when you are at the "Customize Python" screen.
3. **Visual C++ Runtime**: Download [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) and install it.
4. Install **HIP SDK 5.7.1** from [HERE](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) the correct version, "Windows 10 & 11 5.7.1 HIP SDK"

   *** (*** this app installs zluda for 5.7.1 by default, if you want to use 6.1 or 6.2 you have to get the latest zluda link from
    [lshyqqtiger's ZLUDA Fork](https://github.com/lshqqytiger/ZLUDA/releases) then you can use `patchzluda2.bat`, run it paste the link via right click (a correct link would be like this, 
    `https://github.com/lshqqytiger/ZLUDA/releases/download/rel.d60bddbc870827566b3d2d417e00e1d2d8acc026/ZLUDA-windows-rocm6-amd64.zip` press enter and it would patch that zluda into comfy for you. Of course this also would mean you have to change the variables below 
     from "5.7" to "6.x" where needed) ***

5. Add the system variable HIP_PATH, value: `C:\Program Files\AMD\ROCm\5.7\` (This is the default folder with default 5.7.1 installed, if you
   have installed it on another drive, change if necessary, if you have installed 6.2.4, or other newer versions the numbers should reflect that.)
    1. Check the variables on the lower part (System Variables), there should be a variable called: HIP_PATH.
    2. Also check the variables on the lower part (System Variables), there should be a variable called: "Path".
       Double-click it and click "New" add this: `C:\Program Files\AMD\ROCm\5.7\bin` (This is the default folder with default 5.7.1 installed, if you
   have installed it on another drive, change if necessary, if you have installed 6.2.4, or other newer versions the numbers should reflect that.)

    ((( these should be  HIP_PATH, value: `C:\Program Files\AMD\ROCm\6.2\` and Path variable  `C:\Program Files\AMD\ROCm\6.2\bin` )))

   5.1  *** YOU MUST DO THIS ADDITIONAL STEPS : if you want to try miopen-triton with high end gpu : aka plan to use install-n.bat ***
   
    * You MUST use latest amd gpu drivers with this setup otherwise there would be problems later. (drivers >= 25.5.1)
    * You must install "https://visualstudio.microsoft.com/". Only "Build Tools" (aka Desktop Development with C++)  need to be selected and installed, we don't need the others.
    * Install **HIP SDK 6.2.4** from [HERE](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) the correct version, "Windows 10 & 11 6.2.4 HIP SDK".
    * Then download hip sdk addon from this url and extract that into `C:\Program Files\AMD\ROCm\6.2` . (updated for zluda 3.9.5)
    *  (new hip addon for zluda 3.9.5 : (https://drive.google.com/file/d/1Gvg3hxNEj2Vsd2nQgwadrUEY6dYXy0H9/view?usp=sharing))
    *  (Alternative source for hip addon for zluda 3.9.5 : (https://www.mediafire.com/file/ooawc9s34sazerr/HIP-SDK-extension(zluda395).zip/file))
     
6. If you have an AMD GPU below 6800 (6700,6600 etc.), download the recommended library files for your gpu (NOTE : Besides those older gpu's the newest gfx1200 and gfx1201 aka 9070 - 9070xt STILL requires libraries because amd didn't add support for them in the 6.2.4 . In the future they %100 would but using 6.2.4 those gpu's also need libraries - from likelovewant libs)

- from [Brknsoul Repository](https://github.com/brknsoul/ROCmLibs) (for hip 5.7.1)

- from [likelovewant Repository](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (for hip 6.2.4)

    1. Go to folder "C:\Program Files\AMD\ROCm\5.7\bin\rocblas", (or 6.2 if you have installed that) there would be a "library" folder, backup the files
       inside to somewhere else.
    2. Open your downloaded optimized library archive and put them inside the library folder (overwriting if
       necessary): "C:\\Program Files\\AMD\\ROCm\\5.7\\bin\\rocblas\\library" (or 6.2 if you have installed that)
       * There could be a rocblas.dll file in the archive as well, if it is present then copy it inside "C:\Program Files\AMD\ROCm\5.7\bin\rocblas" (or 6.2 if you have installed that)
7. Reboot your system.

## Setup (Windows-Only)

Open a cmd prompt. (Powershell doesn't work, you have to use command prompt.)

```bash
git clone https://github.com/patientx/ComfyUI-Zluda
```

```bash
cd ComfyUI-Zluda
```

```bash
install.bat
```
((( use `install-n.bat` if you want to install for miopen-triton combo for high end gpu's )))

to start for later use (or create a shortcut to) :

```bash
comfyui.bat
```
((( use `comfyui-n.bat` if you want to use miopen-triton combo for high end gpu's, that basically changes the attention to pytorch attention which works with flash attention )))

    --- To use sage-attention , you have to change the "--use-pytorch-cross-attention" to "--use-sage-attention". My advice is make a seperate batch file for all different attention types instead of changing them.
    --- You can use "--use-pytorch-cross-attention", "--use-quad-cross-attention" , "--use-flash-attention" and "--use-sage-attention" . Also you can activate flash or sage with the help some nodes from pytorch attention.

also for later when you need to repatch zluda (maybe a torch update etc.) you can use:

```bash
patchzluda.bat
```

((( `patchzluda-n.bat` for miopen-triton setup)))

- The first generation would take around 10-15 minutes, there won't be any progress or indicator on the webui or cmd
  window, just wait. Zluda creates a database for use with generation with your gpu.

> [!NOTE]
> **This might happen with torch changes , zluda version changes and / or gpu driver changes.**

## Troubleshooting

- Problems with triton , try this : Remove visual studio 2022 (if you have already installed it and getting errors) and install "https://aka.ms/vs/17/release/vs_BuildTools.exe" , and then use  "Developer Command Prompt" to run comfyui. This option shouldn't be needed for many but nevertheless try.
- `CUDA device detected: None` , if seeing this error with the new install-n , make sure you are NOT using the amd driver 25.5.1 . Use a previous driver, it has problems with zluda.
- `RuntimeError: CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize FailedmiopenStatusInternalError cudnn_status: miopenStatusUnknownError` , if this is encountered at the end, while vae-decoding, use tiled-vae decoding either from official comfy nodes or from Tiled-Diffussion (my preference). Also vae-decoding is overall better with tiled-vae decoding. 
- If you installed miopen-triton setup with install.n.bat , getting ":: Triton core imported successfully , :: Running Triton kernel test... , :: Triton test failed , :: Triton available but failed verification" Do this to fix it :
   * Copy "libs" folder from python install directory under(C:\Users\username\AppData\Local\Programs\Python\Python310) Or 311 , whatever python you are using, to under "comfyui-zluda\venv" so ,
   * For example : "C:\Users\username\AppData\Local\Programs\Python\Python310\libs" to "d:\Comfyui-Zluda\venv\* Beware, there are two similar folders under venv , "Lib" and "Library" , so this would be a similarly named folder , don't mistake them.
  Then try running comfy again.
- DO NOT use non-english characters as folder names to put comfyui-zluda under.
- Wipe your pip cache "C:\Users\USERNAME\AppData\Local\pip\cache" You can also do this when venv is active with :
  `pip cache purge`
- `xformers` isn't usable with zluda so any nodes / packages that require it doesn't work. `Flash attention`
  doesn't work.
- Have the latest drivers installed for your amd gpu. **Also, Remove Any Nvidia Drivers** you might have from previous
  nvidia gpu's.
- If you see zluda errors make sure these three files are inside "ComfyUI-Zluda\venv\Lib\site-packages\torch\lib\"
  `cublas64_11.dll (231kb)` `cusparse64_11.dll (199kb)` `nvrtc64_112_0.dll (129kb)` If they are there but much bigger in size please run : `patchzluda.bat` (this is for zluda 3.8.4, other versions might be different sizes)
- If for some reason you can't solve with these and want to start from zero, delete "venv" folder and re-run
  `install.bat`
- If you can't git pull to the latest version, run these commands, `git fetch --all` and then
  `git reset --hard origin/master` now you can git pull
- Problems with `caffe2_nvrtc.dll`: if you are sure you properly installed hip and can see it on path, please DON'T use
  python from windows store, use the link provided or 3.11 from the official website. After uninstalling python from
  windows store and installing the one from the website, be sure the delete venv folder, and run install.bat once again.
- `rocBLAS`-error: If you have an integrated GPU by AMD (e.g. AMD Radeon(TM) Graphics) you need to add `HIP_VISIBLE_DEVICES=1` to your
  environment variables. Other possible variables to use : `ROCR_VISIBLE_DEVICES=1` `HCC_AMDGPU_TARGET=1` . This basically tells it to use 1st gpu -this number could be different if you have multiple gpu's-
  Otherwise it will default to using your iGPU, which will most likely not work. This behavior is caused by a bug in the ROCm-driver.
- Lots of other problems were encountered and solved by users so check the issues if you can't find your problem here.  

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Zluda Wiki from SdNext](https://github.com/vladmandic/automatic/wiki/ZLUDA)
- [Brknsoul for Rocm Libraries](https://github.com/brknsoul/ROCmLibs)
- [Lshqqytiger](https://github.com/lshqqytiger/ZLUDA)
- [LeagueRaINi](https://github.com/LeagueRaINi/ComfyUI)
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
