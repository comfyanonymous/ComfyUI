
<div align="center">

# ComfyUI-ZLUDA

Windows-only version of ComfyUI which uses ZLUDA to get better performance with AMD GPUs.

</div>

*** Comfyui added partial AMD support with the official torch's , at the moment the supported GPU's on windows start from 7000 series and up. For more information about that go to the official comfyui github page. (https://github.com/comfyanonymous/ComfyUI). This fork is still the best solution for older gpu models and even with the official support, most of the cards performance numbers are near what is achieved with zluda AT THE MOMENT. (it would hopefully change as AMD make the official pytorch builds better). ***

<details>
<summary><strong>FOR THOSE THAT ARE GETTING TROJAN DETECTIONS IN NCCL.DLL IN ZLUDA FOLDER</strong></summary>

In the developer's words: "nccl.dll is a dummy file, it does nothing. When one of its functions is called, it will just return 'not supported' status. nccl.dll and cufftw.dll are dummy files introduced only for compatibility (to run applications that reject to start without them, but rarely or never use them).

zluda.exe hijacks Windows API and injects some DLLs. Its behavior can be considered malicious by some antiviruses, but it does not hurt the user.

The antiviruses, including Windows Defender on my computer, didn't detect them as malicious when I made the nightly build. But somehow the nightly build is now detected as a virus on my end too."

**SOLUTION: IGNORE THE WARNING AND EXCLUDE THE ZLUDA (or better the whole comfyui-zluda) FOLDER FROM DEFENDER.**
</details>

<details>
<summary><strong>What's New?</strong></summary>

### Recent Updates

- **Changed node storage folder and added CFZ-Condition-Caching node. This allows you to save-load conditionings -prompts basically- it helps on two fronts, if you are using same prompts over and over it skips the clip part AND more importantly it skips loading clip model all together, giving you more memory to load other stuff, main model being the most important. (It is based on this nodepack , https://github.com/alastor-666-1933/caching_to_not_waste)

<img width="1292" height="979" alt="Screenshot 2025-09-02 182907" src="https://github.com/user-attachments/assets/e7ab712b-4adc-426a-932a-acd0e49a30e0" />

* I also uploaded an example workflow on how to use the nodes in your workflows. It is not fully working , and it is there to an idea how to incorporate to your workflows.

- **Added "cfz-vae-loader" node** to CFZ folder - enables changing VAE precision on the fly without using `--fp16-vae` etc. on the starting command line. This is important because while "WAN" works faster with fp16, Flux produces black output if fp16 VAE is used. Start ComfyUI normally and add this node to your WAN workflow to change it only with that model type.

- **Use update.bat** if comfyui.bat or comfyui-n.bat can't update (as when they are the files that need to be updated, so delete them, run update.bat). When you run your comfyui(-n).bat afterwards, it now copies correct ZLUDA and uses that.

- **Updated included ZLUDA version** for the new install method to 3.9.5 nightly (latest version available). You MUST use latest AMD GPU drivers with this setup otherwise there would be problems later (drivers >= 25.5.1).

### Cache Cleaning Instructions

**WIPING CACHES FOR A CLEAN REINSTALL** (recommended for a painless ZLUDA experience):

Delete everything in these three directories:
1. `C:\Users\yourusername\AppData\Local\ZLUDA\ComputeCache`
2. `C:\Users\yourusername\.miopen`
3. `C:\Users\yourusername\.triton`
You can now use the `cache-clean.bat` in the comfyui-zluda folder to clean all caches quickly.

ZLUDA, MIOpen, and Triton will rebuild everything from scratch, making future operations less problematic.

### New Nodes

- **Added "CFZ Cudnn Toggle" node** - for some models not working with cuDNN (which is enabled by default on new install method). To use it:
  - Connect it before KSampler (latent_image input or any latent input)
  - Disable cuDNN
  - After VAE decoding (where most problems occur), re-enable cuDNN
  - Add it after VAE decoding, select audio_output and connect to save audio node
  - Enable cuDNN now

- **"CFZ Checkpoint Loader" was completely redone** - the previous version was broken and might corrupt models if you loaded with it and quit halfway. The new version works outside checkpoint loading, doesn't touch the original file, and when it quantizes the model, it makes a copy first. 
  - Please delete "cfz_checkpoint_loader.py" and use the newly added "cfz_patcher.py"
  - It has three separate nodes and is much safer and better

**Note**: Both nodes are inside the "cfz" folder. To use them, copy them into custom_nodes - they will appear next time you open ComfyUI. To find them, search for "cfz".

### Model Fixes

- **Florence2 is now fixed** (probably some other nodes too) - you need to disable "do_sample", meaning change it from True to False. Now it works without needing to edit its node.

### Custom ZLUDA Versions

- **Added support for any ZLUDA version** - to use with HIP versions you want (such as 6.1 - 6.2). After installing:
  1. Close the app
  2. Run `patchzluda2.bat`
  3. It will ask for URL of the ZLUDA build you want to use
  4. Choose from [lshyqqtiger's ZLUDA Fork](https://github.com/lshqqytiger/ZLUDA/releases)
  5. Paste the link via right-click (correct link example: `https://github.com/lshqqytiger/ZLUDA/releases/download/rel.d60bddbc870827566b3d2d417e00e1d2d8acc026/ZLUDA-windows-rocm6-amd64.zip`)
  6. Press enter and it will patch that ZLUDA into ComfyUI for you

### Documentation

- **Added a "Small Flux Guide"** - aims to use low VRAM and provides the basic necessary files needed to get Flux generation running. [View Guide](fluxguide.md)

</details>

## Dependencies

* Install GIT from `https://git-scm.com/download/win` During installation don't forget to check the box for "Use Git from the Windows Command line and also from 3rd-party-software" to add Git to your system's PATH.
* Install python 3.11.9 or higher from python website `https://www.python.org/downloads/windows/` (not from Microsoft Store on Windows) Make sure you check the box for "Add Python to PATH when you are at the "Customize Python" screen.
* Install visual c++ runtime library from `https://aka.ms/vs/17/release/vc_redist.x64.exe`

## Setup (Windows-Only)

<details>
<summary>For Old Ryzen APU's, RX400-500 Series GPU's</summary>

* Install HIP SDK 5.7.1 from "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html", "Windows 10 & 11 5.7.1 HIP SDK"

* You *might* need older drivers for sdk 5.7.1 and old zluda to work so if you are getting errors with latest drivers please install an older version (below 25.5.1) 

* Install "https://aka.ms/vs/17/release/vs_BuildTools.exe" 

* Make sure the system variables HIP_PATH and HIP_PATH_57 exist, both should have this value: `C:\Program Files\AMD\ROCm\5.7\`

* Also there is the system path defining variable called: "Path". Double-click it and click "New" add this: `C:\Program Files\AMD\ROCm\5.7\bin`

* Get library files for your GPU from Brknsoul Repository (for HIP 5.7.1) `https://github.com/brknsoul/ROCmLibs`
* (try these for many of the old gpu's as an alternative source of libraries `https://www.mediafire.com/file/boobrm5vjg7ev50/rocBLAS-HIP5.7.1-win%2528old_gpu%2529.rar/file`)

* Go to folder `C:\Program Files\AMD\ROCm\5.7\bin\rocblas`, there would be a "library" folder, backup the files inside to somewhere else.

* Open your downloaded optimized library archive and put them inside the library folder (overwriting if necessary): `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`

* There could be a rocblas.dll file in the archive as well, if it is present then copy it inside `C:\Program Files\AMD\ROCm\5.7\bin`

* Restart your system.

* Open a cmd prompt. Easiest way to do this is, in Windows Explorer go to the folder or drive you want to install this app to, in the address bar type "cmd" and press enter.

* **DON'T INSTALL** into your user directory or inside Windows or Program Files directories. Best option just go to `C:\` or `D:\` if you have other drives and open cmd there.

* Copy these commands one by one and press enter after each one:

```bash
git clone https://github.com/patientx/ComfyUI-Zluda
```

```bash
cd ComfyUI-Zluda
```

```bash
install-for-older-amd.bat
```

* If you have done every previous step correctly, it will install without errors and start ComfyUI-ZLUDA for the first time. If you already have checkpoints copy them into `models/checkpoints` folder so you can use them with ComfyUI's default workflow.

* The first generation will take longer than usual, ZLUDA is compiling for your GPU, it does this once for every new model type. This is necessary and unavoidable.

* You can use `comfyui.bat` or put a shortcut of it on your desktop, to run the app later. My recommendation is make a copy of `comfyui.bat` with another name maybe and modify that copy so when updating you won't get into trouble.
</details>

<details>
<summary><strong>For GPU's below 6800 (VEGA, RX 5000 and remaining 6000's; 6700, 6600 etc)</strong></summary>

* **IMPORTANT**: With this install method you MUST make sure you have the latest GPU drivers (specifically you need drivers above 25.5.1)

* **UPDATE** : There are now new libraries for "some of" these gpu's namely these listed below for HIP 6.4.2 BUT they are not tested and some of those probably won't work with the newer triton-miopen stuff so if you want you can try using the 6.4.2 route with those new libraries AND if you are updating from 6.2.4 to 6.4.2 please remember to uninstall hip 6.2.4 THEN delete the rocm folder inside program files otherwise even after uninstall the addon's added files would remain there and may cause problems.  
* AMD Radeon RX 5700 XT (gfx1010) , AMD Radeon Pro V540 (gfx1011) , AMD Radeon RX 5500 XT (gfx1012)
* AMD Radeon RX 6700, AMD Radeon RX 6700 XT (gfx1031) , AMD Radeon RX 6600, AMD Radeon RX 6600 XT (gfx1032)
* AMD Radeon RX 6500, 6500XT, 6500M, 6400, 6300, 6450, W6400 (gfx1034)
* AMD Radeon 680M, AMD Radeon 660M (gfx1035)
* AMD Radeon Graphics 128SP (All 7000 series IGP Variants) (gfx1036)
* AMD Radeon 780M , AMD Ryzen Z1 ,AMD Radeon 740M (gfx1103)
* But they are not all tested and even if they work with 6.4.2 they might not work with triton-miopen stuff, in that case use legacy installer instead. 

* [There is the legacy installer method still available with `install-legacy.bat` (this is the old "install.bat") which doesn't include miopen-triton stuff, but I strongly recommend them now we have solved most of the problems with them.So if you want you can still install hip 5.7.1 and use the libraries for your gpu for hip 5.7.1 or 6.2.4 and you don't need to install miopen stuff. You can use the `install-legacy.bat` but first try the install-n.bat if you have problems than go back to the legacy one.]

* Install HIP SDK 6.2.4 from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) - "Windows 10 & 11 6.2.4 HIP SDK"

* Make sure the system variables `HIP_PATH` and `HIP_PATH_62` exist, both should have this value: `C:\Program Files\AMD\ROCm\6.2\`

* Also check the system path defining variable called "Path". Double-click it and click "New", then add: `C:\Program Files\AMD\ROCm\6.2\bin`

* Download this addon package from [Google Drive](https://drive.google.com/file/d/1Gvg3hxNEj2Vsd2nQgwadrUEY6dYXy0H9/view?usp=sharing) (or [alternative source](https://www.mediafire.com/file/ooawc9s34sazerr/HIP-SDK-extension(zluda395).zip/file))

* Extract the addon package into `C:\Program Files\AMD\ROCm\6.2` overwriting files if asked

* Get library files for your GPU from [likelovewant Repository](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (for HIP 6.2.4)

* Go to folder `C:\Program Files\AMD\ROCm\6.2\bin\rocblas`, there should be a "library" folder. **Backup the files inside to somewhere else.**

* Open your downloaded optimized library archive and put them inside the library folder (overwriting if necessary): `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\library`

* If there's a `rocblas.dll` file in the archive, copy it inside `C:\Program Files\AMD\ROCm\6.2\bin`

* Install [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)

* **Restart your system**

* Open a command prompt. Easiest way: in Windows Explorer, go to the folder or drive where you want to install this app, in the address bar type "cmd" and press enter

* **DON'T INSTALL** into your user directory or inside Windows or Program Files directories. Best option is to go to `C:\` or `D:\` (if you have other drives) and open cmd there.

* Copy these commands one by one and press enter after each:

```bash
git clone https://github.com/patientx/ComfyUI-Zluda
```

```bash
cd ComfyUI-Zluda
```

```bash
install-n.bat
```

* If you have done every previous step correctly, it will install without errors and start ComfyUI-ZLUDA for the first time. If you already have checkpoints, copy them into `models/checkpoints` folder so you can use them with ComfyUI's default workflow.

* The first generation will take longer than usual, ZLUDA is compiling for your GPU. It does this once for every new model type. This is necessary and unavoidable.

* You can use `comfyui-n.bat` or put a shortcut of it on your desktop to run the app later. My recommendation is to make a copy of `comfyui-n.bat` with another name and modify that copy so when updating you won't get into trouble.

</details>

<details>
<summary><strong>For GPU's above 6800 (6800, 6800XT, other 6000s above & 7000s and 9000s)</strong></summary>

* **IMPORTANT**: With this install method you MUST make sure you have the latest GPU drivers (specifically you need drivers above 25.5.1)
* **UPDATE** : There are now new libraries for "some of" the unsupported gpu's from previous generations namely these listed below for HIP 6.4.2 BUT they are not tested and some of those probably won't work with the newer triton-miopen stuff so if you want you can try using the 6.4.2 route with those new libraries AND if you are updating from 6.2.4 to 6.4.2 please remember to uninstall hip 6.2.4 THEN delete the rocm folder inside program files otherwise even after uninstall the addon's added files would remain there and may cause problems.  
* AMD Radeon RX 5700 XT (gfx1010) , AMD Radeon Pro V540 (gfx1011) , AMD Radeon RX 5500 XT (gfx1012)
* AMD Radeon RX 6700, AMD Radeon RX 6700 XT (gfx1031) , AMD Radeon RX 6600, AMD Radeon RX 6600 XT (gfx1032)
* AMD Radeon RX 6500, 6500XT, 6500M, 6400, 6300, 6450, W6400 (gfx1034)
* AMD Radeon 680M, AMD Radeon 660M (gfx1035)
* AMD Radeon Graphics 128SP (All 7000 series IGP Variants) (gfx1036)
* AMD Radeon 780M , AMD Ryzen Z1 ,AMD Radeon 740M (gfx1103)
* The libraries for these can be had here : [6.4.2 Libraries for unsupported GPU's](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.4.2)
* But they are not all tested and even if they work with 6.4.2 they might not work with triton-miopen stuff, in that case use legacy installer instead. 

* Install HIP SDK 6.4.2 from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) - "Windows 10 & 11 6.4.2 HIP SDK"

* Make sure the system variables `HIP_PATH` and `HIP_PATH_64` exist, both should have this value: `C:\Program Files\AMD\ROCm\6.4\`

* Also check the system path defining variable called "Path". Double-click it and click "New", then add: `C:\Program Files\AMD\ROCm\6.4\bin`

* FOR THE UNSUPPORTED GPU'S THAT HAS LIBRARIES ; DO THESE :

* Get library files for your GPU from [6.4.2 Libraries for unsupported GPU's](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.4.2) (for HIP 6.4.2)

* Go to folder `C:\Program Files\AMD\ROCm\6.4\bin\rocblas`, there should be a "library" folder. **Backup the files inside to somewhere else.**

* Open your downloaded optimized library archive and put them inside the library folder (overwriting if necessary): `C:\Program Files\AMD\ROCm\6.4\bin\rocblas\library`

* If there's a `rocblas.dll` file in the archive, copy it inside `C:\Program Files\AMD\ROCm\6.4\bin`

* Install [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)

* **Restart your system**

* Open a command prompt. Easiest way: in Windows Explorer, go to the folder or drive where you want to install this app, in the address bar type "cmd" and press enter

* **DON'T INSTALL** into your user directory or inside Windows or Program Files directories. Best option is to go to `C:\` or `D:\` (if you have other drives) and open cmd there.

* Copy these commands one by one and press enter after each:

```bash
git clone https://github.com/patientx/ComfyUI-Zluda
```

```bash
cd ComfyUI-Zluda
```

```bash
install-n.bat
```

* If you have done every previous step correctly, it will install without errors and start ComfyUI-ZLUDA for the first time. If you already have checkpoints, copy them into `models/checkpoints` folder so you can use them with ComfyUI's default workflow.

* The first generation will take longer than usual, ZLUDA is compiling for your GPU. It does this once for every new model type. This is necessary and unavoidable.

* You can use `comfyui-n.bat` or put a shortcut of it on your desktop to run the app later. My recommendation is to make a copy of `comfyui-n.bat` with another name and modify that copy so when updating you won't get into trouble.

</details>

> [!IMPORTANT]
> 📢 ***REGARDING KEEPING THE APP UP TO DATE***
>
> Avoid using the update function from the manager, instead use `git pull`, which we
> are doing on every start if `comfyui.bat` or `comfyui-n.bat`is used. 
>
> Only use comfy manager to update the extensions
> (Manager -> Custom Nodes Manager -> Set Filter To Installed -> Click Check Update On The Bottom Of The Window)
> otherwise it breaks the basic installation, and in that case run the install once again.

## Troubleshooting

- If you are getting `RuntimeError: GET was unable to find an engine to execute this computation` or `RuntimeError: FIND was unable to find an engine to execute this computation` in the vae decoding stage, please use CFZ CUDNN Toggle node between ksampler latent and vae decoding. And leave the enable_cudnn setting "False" , this persists until you close the comfyui from the commandline for the rest of that run. So you won't see this error again.

<img width="667" height="350" alt="Screenshot 2025-08-25 123335" src="https://github.com/user-attachments/assets/db56d460-34aa-4fda-94e2-f0bae7162691" />

That node can actually be used between conditioning or image loading etc so it's not only usable between latent and vae decoding , also you can use it in a simple workflow that it makes the setting disabled , than you can use any other workflow for the rest of the instance without worry. (try running the  [1step-cudnn-disabler-workflow](https://github.com/patientx/ComfyUI-Zluda/blob/master/cfz/workflows/1step-cudnn-disabler-workflow.json) in cfz folder once after you start comfyui-zluda, it can use any sd15 or sdxl model it would just generate 1 step image than preview it so no saving) after that workflow runs once, switch to any workflow or start a new one.

- Problems with triton , try this : Remove visual studio 2022 (if you have already installed it and getting errors) and install "https://aka.ms/vs/17/release/vs_BuildTools.exe" , and then use  "Developer Command Prompt" to run comfyui. This option shouldn't be needed for many but nevertheless try.
- `RuntimeError: CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize FailedmiopenStatusInternalError cudnn_status: miopenStatusUnknownError` , if this is encountered at the end, while vae-decoding, use tiled-vae decoding either from official comfy nodes or from Tiled-Diffussion (my preference). Also vae-decoding is overall better with tiled-vae decoding. 
- DO NOT use non-english characters as folder names to put comfyui-zluda under.
- Wipe your pip cache "C:\Users\USERNAME\AppData\Local\pip\cache" You can also do this when venv is active with :  `pip cache purge`
- `xformers` isn't usable with zluda so any nodes / packages that require it doesn't work.
- Have the latest drivers installed for your amd gpu. **Also, Remove Any Nvidia Drivers** you might have from previous nvidia gpu's.
- If for some reason you can't solve with these and want to start from zero, delete "venv" folder and re-run the whole setup again step by step.
- If you can't git pull to the latest version, run these commands, `git fetch --all` and then `git reset --hard origin/master` now you can git pull
- Problems with `caffe2_nvrtc.dll`: if you are sure you properly installed hip and can see it on path, please DON'T use
  python from windows store, use the link provided or 3.11 from the official website. After uninstalling python from
  windows store and installing the one from the website, be sure the delete venv folder, and run install.bat once again.
- If you have an integrated GPU by AMD (e.g. AMD Radeon(TM) Graphics) you need to add `HIP_VISIBLE_DEVICES=1` to your environment variables. Other possible variables to use :
   `ROCR_VISIBLE_DEVICES=1` `HCC_AMDGPU_TARGET=1` . This basically tells it to use 1st gpu -this number could be different if you have multiple gpu's-
  Otherwise it will default to using your iGPU, which will most likely not work. This behavior is caused by a bug in the ROCm-driver.
- Lots of other problems were encountered and solved by users so check the issues if you can't find your problem here.  

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Zluda Wiki from SdNext](https://github.com/vladmandic/sdnext/wiki/ZLUDA)
- [Brknsoul for Rocm Libraries](https://github.com/brknsoul/ROCmLibs)
- [likelovewant for Rocm Libraries](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4)
- [Lshqqytiger](https://github.com/lshqqytiger/ZLUDA)
- [LeagueRaINi](https://github.com/LeagueRaINi/ComfyUI)
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
