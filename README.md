<h1># ComfyUI-ZLUDA</h1>

* Windows only version of comfyui which uses ZLUDA to get better performance with AMD GPUs.

<h2>## What's New ?</h2>

* Changed  ``` start.bat  ```  to  ``` comfyui.bat ```  because there is already a windows command by that name, which creates some problems. Also added  ``` fix-update.bat  ``` which solves the problem that causes not being able to update to latest version.

* Updated ZLUDA to 3.8.4, thanks for lshqqytiger still supporting HIP SDK 5.7 . Install will install that version from now on, and if you are already on the previous one please run ``` patchzluda.bat ``` once. Of course remember the first time for every type of model would take extra time. You can still use both zluda's on the same machine btw. But I recommend updating.

*** BIG UPDATE. Things got broken, had to reset the fork, to get back and update successfully , on the comfyui-zluda directory run these one after another : ``` git fetch --all ``` (enter) ``` git reset --hard origin/master ``` (enter) now you can run comfyui.bat , it will update to the latest version. And also after this a reboot of windows might be needed if the generation time seems to be low. After reboot speed returns to normal.

* Regarding keeping the app up to date, don't use the update function from the manager, instead use git pull which we are doing on every start if start.bat is used. (APP ALREADY DOES IT EVERYTIME YOU OPEN IT , IF YOU ARE USING COMFYUI.BAT, SO THIS WAY IT IS ALWAYS UP-TO-DATE WITH WHATEVER IS ON MY GITHUB PAGE)  Only use comfy manager to update the extensions  ( MANAGER - CUSTOM NODES MANAGER - SET FILTER TO INSTALLED - CLICK CHECK UPDATE ON THE BOTTOM OF THE WINDOW) otherwise it breaks the basic installation, and in that case run install.bat once again.
---

<details>
 <summary><h2>## CREDITS</h2></summary>
 
- comfyui (https://github.com/comfyanonymous/ComfyUI)
- Zluda wiki from sdnext (https://github.com/vladmandic/automatic/wiki/ZLUDA)
- brknsoul for rocm libraries (https://github.com/brknsoul/ROCmLibs)
- lshqqytiger (https://github.com/lshqqytiger/ZLUDA)
- LeagueRaINi (https://github.com/LeagueRaINi/ComfyUI)
- ComfyUI-Manager (https://github.com/ltdrdata/ComfyUI-Manager)
 </details>

<details open>
 <summary><h2>## DEPENDENCIES</h2></summary>

If coming from the very start, you need :

1. **Git** : download from https://git-scm.com/download/win .
	During installation don't forget to check the box for "Use Git from the Windows Command line and also from 3rd-party-software" to add Git to your system's PATH .

2. **Python** (3.10.11 or 3.11 from the official website) : Install latest release from python.org .  DON'T USE WINDOWS STORE VERSION. If you have that installed, uninstall and please install from python.org .
	  During installation don't forget to check the box for "Add Python to PATH when you are at the the "Customize Python" screen.

3. **Visual C++ Runtime** : download from https://aka.ms/vs/17/release/vc_redist.x64.exe , install it.

4. Install **HIP SDK 5.7.1** from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html ((UPDATE: HIP 6.1.2 released now, but there are problems so no need to use that one please be careful about selecting the correct version, "Windows 10 & 11	5.7.1	HIP SDK"))

5.  Add the system variable HIP_PATH , value : C:\\Program Files\\AMD\\ROCm\\5.7\\ (this is the default folder, if you installed it on another drive, change if necessary)

- 	1. Check the variables on the lower part (System Variables) , there should be a variable called : HIP_PATH .
- 	2. Also check the variables on the lower part (System Variables) , there should be a variable called : "Path".	   Double click it and click "New" add this : C:\Program Files\AMD\ROCm\5.7\bin
	
6. If you have an AMD GPU below 6800 (6700,6600 etc.) , download the recommended library files for you gpu from : https://github.com/brknsoul/ROCmLibs/

- 	1. Go to folder C:\Program Files\AMD\ROCm\5.7\bin\rocblas , there would be a "library" folder, backup the files inside to somewhere else.
- 	2. Open your downloaded optimized library archive and put them inside the library folder (overwriting if necessary) : "C:\\Program Files\\AMD\\ROCm\\5.7\\bin\\rocblas\\library"

7. Reboot your system.
</details>

<h2>## SETUP (FOR WINDOWS ONLY)</h2>

Open a cmd prompt. 

```bash
git clone https://github.com/patientx/ComfyUI-Zluda
```
```bash
cd ComfyUI-Zluda
```
```bash
install.bat
```
to start for later use (or create a shortcut to) :
```bash
comfyui.bat
```
also for later when you need to repatch zluda (maybe a torch update etc.) you can use :
```bash
patchzluda.bat
```
- The first generation would take around 10-15 minutes, there won't any progress or indicator on the webui or cmd window, just wait. Zluda creates a database for use with generation with your gpu.

 ** !!! This might happen with torch changes , zluda version changes and / or gpu driver changes. !!! **

<h2>## TROUBLESHOOTING</h2>

- wipe your pip cache ( C:\Users\[your windows username]\AppData\Local\pip\cache )
  You can also do this when venv is active with : ``` pip cache purge ```
  
- ```xformers``` isn't usable with zluda so any nodes / packages that require it doesn't work. ```Flash attention``` doesn't work. And lastly using ```codeformer``` for face restoration gives "Failed inference: CUDA driver error: unknown error" You should use gfpgan / gpen / restoreformer or other face restoration models.
  
- have the latest drivers installed for your amd gpu. ALSO REMOVE ANY NVIDIA DRIVERS you might have from previous nvidia gpu's.
- if you see zluda errors make sure these three files are inside "ComfyUI-Zluda\venv\Lib\site-packages\torch\lib\" 
   cublas64_11.dll (196kb) cusparse64_11.dll (193kb) nvrtc64_112_0.dll (125kb)
  If they are there but bigger files run : ``` patchzluda.bat ```
- if for some reason you can't solve with these and want to start from zero, delete "venv" folder and re-run ``` install.bat ```
- If you can't git pull to the latest version , run these commands, ``` git fetch --all ``` and then ``` git reset --hard origin/master ``` now you can git pull
- Problems with "caffe2_nvrtc.dll" : if you are sure you properly installed hip and can see it on path, please DON'T use python from windows store, use the link provided or 3.11 from the official website. After uninstalling python from windows store and installing the one from the website, be sure the delete venv folder, and run install.bat once again.
