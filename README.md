<div align="center">

# ComfyUI-ZLUDA

Windows-only version of ComfyUI which uses ZLUDA to get better performance with AMD GPUs.

</div>

## Table of Contents

- [What's New?](#whats-new)
- [Dependencies](#dependencies)
- [Setup (Windows-Only)](#setup-windows-only)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## What's New?

* Reverted zluda version back to 3.8.4. After updating try running patchzluda and if still have problems, delete venv and re-run install.bat.
* U̶p̶d̶a̶t̶e̶d̶ Z̶L̶U̶D̶A̶ v̶e̶r̶s̶i̶o̶n̶ t̶o̶ 3̶.8̶.5̶. I̶f̶ y̶o̶u̶ h̶a̶v̶e̶ a̶l̶r̶e̶a̶d̶y̶ i̶n̶s̶t̶a̶l̶l̶e̶d̶ c̶o̶m̶f̶y̶u̶i̶-̶z̶l̶u̶d̶a̶, y̶o̶u̶ c̶a̶n̶ u̶p̶d̶a̶t̶e̶ z̶l̶u̶d̶a̶ w̶i̶t̶h̶ r̶u̶n̶n̶i̶n̶g̶ `̶p̶a̶t̶c̶h̶z̶l̶u̶d̶a̶.b̶a̶t̶`̶ o̶n̶c̶e̶. O̶f̶ c̶o̶u̶r̶s̶e̶, r̶e̶m̶e̶m̶b̶e̶r̶ t̶h̶e̶  f̶i̶r̶s̶t̶ t̶i̶m̶e̶ f̶o̶r̶ e̶v̶e̶r̶y̶ t̶y̶p̶e̶ o̶f̶ m̶o̶d̶e̶l̶ w̶o̶u̶l̶d̶ t̶a̶k̶e̶ e̶x̶t̶r̶a̶ t̶i̶m̶e̶. 
* Added a "small flux guide." This aims to use low vram and provides the very basic necessary files needed to get flux generation running. [HERE](fluxguide.md)
* Added --reserve-vram with the value of 0.9 to commandline options that run with the app on startup. Greatly helps reduce using too much memory on generations.
* Changed `start.bat` to `comfyui.bat` because there is already a windows command by that name, which
  creates some problems. Also added  `fix-update.bat` which solves the problem that causes not being able to
  update to the latest version.
* Updated ZLUDA to 3.8.4, thanks to lshqqytiger for still supporting HIP SDK 5.7.1. Install will install that version from
  now on, and if you are already on the previous one please run `patchzluda.bat` once. Of course, remember the
  first time for every type of model would take extra time. You can still use both zluda's on the same machine btw. But
  I recommend updating.

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
2. **Python** (3.10.11 or 3.11 from the official website): Install the latest release from python.org. **Don't Use
   Windows Store Version**. If you have that installed, uninstall and please install from python.org. During
   installation remember to check the box for "Add Python to PATH when you are at the "Customize Python" screen.
3. **Visual C++ Runtime**: Download [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) and install it.
4. Install **HIP SDK 5.7.1** from [HERE](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
    - **Update**: HIP 6.1.x is available but as of 01.2025 there are no speed improvements of it over using 5.7.1 as far as I know , please be careful about
      selecting the correct version, "Windows 10 & 11 5.7.1 HIP SDK"
5. Add the system variable HIP_PATH, value: `C:\\Program Files\\AMD\\ROCm\\5.7\\` (This is the default folder, if you
   have installed it on another drive, change if necessary)
    1. Check the variables on the lower part (System Variables), there should be a variable called: HIP_PATH.
    2. Also check the variables on the lower part (System Variables), there should be a variable called: "Path".
       Double-click it and click "New" add this: `C:\Program Files\AMD\ROCm\5.7\bin`
6. If you have an AMD GPU below 6800 (6700,6600 etc.), download the recommended library files for your gpu
   from [Brknsoul Repository](https://github.com/brknsoul/ROCmLibs)
    1. Go to folder "C:\Program Files\AMD\ROCm\5.7\bin\rocblas", there would be a "library" folder, backup the files
       inside to somewhere else.
    2. Open your downloaded optimized library archive and put them inside the library folder (overwriting if
       necessary): "C:\\Program Files\\AMD\\ROCm\\5.7\\bin\\rocblas\\library"
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

to start for later use (or create a shortcut to) :

```bash
comfyui.bat
```

also for later when you need to repatch zluda (maybe a torch update etc.) you can use:

```bash
patchzluda.bat
```

- The first generation would take around 10-15 minutes, there won't be any progress or indicator on the webui or cmd
  window, just wait. Zluda creates a database for use with generation with your gpu.

> [!NOTE]
> **This might happen with torch changes , zluda version changes and / or gpu driver changes.**

## Troubleshooting

- Wipe your pip cache "C:\Users\USERNAME\AppData\Local\pip\cache" You can also do this when venv is active with :
  `pip cache purge`
- `xformers` isn't usable with zluda so any nodes / packages that require it doesn't work. `Flash attention`
  doesn't work. And lastly using `codeformer` for face restoration gives "Failed inference: CUDA driver error:
  unknown error" You should use gfpgan / gpen / restoreformer or other face restoration models.
- Have the latest drivers installed for your amd gpu. **Also, Remove Any Nvidia Drivers** you might have from previous
  nvidia
  gpu's.
- If you see zluda errors make sure these three files are inside "ComfyUI-Zluda\venv\Lib\site-packages\torch\lib\"
  `cublas64_11.dll (231kb)` `cusparse64_11.dll (199kb)` `nvrtc64_112_0.dll (129kb)` If they are there but much bigger in size please run : `patchzluda.bat`
- If for some reason you can't solve with these and want to start from zero, delete "venv" folder and re-run
  `install.bat`
- If you can't git pull to the latest version, run these commands, `git fetch --all` and then
  `git reset --hard origin/master` now you can git pull
- Problems with `caffe2_nvrtc.dll`: if you are sure you properly installed hip and can see it on path, please DON'T use
  python from windows store, use the link provided or 3.11 from the official website. After uninstalling python from
  windows store and installing the one from the website, be sure the delete venv folder, and run install.bat once again.
- `rocBLAS`-error: If you have an integrated GPU by AMD (e.g. AMD Radeon(TM) Graphics) you need to add `HIP_VISIBLE_DEVICES=1` to your
  environment variables. Otherwise it will default to using your iGPU, which will most likely not work. This behavior is caused by a bug in the ROCm-driver.

___

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Zluda Wiki from SdNext](https://github.com/vladmandic/automatic/wiki/ZLUDA)
- [Brknsoul for Rocm Libraries](https://github.com/brknsoul/ROCmLibs)
- [Lshqqytiger](https://github.com/lshqqytiger/ZLUDA)
- [LeagueRaINi](https://github.com/LeagueRaINi/ComfyUI)
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
