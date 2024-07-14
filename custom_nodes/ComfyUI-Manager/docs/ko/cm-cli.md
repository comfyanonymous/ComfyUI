# `cm-cli`: ComfyUI-Manager CLI

`cm-cli` 는 ComfyUI를 실행시키지 않고 command line에서 ComfyUI-Manager의 여러가지 기능을 사용할 수 있도록 도와주는 도구입니다.


```
-= ComfyUI-Manager CLI (V2.24) =-


python cm-cli.py [OPTIONS]

OPTIONS:
    [install|reinstall|uninstall|update|disable|enable|fix] node_name ... ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    [update|disable|enable|fix] all ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    [simple-show|show] [installed|enabled|not-installed|disabled|all|snapshot|snapshot-list] ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    save-snapshot ?[--output <snapshot .json/.yaml>]
    restore-snapshot <snapshot .json/.yaml> ?[--pip-non-url] ?[--pip-non-local-url] ?[--pip-local-url]
    cli-only-mode [enable|disable]
    restore-dependencies
    clear
```

## How To Use?
* `python cm-cli.py` 를 통해서 실행 시킬 수 있습니다.
* 예를 들어 custom node를 모두 업데이트 하고 싶다면
    * ComfyUI-Manager경로 에서 `python cm-cli.py update all` 를 command를 실행할 수 있습니다.
    * ComfyUI 경로에서 실행한다면, `python custom_nodes/ComfyUI-Manager/cm-cli.py update all` 와 같이 cm-cli.py 의 경로를 지정할 수도 있습니다.

## Prerequisite
* ComfyUI 를 실행하는 python과 동일한 python 환경에서 실행해야 합니다.
    * venv를 사용할 경우 해당 venv를 activate 한 상태에서 실행해야 합니다.
    * portable 버전을 사용할 경우 run_nvidia_gpu.bat 파일이 있는 경로인 경우, 다음과 같은 방식으로 코맨드를 실행해야 합니다.
        `.\python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Manager\cm-cli.py update all`
* ComfyUI 의 경로는 COMFYUI_PATH 환경 변수로 설정할 수 있습니다. 만약 생략할 경우 다음과 같은 경고 메시지가 나타나며, ComfyUI-Manager가 설치된 경로를 기준으로 상대 경로로 설정됩니다.
        ```
        WARN: The `COMFYUI_PATH` environment variable is not set. Assuming `custom_nodes/ComfyUI-Manager/../../` as the ComfyUI path.
        ```

## Features

### 1. --channel, --mode
* 정보 보기 기능과 커스텀 노드 관리 기능의 경우는 --channel과 --mode를 통해 정보 DB를 설정할 수 있습니다.
* 예들 들어 `python cm-cli.py update all --channel recent --mode remote`와 같은 command를 실행할 경우, 현재 ComfyUI-Manager repo에 내장된 로컬의 정보가 아닌 remote의 최신 정보를 기준으로 동작하며, recent channel에 있는 목록을 대상으로만 동작합니다.
* --channel, --mode 는 `simple-show, show, install, uninstall, update, disable, enable, fix` command에서만 사용 가능합니다.

### 2. 관리 정보 보기

`[simple-show|show] [installed|enabled|not-installed|disabled|all|snapshot|snapshot-list] ?[--channel <channel name>] ?[--mode [remote|local|cache]]`


* `[show|simple-show]` - `show`는 상세하게 정보를 보여주며, `simple-show`는 간단하게 정보를 보여줍니다.


`python cm-cli.py show installed` 와 같은 코맨드를 실행하면 설치된 커스텀 노드의 정보를 상세하게 보여줍니다.
```
-= ComfyUI-Manager CLI (V2.24) =-

FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
[    ENABLED    ]  ComfyUI-Manager                                   (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI-Impact-Pack                               (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI-Inspire-Pack                              (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI_experiments                               (author: comfyanonymous)
[    ENABLED    ]  ComfyUI-SAI_API                                   (author: Stability-AI)
[    ENABLED    ]  stability-ComfyUI-nodes                           (author: Stability-AI)
[    ENABLED    ]  comfyui_controlnet_aux                            (author: Fannovel16)
[    ENABLED    ]  ComfyUI-Frame-Interpolation                       (author: Fannovel16)
[    DISABLED   ]  ComfyUI-Loopchain                                 (author: Fannovel16)
```

`python cm-cli.py simple-show installed` 와 같은 코맨드를 이용해서 설치된 커스텀 노드의 정보를 간단하게 보여줍니다.

```
-= ComfyUI-Manager CLI (V2.24) =-

FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
ComfyUI-Manager                                   
ComfyUI-Impact-Pack                               
ComfyUI-Inspire-Pack                              
ComfyUI_experiments                               
ComfyUI-SAI_API                                   
stability-ComfyUI-nodes                           
comfyui_controlnet_aux                            
ComfyUI-Frame-Interpolation                       
ComfyUI-Loopchain                                 
```

* `[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]`
    * `enabled`, `disabled`: 설치된 커스텀 노드들 중 enable 되었거나, disable된 노드들을 보여줍니다.
    * `installed`: enable, disable 여부와 상관없이 설치된 모든 노드를 보여줍니다
    * `not-installed`: 설치되지 않은 커스텀 노드의 목록을 보여줍니다.
    * `all`: 모든 커스텀 노드의 목록을 보여줍니다.
    * `snapshot`: 현재 설치된 커스텀 노드의 snapshot 정보를 보여줍니다. `show`롤 통해서 볼 경우는 json 출력 형태로  보여주며, `simple-show`를 통해서 볼 경우는 간단하게, 커밋 해시와 함께 보여줍니다.
    * `snapshot-list`: ComfyUI-Manager/snapshots 에 저장된 snapshot 파일의 목록을 보여줍니다.

### 3. 커스텀 노드 관리 하기

`[install|reinstall|uninstall|update|disable|enable|fix] node_name ... ?[--channel <channel name>] ?[--mode [remote|local|cache]]`

* `python cm-cli.py install ComfyUI-Impact-Pack ComfyUI-Inspire-Pack ComfyUI_experiments` 와 같이 커스텀 노드의 이름을 나열해서 관리 기능을 적용할 수 있습니다.
* 커스텀 노드의 이름은 `show`를 했을 때 보여주는 이름이며, git repository의 이름입니다. 
(추후 nickname 을 사용가능하돌고 업데이트 할 예정입니다.)

`[update|disable|enable|fix] all ?[--channel <channel name>] ?[--mode [remote|local|cache]]`

* `update, disable, enable, fix` 기능은 all 로 지정 가능합니다.

* 세부 동작
    * `install`: 지정된 커스텀 노드들을 설치합니다
    * `reinstall`: 지정된 커스텀 노드를 삭제하고 재설치 합니다.
    * `uninstall`: 지정된 커스텀 노드들을 삭제합니다.
    * `update`: 지정된 커스텀 노드들을 업데이트합니다.
    * `disable`: 지정된 커스텀 노드들을 비활성화합니다.
    * `enable`: 지정된 커스텀 노드들을 활성화합니다.
    * `fix`: 지정된 커스텀 노드의 의존성을 고치기 위한 시도를 합니다.


### 4. 스냅샷 관리 기능
* `python cm-cli.py save-snapshot ?[--output <snapshot .json/.yaml>]`: 현재의 snapshot을 저장합니다.
  * --output 으로 임의의 경로에 .yaml 파일과 format으로 저장할 수 있습니다.
* `python cm-cli.py restore-snapshot <snapshot .json/.yaml>`: 지정된 snapshot으로 복구합니다.
  * snapshot 경로에 파일이 존재하는 경우 해당 snapshot을 로드합니다.
  * snapshot 경로에 파일이 존재하지 않는 경우 묵시적으로, ComfyUI-Manager/snapshots 에 있다고 가정합니다.
  * `--pip-non-url`: PyPI 에 등록된 pip 패키지들에 대해서 복구를 수행
  * `--pip-non-local-url`: web URL에 등록된 pip 패키지들에 대해서 복구를 수행
  * `--pip-local-url`: local 경로를 지정하고 있는 pip 패키지들에 대해서 복구를 수행 


### 5. CLI only mode

ComfyUI-Manager를 CLI로만 사용할 것인지를 설정할 수 있습니다.

`cli-only-mode [enable|disable]`

* security 혹은 policy 의 이유로 GUI 를 통한 ComfyUI-Manager 사용을 제한하고 싶은 경우 이 모드를 사용할 수 있습니다.
    * CLI only mode를 적용할 경우 ComfyUI-Manager 가 매우 제한된 상태로 로드되어, 내부적으로 제공하는 web API가 비활성화 되며, 메인 메뉴에서도 Manager 버튼이 표시되지 않습니다.


### 6. 의존성 설치

`restore-dependencies`

* `ComfyUI/custom_nodes` 하위 경로에 커스텀 노드들이 설치되어 있긴 하지만, 의존성이 설치되지 않은 경우 사용할 수 있습니다.
* colab 과 같이 cloud instance를 새로 시작하는 경우 의존성 재설치 및 설치 스크립트가 재실행 되어야 하는 경우 사용합니다.
* ComfyUI을 재설치할 경우, custom_nodes 경로만 백업했다가 재설치 할 경우 활용 가능합니다.


### 7. clear

GUI에서 install, update를 하거나 snapshot 을 restore하는 경우 예약을 통해서 다음번 ComfyUI를 실행할 경우 실행되는 구조입니다. `clear` 는 이런 예약 상태를 clear해서, 아무런 사전 실행이 적용되지 않도록 합니다.