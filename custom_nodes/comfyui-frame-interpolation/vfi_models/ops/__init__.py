import torch.multiprocessing as mp

if mp.current_process().name == "MainProcess":
    import yaml
    import os
    from pathlib import Path

    config_path = Path(Path(__file__).parent.parent.parent.resolve(), "config.yaml")

    if os.path.exists(config_path):
        config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        ops_backend = config["ops_backend"]
    else:
        ops_backend = "taichi"

    assert ops_backend in ["taichi", "cupy"]

    if ops_backend == "taichi":
        from .taichi_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
    else:
        from .cupy_ops import softsplat, ModuleSoftsplat, FunctionSoftsplat, softsplat_func, costvol_func, sepconv_func, init, batch_edt, FunctionAdaCoF, ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation

