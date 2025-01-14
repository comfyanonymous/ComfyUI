try:
    torch_device_name = get_torch_device_name(get_torch_device())
   
    if "[ZLUDA]" in torch_device_name:
        _torch_stft = torch.stft
        _torch_istft = torch.istft

        def z_stft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
            return _torch_stft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(input.device)

        def z_istft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
            return _torch_istft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(input.device)

        def z_jit(f, *_, **__):
            f.graph = torch._C.Graph()
            return f

    # hijacks
        torch.stft = z_stft
        torch.istft = z_istft
        torch.jit.script = z_jit
        print(" ")
        print("***----------------------ZLUDA--------------------------***")
        print("  ::  ZLUDA detected, disabling non-supported functions.")
        torch.backends.cudnn.enabled = False
        print("  ::  (cuDNN, flash_sdp, mem_efficient_sdp disabled) ")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        print("***-----------------------------------------------------***")
    print("  ::  Device:", torch_device_name)
    print(" ")