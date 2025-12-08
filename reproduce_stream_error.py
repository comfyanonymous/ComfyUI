
import torch
import logging

logging.basicConfig(level=logging.INFO)

def test_stream():
    if not torch.cuda.is_available():
        print("CUDA not available, cannot test cuda stream")
        return

    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device, priority=0)
    
    print(f"Stream type: {type(stream)}")
    print(f"Has __enter__: {hasattr(stream, '__enter__')}")
    
    try:
        with stream:
            print("Stream context manager works")
    except AttributeError as e:
        print(f"AttributeError caught: {e}")
    except Exception as e:
        print(f"Other exception caught: {e}")

if __name__ == "__main__":
    test_stream()
