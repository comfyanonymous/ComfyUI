import cupy
import os
import re
import torch
import typing
from pathlib import Path
import platform

##########################################################


objCudacache = {}


def cuda_int32(intIn: int):
    return cupy.int32(intIn)


# end


def cuda_float32(fltIn: float):
    return cupy.float32(fltIn)


# end


def cuda_kernel(strFunction: str, strKernel: str, objVariables: typing.Dict, **replace_kwargs):
    if "device" not in objCudacache:
        objCudacache["device"] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert False

        # end
    # end

    strKey += objCudacache["device"]

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace("{{" + strVariable + "}}", str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace("{{" + strVariable + "}}", objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace("{{type}}", "unsigned char")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace("{{type}}", "half")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace("{{type}}", "float")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace("{{type}}", "double")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace("{{type}}", "int")

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace("{{type}}", "long")

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert False

            elif True:
                print(strVariable, type(objValue))
                assert False

            # end
        # end

        while True:
            objMatch = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
        # end

        while True:
            objMatch = re.search("(OFFSET_)([0-4])(\()([^\)]+)(\))", strKernel)

            if objMatch is None:
                break
            # end

            intArgs = int(objMatch.group(2))
            strArgs = objMatch.group(4).split(",")

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()
            strIndex = [
                "(("
                + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
                + ")*"
                + str(intStrides[intArg])
                + ")"
                for intArg in range(intArgs)
            ]

            strKernel = strKernel.replace(
                objMatch.group(0), "(" + str.join("+", strIndex) + ")"
            )
        # end

        while True:
            objMatch = re.search("(VALUE_)([0-4])(\()", strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == "(" else 0
                intParentheses -= 1 if strKernel[intStop] == ")" else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(",")

            assert intArgs == len(strArgs) - 1

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append(
                    "(("
                    + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
                    + ")*"
                    + str(intStrides[intArg])
                    + ")"
                )
            # end

            strKernel = strKernel.replace(
                "VALUE_" + str(intArgs) + "(" + strKernel[intStart:intStop] + ")",
                strTensor + "[" + str.join("+", strIndex) + "]",
            )
        # end

        for replace_key, value in replace_kwargs.items():
            strKernel = strKernel.replace(replace_key, value)

        objCudacache[strKey] = {"strFunction": strFunction, "strKernel": strKernel}
    # end

    return strKey


# end
def get_cuda_home_path():
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    import torch
    torch_lib_path = Path(torch.__file__).parent / "lib"
    torch_lib_path = str(torch_lib_path.resolve())
    if os.path.exists(torch_lib_path):
        nvrtc = filter(lambda lib_file: "nvrtc-builtins" in lib_file, os.listdir(torch_lib_path))
        nvrtc = list(nvrtc)
        return torch_lib_path if len(nvrtc) > 0 else None

@cupy.memoize(for_each_device=True)
def cuda_launch(strKey: str):
    if True:#"CUDA_HOME" not in os.environ:
        cuda_home = get_cuda_home_path()
        if cuda_home is not None:
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
        else:
            os.environ["CUDA_HOME"] = "/usr/local/cuda/"
            os.environ["CUDA_PATH"] = "/usr/local/cuda/"
    # print(objCudacache[strKey]['strKernel'])
    # return cupy.cuda.compile_with_cache(objCudacache[strKey]['strKernel'], tuple(['-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include'])).get_function(objCudacache[strKey]['strFunction'])
    return cupy.RawModule(code=objCudacache[strKey]["strKernel"]).get_function(
        objCudacache[strKey]["strFunction"]
    )
