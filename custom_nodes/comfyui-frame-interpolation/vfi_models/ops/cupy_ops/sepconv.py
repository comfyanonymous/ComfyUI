import torch
from .utils import cuda_launch, cuda_kernel, cuda_int32

sepconv_vergrad = """
    extern "C" __global__ void __launch_bounds__(512) sepconv_vergrad(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenVer,
        const {{type}}* __restrict__ tenHor,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenIngrad,
        {{type}}* __restrict__ tenVergrad,
        {{type}}* __restrict__ tenHorgrad
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad) / SIZE_1(tenVergrad) ) % SIZE_0(tenVergrad);
        const int intC = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad)                      ) % SIZE_1(tenVergrad);
        const int intY = ( intIndex / SIZE_3(tenVergrad)                                           ) % SIZE_2(tenVergrad);
        const int intX = ( intIndex                                                                ) % SIZE_3(tenVergrad);

        {{type}} fltVergrad = 0.0;

        {{type}} fltKahanc = 0.0;
        {{type}} fltKahany = 0.0;
        {{type}} fltKahant = 0.0;

        for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
            for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                fltKahany = VALUE_4(tenHor, intN, intFx, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intC, intX + intFx) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                fltKahany = fltKahany - fltKahanc;
                fltKahant = fltVergrad + fltKahany;
                fltKahanc = (fltKahant - fltVergrad) - fltKahany;
                fltVergrad = fltKahant;
            }
        }

        tenVergrad[intIndex] = fltVergrad;
    } }
"""

sepconv_ingrad = """
    extern "C" __global__ void __launch_bounds__(512) sepconv_ingrad(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenVer,
        const {{type}}* __restrict__ tenHor,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenIngrad,
        {{type}}* __restrict__ tenVergrad,
        {{type}}* __restrict__ tenHorgrad
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
        const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
        const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
        const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

        {{type}} fltIngrad = 0.0;

        {{type}} fltKahanc = 0.0;
        {{type}} fltKahany = 0.0;
        {{type}} fltKahant = 0.0;

        for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
            int intKy = intY + intFy - (SIZE_1(tenVer) - 1);

            if (intKy < 0) { continue; }
            if (intKy >= SIZE_2(tenVer)) { continue; }

            for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                int intKx = intX + intFx - (SIZE_1(tenHor) - 1);

                if (intKx < 0) { continue; }
                if (intKx >= SIZE_3(tenHor)) { continue; }

                fltKahany = VALUE_4(tenVer, intN, (SIZE_1(tenVer) - 1) - intFy, intKy, intKx) * VALUE_4(tenHor, intN, (SIZE_1(tenHor) - 1) - intFx, intKy, intKx) * VALUE_4(tenOutgrad, intN, intC, intKy, intKx);
                fltKahany = fltKahany - fltKahanc;
                fltKahant = fltIngrad + fltKahany;
                fltKahanc = (fltKahant - fltIngrad) - fltKahany;
                fltIngrad = fltKahant;
            }
        }

        tenIngrad[intIndex] = fltIngrad;
    } }
"""

sepconv_out = """
    extern "C" __global__ void __launch_bounds__(512) sepconv_out(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenVer,
        const {{type}}* __restrict__ tenHor,
        {{type}}* __restrict__ tenOut
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
        const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
        const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
        const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

        {{type}} fltOut = 0.0;

        {{type}} fltKahanc = 0.0;
        {{type}} fltKahany = 0.0;
        {{type}} fltKahant = 0.0;

        for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
            for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                fltKahany = VALUE_4(tenIn, intN, intC, intY + intFy, intX + intFx) * VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenHor, intN, intFx, intY, intX);
                fltKahany = fltKahany - fltKahanc;
                fltKahant = fltOut + fltKahany;
                fltKahanc = (fltKahant - fltOut) - fltKahany;
                fltOut = fltKahant;
            }
        }

        tenOut[intIndex] = fltOut;
    } }
"""

sepconv_horgrad = """
    extern "C" __global__ void __launch_bounds__(512) sepconv_horgrad(
        const int n,
        const {{type}}* __restrict__ tenIn,
        const {{type}}* __restrict__ tenVer,
        const {{type}}* __restrict__ tenHor,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenIngrad,
        {{type}}* __restrict__ tenVergrad,
        {{type}}* __restrict__ tenHorgrad
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad) / SIZE_1(tenHorgrad) ) % SIZE_0(tenHorgrad);
        const int intC = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad)                      ) % SIZE_1(tenHorgrad);
        const int intY = ( intIndex / SIZE_3(tenHorgrad)                                           ) % SIZE_2(tenHorgrad);
        const int intX = ( intIndex                                                                ) % SIZE_3(tenHorgrad);

        {{type}} fltHorgrad = 0.0;

        {{type}} fltKahanc = 0.0;
        {{type}} fltKahany = 0.0;
        {{type}} fltKahant = 0.0;

        for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
            for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                fltKahany = VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intFy, intX + intC) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                fltKahany = fltKahany - fltKahanc;
                fltKahant = fltHorgrad + fltKahany;
                fltKahanc = (fltKahant - fltHorgrad) - fltKahany;
                fltHorgrad = fltKahant;
            }
        }

        tenHorgrad[intIndex] = fltHorgrad;
    } }
"""

class sepconv_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenVer, tenHor):
        tenOut = tenIn.new_empty(
            [
                tenIn.shape[0],
                tenIn.shape[1],
                tenVer.shape[2] and tenHor.shape[2],
                tenVer.shape[3] and tenHor.shape[3],
            ]
        )

        if tenIn.is_cuda == True:
            cuda_launch(
                cuda_kernel(
                    "sepconv_out",
                    sepconv_out,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOut": tenOut,
                    },
                )
            )(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenOut.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOut.data_ptr(),
                ],
            )

        elif tenIn.is_cuda != True:
            assert False

        # end

        self.save_for_backward(tenIn, tenVer, tenHor)

        return tenOut

    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenVer, tenHor = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenIngrad = (
            tenIn.new_empty(
                [tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        tenVergrad = (
            tenVer.new_empty(
                [tenVer.shape[0], tenVer.shape[1], tenVer.shape[2], tenVer.shape[3]]
            )
            if self.needs_input_grad[1] == True
            else None
        )
        tenHorgrad = (
            tenHor.new_empty(
                [tenHor.shape[0], tenHor.shape[1], tenHor.shape[2], tenHor.shape[3]]
            )
            if self.needs_input_grad[2] == True
            else None
        )

        if tenIngrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_ingrad",
                    sepconv_ingrad,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenIngrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenIngrad.data_ptr(),
                    None,
                    None,
                ],
            )
        # end

        if tenVergrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_vergrad",
                    sepconv_vergrad,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenVergrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenVergrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    tenVergrad.data_ptr(),
                    None,
                ],
            )
        # end

        if tenHorgrad is not None:
            cuda_launch(
                cuda_kernel(
                    "sepconv_horgrad",
                    sepconv_horgrad,
                    {
                        "tenIn": tenIn,
                        "tenVer": tenVer,
                        "tenHor": tenHor,
                        "tenOutgrad": tenOutgrad,
                        "tenIngrad": tenIngrad,
                        "tenVergrad": tenVergrad,
                        "tenHorgrad": tenHorgrad,
                    },
                )
            )(
                grid=tuple([int((tenHorgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(tenHorgrad.nelement()),
                    tenIn.data_ptr(),
                    tenVer.data_ptr(),
                    tenHor.data_ptr(),
                    tenOutgrad.data_ptr(),
                    None,
                    None,
                    tenHorgrad.data_ptr(),
                ],
            )
        # end

        return tenIngrad, tenVergrad, tenHorgrad

    # end


# end
__all__ = ["sepconv_func"]