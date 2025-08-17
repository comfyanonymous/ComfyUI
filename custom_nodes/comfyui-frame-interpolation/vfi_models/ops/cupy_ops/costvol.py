from .utils import cuda_kernel, cuda_launch, cuda_int32
import torch, collections

costvol_out = """
    extern "C" __global__ void __launch_bounds__(512) costvol_out(
        const int n,
        const {{type}}* __restrict__ tenOne,
        const {{type}}* __restrict__ tenTwo,
        {{type}}* __restrict__ tenOut
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) ) % SIZE_0(tenOut);
        const int intC = -1;
        const int intY = ( intIndex / SIZE_3(tenOut)                  ) % SIZE_2(tenOut);
        const int intX = ( intIndex                                   ) % SIZE_3(tenOut);

        {{type}} fltOne[{{intChans}}];

        for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
            fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
        }

        int intOffset = OFFSET_4(tenOut, intN, 0, intY, intX);

        for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
            for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                {{type}} fltValue = 0.0f;

                if ((intOy >= 0) && (intOy < SIZE_2(tenOut)) && (intOx >= 0) && (intOx < SIZE_3(tenOut))) {
                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        fltValue += abs(fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx));
                    }
                } else {
                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        fltValue += abs(fltOne[intValue]);
                    }
                }

                tenOut[intOffset] = fltValue / SIZE_1(tenOne);
                intOffset += SIZE_2(tenOut) * SIZE_3(tenOut);
            }
        }
    } }
"""

costvol_onegrad = """
    extern "C" __global__ void __launch_bounds__(512) costvol_onegrad(
        const int n,
        const {{type}}* __restrict__ tenOne,
        const {{type}}* __restrict__ tenTwo,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenOnegrad,
        {{type}}* __restrict__ tenTwograd
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenOnegrad) / SIZE_2(tenOnegrad) ) % SIZE_0(tenOnegrad);
        const int intC = -1;
        const int intY = ( intIndex / SIZE_3(tenOnegrad)                      ) % SIZE_2(tenOnegrad);
        const int intX = ( intIndex                                           ) % SIZE_3(tenOnegrad);

        {{type}} fltOne[{{intChans}}];

        for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
            fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
        }

        int intOffset = OFFSET_4(tenOutgrad, intN, 0, intY, intX);

        for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
            for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                if ((intOy >= 0) && (intOy < SIZE_2(tenOutgrad)) && (intOx >= 0) && (intOx < SIZE_3(tenOutgrad))) {
                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        if (fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx) >= 0.0f) {
                            tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += +tenOutgrad[intOffset] / SIZE_1(tenOne);
                        } else {
                            tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += -tenOutgrad[intOffset] / SIZE_1(tenOne);
                        }
                    }
                } else {
                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        if (fltOne[intValue] >= 0.0f) {
                            tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += +tenOutgrad[intOffset] / SIZE_1(tenOne);
                        } else {
                            tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += -tenOutgrad[intOffset] / SIZE_1(tenOne);
                        }
                    }
                }

                intOffset += SIZE_2(tenOutgrad) * SIZE_3(tenOutgrad);
            }
        }
    } }
"""

costvol_twograd = """
    extern "C" __global__ void __launch_bounds__(512) costvol_twograd(
        const int n,
        const {{type}}* __restrict__ tenOne,
        const {{type}}* __restrict__ tenTwo,
        const {{type}}* __restrict__ tenOutgrad,
        {{type}}* __restrict__ tenOnegrad,
        {{type}}* __restrict__ tenTwograd
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(tenTwograd) / SIZE_2(tenTwograd) ) % SIZE_0(tenTwograd);
        const int intC = -1;
        const int intY = ( intIndex / SIZE_3(tenTwograd)                      ) % SIZE_2(tenTwograd);
        const int intX = ( intIndex                                           ) % SIZE_3(tenTwograd);

        {{type}} fltOne[{{intChans}}];

        for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
            fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
        }

        int intOffset = OFFSET_4(tenOutgrad, intN, 0, intY, intX);

        for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
            for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                if ((intOy >= 0) && (intOy < SIZE_2(tenOutgrad)) && (intOx >= 0) && (intOx < SIZE_3(tenOutgrad))) {
                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        if (fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx) >= 0.0f) {
                            atomicAdd(&tenTwograd[OFFSET_4(tenTwograd, intN, intValue, intOy, intOx)], -tenOutgrad[intOffset] / SIZE_1(tenOne));
                        } else {
                            atomicAdd(&tenTwograd[OFFSET_4(tenTwograd, intN, intValue, intOy, intOx)], +tenOutgrad[intOffset] / SIZE_1(tenOne));
                        }
                    }
                } else {
                    // ...
                }

                intOffset += SIZE_2(tenOutgrad) * SIZE_3(tenOutgrad);
            }
        }
    } }          
"""

class costvol_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenOne, tenTwo):
        tenOut = tenOne.new_empty(
            [tenOne.shape[0], 81, tenOne.shape[2], tenOne.shape[3]]
        )

        cuda_launch(
            cuda_kernel(
                "costvol_out",
                costvol_out,
                {
                    "intChans": tenOne.shape[1],
                    "tenOne": tenOne,
                    "tenTwo": tenTwo,
                    "tenOut": tenOut,
                },
            )
        )(
            grid=tuple(
                [
                    int(
                        (
                            (tenOut.shape[0] * tenOut.shape[2] * tenOut.shape[3])
                            + 512
                            - 1
                        )
                        / 512
                    ),
                    1,
                    1,
                ]
            ),
            block=tuple([512, 1, 1]),
            args=[
                cuda_int32(tenOut.shape[0] * tenOut.shape[2] * tenOut.shape[3]),
                tenOne.data_ptr(),
                tenTwo.data_ptr(),
                tenOut.data_ptr(),
            ],
            stream=collections.namedtuple("Stream", "ptr")(
                torch.cuda.current_stream().cuda_stream
            ),
        )

        self.save_for_backward(tenOne, tenTwo)

        return tenOut

    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenOne, tenTwo = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenOnegrad = (
            tenOne.new_zeros(
                [tenOne.shape[0], tenOne.shape[1], tenOne.shape[2], tenOne.shape[3]]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        tenTwograd = (
            tenTwo.new_zeros(
                [tenTwo.shape[0], tenTwo.shape[1], tenTwo.shape[2], tenTwo.shape[3]]
            )
            if self.needs_input_grad[1] == True
            else None
        )

        if tenOnegrad is not None:
            cuda_launch(
                cuda_kernel(
                    "costvol_onegrad",
                    costvol_onegrad,
                    {
                        "intChans": tenOne.shape[1],
                        "tenOne": tenOne,
                        "tenTwo": tenTwo,
                        "tenOutgrad": tenOutgrad,
                        "tenOnegrad": tenOnegrad,
                        "tenTwograd": tenTwograd,
                    },
                )
            )(
                grid=tuple(
                    [
                        int(
                            (
                                (
                                    tenOnegrad.shape[0]
                                    * tenOnegrad.shape[2]
                                    * tenOnegrad.shape[3]
                                )
                                + 512
                                - 1
                            )
                            / 512
                        ),
                        1,
                        1,
                    ]
                ),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(
                        tenOnegrad.shape[0] * tenOnegrad.shape[2] * tenOnegrad.shape[3]
                    ),
                    tenOne.data_ptr(),
                    tenTwo.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenOnegrad.data_ptr(),
                    tenTwograd.data_ptr(),
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        if tenTwograd is not None:
            cuda_launch(
                cuda_kernel(
                    "costvol_twograd",
                    costvol_twograd,
                    {
                        "intChans": tenOne.shape[1],
                        "tenOne": tenOne,
                        "tenTwo": tenTwo,
                        "tenOutgrad": tenOutgrad,
                        "tenOnegrad": tenOnegrad,
                        "tenTwograd": tenTwograd,
                    },
                )
            )(
                grid=tuple(
                    [
                        int(
                            (
                                (
                                    tenTwograd.shape[0]
                                    * tenTwograd.shape[2]
                                    * tenTwograd.shape[3]
                                )
                                + 512
                                - 1
                            )
                            / 512
                        ),
                        1,
                        1,
                    ]
                ),
                block=tuple([512, 1, 1]),
                args=[
                    cuda_int32(
                        tenTwograd.shape[0] * tenTwograd.shape[2] * tenTwograd.shape[3]
                    ),
                    tenOne.data_ptr(),
                    tenTwo.data_ptr(),
                    tenOutgrad.data_ptr(),
                    tenOnegrad.data_ptr(),
                    tenTwograd.data_ptr(),
                ],
                stream=collections.namedtuple("Stream", "ptr")(
                    torch.cuda.current_stream().cuda_stream
                ),
            )
        # end

        return tenOnegrad, tenTwograd, None, None

    # end


# end

__all__ = ["costvol_func"]