import torch
from .utils import cuda_kernel, cuda_launch, cuda_int32
import math

kernel_AdaCoF_updateOutput = """
    extern "C" __global__ void kernel_AdaCoF_updateOutput(
        const int n,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int c         = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int i         = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int j         = ( intIndex                                                    ) % SIZE_3(output);

        for (int k = 0; k < F_SIZE; k += 1) {
        for (int l = 0; l < F_SIZE; l += 1) {
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        dblOutput += w * (
            VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A))*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A))*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)*(beta-(float)B)
            );
        }
        }

        output[intIndex] = dblOutput;
    } }
"""

kernel_AdaCoF_updateGradWeight = """
    extern "C" __global__ void kernel_AdaCoF_updateGradWeight(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* offset_i,
        const float* offset_j,
        float* gradWeight
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
        const int intDepth   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                      ) % SIZE_1(gradWeight);
        const int i          = ( intIndex / SIZE_3(gradWeight)                                           ) % SIZE_2(gradWeight);
        const int j          = ( intIndex                                                                ) % SIZE_3(gradWeight);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;
        
        floatOutput += delta * (
            VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A))*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A))*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)*(beta-(float)B)
            );
        }

        gradWeight[intIndex] = floatOutput;
    } }
"""

kernel_AdaCoF_updateGradAlpha = """
    extern "C" __global__ void kernel_AdaCoF_updateGradAlpha(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_i
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i) / SIZE_1(gradOffset_i) ) % SIZE_0(gradOffset_i);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i)                        ) % SIZE_1(gradOffset_i);
        const int i          = ( intIndex / SIZE_3(gradOffset_i)                                               ) % SIZE_2(gradOffset_i);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_i);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(1-(beta-(float)B)) - 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(beta-(float)B)
            );
        }

        gradOffset_i[intIndex] = floatOutput;
    } }
"""

kernel_AdaCoF_updateGradBeta = """
    extern "C" __global__ void kernel_AdaCoF_updateGradBeta(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_j
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j) / SIZE_1(gradOffset_j) ) % SIZE_0(gradOffset_j);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j)                        ) % SIZE_1(gradOffset_j);
        const int i          = ( intIndex / SIZE_3(gradOffset_j)                                               ) % SIZE_2(gradOffset_j);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_j);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A)) - 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)
            );
        }

        gradOffset_j[intIndex] = floatOutput;
    } }
"""

class FunctionAdaCoF(torch.autograd.Function):
    # end
    @staticmethod
    def forward(ctx, input, weight, offset_i, offset_j, dilation):
        ctx.save_for_backward(input, weight, offset_i, offset_j)
        ctx.dilation = dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (
            intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1
        )
        assert (
            intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1
        )

        assert input.is_contiguous() == True
        assert weight.is_contiguous() == True
        assert offset_i.is_contiguous() == True
        assert offset_j.is_contiguous() == True

        output = input.new_zeros(
            intSample, intInputDepth, intOutputHeight, intOutputWidth
        )

        if input.is_cuda == True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cuda_launch(
                cuda_kernel(
                    "kernel_AdaCoF_updateOutput",
                    kernel_AdaCoF_updateOutput,
                    {
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "output": output,
                    },
                    F_SIZE=str(intFilterSize),
                    DILATION=str(dilation)
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    output.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end
    @staticmethod
    def backward(ctx, gradOutput):
        input, weight, offset_i, offset_j = ctx.saved_tensors
        dilation = ctx.dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (
            intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1
        )
        assert (
            intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1
        )

        assert gradOutput.is_contiguous() == True

        gradInput = (
            input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth)
            if ctx.needs_input_grad[0] == True
            else None
        )
        gradWeight = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[1] == True
            else None
        )
        gradOffset_i = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[2] == True
            else None
        )
        gradOffset_j = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[2] == True
            else None
        )

        if input.is_cuda == True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n_w = gradWeight.nelement()
            cuda_launch(
                cuda_kernel(
                    "kernel_AdaCoF_updateGradWeight",
                    kernel_AdaCoF_updateGradWeight,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradWeight": gradWeight,
                    },
                    F_SIZE=str(intFilterSize),
                    DILATION=str(dilation)
                ),
            )(
                grid=tuple([int((n_w + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_w,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradWeight.data_ptr(),
                ],
                stream=Stream,
            )

            # alpha grad
            n_i = gradOffset_i.nelement()
            cuda_launch(
                cuda_kernel(
                    "kernel_AdaCoF_updateGradAlpha",
                    kernel_AdaCoF_updateGradAlpha,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradOffset_i": gradOffset_i,
                    },
                    F_SIZE=str(intFilterSize),
                    DILATION=str(dilation)
                ),
            )(
                grid=tuple([int((n_i + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_i,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradOffset_i.data_ptr(),
                ],
                stream=Stream,
            )

            # beta grad
            n_j = gradOffset_j.nelement()
            cuda_launch(
                cuda_kernel(
                    "kernel_AdaCoF_updateGradBeta",
                    kernel_AdaCoF_updateGradBeta,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradOffset_j": gradOffset_j,
                    },
                    F_SIZE=str(intFilterSize),
                    DILATION=str(dilation)
                ),
            )(
                grid=tuple([int((n_j + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_j,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradOffset_j.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradWeight, gradOffset_i, gradOffset_j, None

__all__ = ["FunctionAdaCoF"]
