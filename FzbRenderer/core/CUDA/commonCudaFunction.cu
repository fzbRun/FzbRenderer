#pragma once

#include "commonCudaFunction.cuh"

#ifndef COMMON_CUDA_FUNCTION_CU
#define COMMON_CUDA_FUNCTION_CU

double cpuSecond() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

//------------------------------------------warp内部操作-----------------------------------------------
/*
这里是关于warp内部操作（原语）的学习
首先，明确几点重要的点
1. __activemask()是指执行到此刻的线程的掩码。但是由于很多时候warp是半个半个执行的（硬件区域一般是16个硬件一组，要两个时钟周期执行完一个warp），可能导致
    即使warp内线程在一个分支中，但是不是同时执行到__activemask()的，就可能出现前半个warp只能得到0-15为1，16-31为0，而后半个warp得到全是1的情况。所以不能
    直接用于原语的参数。一般来说需要在分支前使用__ballot_sync()来得到掩码。
2. 大部分原语需要掩码显示执行线程，但是volta架构后可以不需要管，直接用0xffffffff即可，原语内部会处理，对于不激活的线程，原语结果会是未定义（但资料说就是0）
    这样，在分支中，如果想对激活线程进行累加，可以直接用0xffffffff。但是对于大小值判断则不行，因为不激活线程会返回0
3. 同样在volta架构之后，__syncwarp()可以处理条件分支，只不过在原语之后线程会再次发散。
4. 由于编译环境和硬件的不同，隐式同步是不安全的，比如在if-else后分支同步，但可能由于环境不同，同步没有发生，即使在前后使用__syncwarp()，如
     __syncwarp(); v = __shfl(0); __syncwarp() != __shfl(0)
    因此我们需要使用显式同步，即 使用带sync的原语，如__shfl_sync。
*/

// 假设 value 是每个线程的输入，返回该 warp 内的最大值
__device__ int warpMax(int value) {
    //// 全活跃线程掩码（32 线程 warp 中通常都是活跃的）
    //unsigned mask = __activemask();
    //// 每次向下偏移 offset 并与自己的值做比较
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value > other ? value : other);
    //}
    //// 到这里，lane 0 上的 value 就是整个 warp 的最大值
    //return value;
    return __reduce_max_sync(0xffffffff, value);
}
__device__ int warpMin(int value) {
    //unsigned mask = __activemask();
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value < other ? value : other);
    //}
    //return value;
    return __reduce_min_sync(0xffffffff, value);
}

__device__ float warpMax(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}
__device__ float warpMin(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = fminf(value, other);
    }
    return value;
}
__device__ int warpReduce(int localSum)
{
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 16);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 8);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 4);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 2);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 1);
    //return localSum;
    return __reduce_add_sync(0xffffffff, localSum);
}
__device__ float warpReduce(float localSum)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    return localSum;
}
//------------------------------------------------求和--------------------------------------------
__global__ void sumFloatKernel(float* input, float* output, uint32_t dataNum) {
    extern __shared__ float sharedData[];
    int threadIndex = threadIdx.x;

    sharedData[threadIndex] = 0;
    int dataIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (dataIndex >= dataNum) {
        return;
    }
    float inputData = input[dataIndex];
    if (dataIndex + blockDim.x < dataNum) {
        inputData += input[dataIndex + blockDim.x];
    }
    sharedData[threadIndex] = inputData;
    __syncthreads();

    if (dataNum > 1024) {
        if (threadIndex >= blockDim.x / 2) {
            return;
        }
    }
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIndex < s) {
            sharedData[threadIndex] += sharedData[threadIndex + s];
        }
        __syncthreads();
    }
    if (threadIndex < 32) { //这里只是针对4MB，最后一个块有1024个线程，如果是其他数量，需要额外判断
        float sum = warpReduce(sharedData[threadIndex]);
        if (threadIndex == 0) {
            output[blockIdx.x] = sum;
        }
    }

}
float* sumFloat(float* data, uint32_t dataNum, cudaStream_t stream) {
    uint32_t gridSize = ceil((float)dataNum / 2048);
    //gridSize = ceil((float)gridSize / 2);
    uint32_t blockSize = dataNum > 1024 ? 1024 : ceil(float(dataNum) / 2);

    float* input;
    CHECK(cudaMalloc((void**)&input, sizeof(float) * dataNum));
    CHECK(cudaMemcpy(input, data, sizeof(float) * dataNum, cudaMemcpyDeviceToDevice));
    float* output;
    CHECK(cudaMalloc((void**)&output, sizeof(float) * gridSize));
    CHECK(cudaDeviceSynchronize());

    while (dataNum > 1) {
        sumFloatKernel << <gridSize, blockSize, blockSize * sizeof(float), stream >> > (input, output, dataNum);
        CHECK(cudaStreamSynchronize(stream));

        //float* testData_host = (float*)malloc(sizeof(float) * gridSize);
        //CHECK(cudaMemcpy(testData_host, output, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
        //for (int i = 0; i < gridSize; i++) {
        //    std::cout << testData_host[i] << std::endl;
        //}
        //std::cout << std::endl;
        //free(testData_host);

        dataNum = gridSize;
        blockSize = dataNum > 1024 ? 1024 : ceil(float(dataNum) / 2);
        gridSize = ceil((float)gridSize / 2048);
        //gridSize = ceil((float)gridSize / 2);
        float* temp = input;
        input = output;
        output = temp;
    }

    float* sum;
    CHECK(cudaMalloc((void**)&sum, sizeof(float)));
    CHECK(cudaMemcpy(sum, input, sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK(cudaFree(input));
    CHECK(cudaFree(output));

    return sum;
}

//------------------------------------------打包操作-----------------------------------------------
__device__ uint32_t packUint3(uint3 valueU3) {
    return ((valueU3.x & 0x3FF) << 20) | ((valueU3.y & 0x3FF) << 10) | (valueU3.z & 0x3FF);
}

__device__ uint3 unpackUint(uint32_t value) {
    return make_uint3((value >> 20) & 0x3FF, (value >> 10) & 0x3FF, value & 0x3FF);
}
//------------------------------------------原子操作-----------------------------------------------
__device__ float atomicAddFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); // __float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        float tmp = old_f + val;
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

__device__ float atomicMinFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // 选择老值和 val 里的最小者
        float tmp = (old_f < val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

__device__ float atomicMaxFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // 选择老值和 val 里的最大者
        float tmp = (old_f > val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

//----------------------------------------随机数------------------------------------------------
__device__ uint32_t pcg(uint32_t& state)
{
    uint32_t prev = state * 747796405u + 2891336453u;
    uint32_t word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

__device__ uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    //v = v ^ (v >> 16u);
    uint2 t;
    t.x = v.x >> 16u;
    t.y = v.y >> 16u;
    v.x = v.x ^ t.x;
    v.y = v.y ^ t.y;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    //v = v ^ (v >> 16u);
    t.x = v.x >> 16u;
    t.y = v.y >> 16u;
    v.x = v.x ^ t.x;
    v.y = v.y ^ t.y;

    return v;
}

__device__ float rand(uint32_t& seed)
{
    uint32_t val = pcg(seed);
    return (float(val) * (1.0 / float(0xffffffffu)));
}


//低差异序列
__device__ float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
__device__ float2 Hammersley(uint i, uint N)
{
    return make_float2(float(i) / float(N), RadicalInverse_VdC(i));
}

#endif