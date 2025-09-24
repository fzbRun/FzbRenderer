#pragma once

#include "commonCudaFunction.cuh"

#ifndef COMMON_CUDA_FUNCTION_CU
#define COMMON_CUDA_FUNCTION_CU

double cpuSecond() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void checkKernelFunction() {
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("�˺�������ʧ��: %s\n", cudaGetErrorString(launchErr));
        return; // ���ʵ��Ĵ�����
    }
    else {
        printf("�˺����ѳɹ�����\n");
    }

    // Ȼ��ȴ��˺�����ɲ����ִ�д���
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("�˺���ִ�д���: %s\n", cudaGetErrorString(syncErr));
    }
    else {
        printf("�˺����ѳɹ�ִ�����\n");
    }
}

//------------------------------------------warp�ڲ�����----------------------------------------------

// ���� value ��ÿ���̵߳����룬���ظ� warp �ڵ����ֵ
__device__ int warpMax(int value, uint32_t mask) {
    //// ȫ��Ծ�߳����루32 �߳� warp ��ͨ�����ǻ�Ծ�ģ�
    //unsigned mask = __activemask();
    //// ÿ������ƫ�� offset �����Լ���ֵ���Ƚ�
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value > other ? value : other);
    //}
    //// �����lane 0 �ϵ� value �������� warp �����ֵ
    //return value;
    return __reduce_max_sync(mask, value);
}
__device__ int warpMin(int value, uint32_t mask) {
    //unsigned mask = __activemask();
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value < other ? value : other);
    //}
    //return value;
    return __reduce_min_sync(mask, value);
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
__device__ int warpReduce(int localSum, uint32_t mask)
{
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 16);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 8);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 4);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 2);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 1);
    //return localSum;
    return __reduce_add_sync(mask, localSum);
}
__device__ float warpReduce(float localSum, uint32_t mask)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(mask, localSum, offset);
    }
    return localSum;
}

__device__ bool valueEqual(int val, uint32_t mask) {
    int ref = __shfl_sync(mask, val, __ffs(mask) - 1);
    unsigned vote = __ballot_sync(mask, val == ref); // ��Щ lane ���� ref
    return vote == mask; // ������в��� lane ������ ref
}
//------------------------------------------------���--------------------------------------------
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
    if (threadIndex < 32) { //����ֻ�����4MB�����һ������1024���̣߳������������������Ҫ�����ж�
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

//------------------------------------------�������-----------------------------------------------
__device__ uint32_t packUint3(uint3 valueU3) {
    return ((valueU3.x & 0x3FF) << 20) | ((valueU3.y & 0x3FF) << 10) | (valueU3.z & 0x3FF);
}

__device__ uint3 unpackUint(uint32_t value) {
    return make_uint3((value >> 20) & 0x3FF, (value >> 10) & 0x3FF, value & 0x3FF);
}

__device__ __forceinline__ uint32_t packUnorm4x8(const float4 v) {
    // clamp to [0,1], then map to 0..255 and round
    float4 c;
    c.x = fminf(fmaxf(v.x, 0.0f), 1.0f);
    c.y = fminf(fmaxf(v.y, 0.0f), 1.0f);
    c.z = fminf(fmaxf(v.z, 0.0f), 1.0f);
    c.w = fminf(fmaxf(v.w, 0.0f), 1.0f);

    uint32_t r = (uint32_t)(c.x * 255.0f + 0.5f) & 0xFFu;
    uint32_t g = (uint32_t)(c.y * 255.0f + 0.5f) & 0xFFu;
    uint32_t b = (uint32_t)(c.z * 255.0f + 0.5f) & 0xFFu;
    uint32_t a = (uint32_t)(c.w * 255.0f + 0.5f) & 0xFFu;

    // pack: lowest byte = R, then G, B, highest = A
    return (a << 24) | (b << 16) | (g << 8) | r;
}
__device__ __forceinline__ float4 unpackUnorm4x8(const uint32_t v) {
    uint32_t r = v & 0xFFu;
    uint32_t g = (v >> 8) & 0xFFu;
    uint32_t b = (v >> 16) & 0xFFu;
    uint32_t a = (v >> 24) & 0xFFu;
    float4 out;
    out.x = (float)r / 255.0f;
    out.y = (float)g / 255.0f;
    out.z = (float)b / 255.0f;
    out.w = (float)a / 255.0f;
    return out;
}
//------------------------------------------ԭ�Ӳ���-----------------------------------------------
__device__ float atomicAddFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); // __float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        float tmp = old_f + val;
        new_i = __float_as_int(tmp);
        // atomicCAS ���ص��� *addr ִ��֮ǰ�ľ�λģʽ
        // �������� old_i��ʧ�ܾ���������һ�εıȽϻ�׼
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // ���ߣ� while (old_i != new_i)��������������

    return old_f;
}

__device__ float atomicMinFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // ѡ����ֵ�� val �����С��
        float tmp = (old_f < val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS ���ص��� *addr ִ��֮ǰ�ľ�λģʽ
        // �������� old_i��ʧ�ܾ���������һ�εıȽϻ�׼
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // ���ߣ� while (old_i != new_i)��������������

    return old_f;
}

__device__ float atomicMaxFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // ѡ����ֵ�� val ��������
        float tmp = (old_f > val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS ���ص��� *addr ִ��֮ǰ�ľ�λģʽ
        // �������� old_i��ʧ�ܾ���������һ�εıȽϻ�׼
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // ���ߣ� while (old_i != new_i)��������������

    return old_f;
}

__device__ void atomicMeanFloat4(uint32_t* addr, float4 val) {
    uint32_t newVal = packUnorm4x8(val);
    uint32_t curStoredVal = *addr;
    uint32_t prevStoredVal = curStoredVal;
    while (true) {
        uint32_t old = atomicCAS(addr, prevStoredVal, newVal);
        if (old == prevStoredVal) break;
        prevStoredVal = old;
        float4 rval = unpackUnorm4x8(prevStoredVal);

        rval.x = rval.x * rval.w;
        rval.y = rval.y * rval.w;
        rval.z = rval.z * rval.w;

        float4 curValF;
        curValF.x = rval.x + val.x;
        curValF.y = rval.y + val.y;
        curValF.z = rval.z + val.z;
        curValF.w = rval.w + val.w;

        if (curValF.w > 0.0f) {
            curValF.x /= curValF.w;
            curValF.y /= curValF.w;
            curValF.z /= curValF.w;
        }
        else {
            curValF.x = 0.0f;
            curValF.y = 0.0f;
            curValF.z = 0.0f;
        }
        newVal = packUnorm4x8(curValF);
    }
}


//----------------------------------------�����------------------------------------------------
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


//�Ͳ�������
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

__global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}
//__device__ float getCudaRandomNumber() {
//    uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
//    return curand_uniform(&systemRandomNumberStates[threadIndex]);    //0-1
//}
__device__ float getRandomNumber(uint32_t& randomNumberSeed) {
    if (useCudaRandom) {
        uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
        return curand_uniform(&systemRandomNumberStates[threadIndex]);    //0-1
    }
    else return rand(randomNumberSeed);
}

//---------------------------------------------------------------------------------------

__global__ void addDate_float_device(float* data, float date, uint32_t dataNum) {
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= dataNum) return;
    data[threadIndex] += date;
}
void addDateCUDA_float(float* data, float date, uint32_t dataNum) {
    float* data_device;
    CHECK(cudaMalloc((void**)&data_device, sizeof(float) * dataNum));
    CHECK(cudaMemcpy(data_device, data, sizeof(float) * dataNum, cudaMemcpyHostToDevice));
    uint32_t gridSize = (dataNum + 511) / 512;
    uint32_t blockSize = dataNum > 512 ? 512 : dataNum;
    addDate_float_device << < gridSize, blockSize >> > (data_device, date, dataNum);
    CHECK(cudaMemcpy(data, data_device, sizeof(float) * dataNum, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(data_device));
}

__global__ void addDate_uint_device(uint32_t* data, uint32_t date, uint32_t dataNum) {
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= dataNum) return;
    data[threadIndex] += date;
}
void addDateCUDA_uint(uint32_t* data, uint32_t date, uint32_t dataNum) {
    uint32_t* data_device;
    CHECK(cudaMalloc((void**)&data_device, sizeof(uint32_t) * dataNum));
    CHECK(cudaMemcpy(data_device, data, sizeof(uint32_t) * dataNum, cudaMemcpyHostToDevice));
    uint32_t gridSize = (dataNum + 511) / 512;
    uint32_t blockSize = dataNum > 512 ? 512 : dataNum;
    addDate_uint_device << < gridSize, blockSize >> > (data_device, date, dataNum);
    CHECK(cudaMemcpy(data, data_device, sizeof(uint32_t) * dataNum, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(data_device));
}

#endif