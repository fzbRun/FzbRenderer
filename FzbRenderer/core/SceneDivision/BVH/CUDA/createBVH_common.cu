#include "./createBVH.cuh"
#include <helper_math.h>
#include <random>
#include <glm/ext/matrix_transform.hpp>

#ifndef CREATE_BVH_COMMON_CU
#define CREATE_BVH_COMMON_CU

//---------------------------------------------------------核函数---------------------------------------------------------
__global__ void createTriangleInfoArrayCUDA(FzbBvhNodeTriangleInfo* triangleInfoArray, uint32_t* sceneIndices, uint32_t indexOffset, 
    uint32_t indexNum, uint32_t triangleOffset, uint32_t vertexFormat) {
    uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;   //表示mesh的第几个索引
    if (threadIndex >= indexNum) {
        return;
    }
    uint32_t triangleIndex = threadIndex / 3;
    uint32_t indexIndex = threadIndex % 3;

    if (indexIndex == 0) {
        triangleInfoArray[triangleOffset + triangleIndex].vertexFormat = vertexFormat;
        triangleInfoArray[triangleOffset + triangleIndex].indices0 = sceneIndices[indexOffset + threadIndex];
    }
    else if (indexIndex == 1) {
        triangleInfoArray[triangleOffset + triangleIndex].indices1 = sceneIndices[indexOffset + threadIndex];
    }
    else {
        triangleInfoArray[triangleOffset + triangleIndex].indices2 = sceneIndices[indexOffset + threadIndex];
    }
}

__global__ void initBvhNode(FzbBvhNode* bvhNodeArray, uint32_t nodeNum) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= nodeNum) {
        return;
    }
    FzbBvhNode node;
    node.leftNodeIndex = 0;
    node.rightNodeIndex = 0;
    node.AABB.leftX = FLT_MAX;
    node.AABB.rightX = -FLT_MAX;
    node.AABB.leftY = FLT_MAX;
    node.AABB.rightY = -FLT_MAX;
    node.AABB.leftZ = FLT_MAX;
    node.AABB.rightZ = -FLT_MAX;
    bvhNodeArray[threadIndex] = node;
}
//-------------------------------------------------------函数---------------------------------------------------------------
void BVHCuda::createTriangleInfoArray(FzbMainScene* scene, cudaStream_t stream) {
    cudaExternalMemory_t indexExtMem = importVulkanMemoryObjectFromNTHandle(scene->indexBuffer.handle, scene->indexBuffer.size, false);
    uint32_t* sceneIndices = (uint32_t*)mapBufferOntoExternalMemory(indexExtMem, 0, scene->indexBuffer.size);

    this->triangleNum = scene->indexBuffer.size / sizeof(uint32_t) / 3;
    CHECK(cudaMalloc((void**)&bvhTriangleInfoArray, sizeof(FzbBvhNodeTriangleInfo) * triangleNum));

    uint32_t triangleOffset = 0;
    /*
    for (int i = 0; i < scene->sceneMeshIndices.size(); i++) {
        FzbMesh& mesh = scene->sceneMeshSet[scene->sceneMeshIndices[i]];

        FzbVertexFormat vertexFormat = mesh.vertexFormat;
        uint32_t vertexFormatU = (vertexFormat.useTangent << 2) | (vertexFormat.useTexCoord << 1) | (vertexFormat.useNormal);
        uint32_t meshIndexOffset = mesh.indexArrayOffset;
        uint32_t indexNum = mesh.indexArraySize;

        uint32_t gridSize = ceil((float)mesh.indexArraySize / 1024);
        uint32_t blockSize = mesh.indexArraySize > 1024 ? 1024 : mesh.indexArraySize;
        createTriangleInfoArrayCUDA << < gridSize, blockSize, 0, stream >> > (bvhTriangleInfoArray, sceneIndices, meshIndexOffset, indexNum, triangleOffset, vertexFormatU);

        triangleOffset += indexNum / 3;
    }
    */
    for (auto& meshIndices : scene->differentVertexFormatMeshIndexs) {
        FzbVertexFormat vertexFormat = meshIndices.first;
        uint32_t vertexFormatU = (vertexFormat.useTangent << 2) | (vertexFormat.useTexCoord << 1) | (vertexFormat.useNormal);
        uint32_t indexNum = 0;
        uint32_t indexOffset = scene->sceneMeshSet[meshIndices.second[0]].indexArrayOffset;
        for (int i = 0; i < meshIndices.second.size(); ++i) {
            FzbMesh& mesh = scene->sceneMeshSet[meshIndices.second[i]];
            indexNum += mesh.indexArraySize;
        }
        uint32_t gridSize = ceil((float)indexNum / 1024);
        uint32_t blockSize = indexNum > 1024 ? 1024 : indexNum;
        createTriangleInfoArrayCUDA << < gridSize, blockSize, 0, stream >> > (bvhTriangleInfoArray, sceneIndices, indexOffset, indexNum, triangleOffset, vertexFormatU);

        triangleOffset += indexNum / 3;
    }
    CHECK(cudaDestroyExternalMemory(indexExtMem));
}

void BVHCuda::getBvhCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE bvhNodeArrayHandle, HANDLE bvhTriangleInfoArrayHandle) {
    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }

    bvhNodeArrayExtMem = importVulkanMemoryObjectFromNTHandle(bvhNodeArrayHandle, sizeof(FzbBvhNode) * (2 * triangleNum - 1), false);
    FzbBvhNode* vkBvhNodeArray = (FzbBvhNode*)mapBufferOntoExternalMemory(bvhNodeArrayExtMem, 0, sizeof(FzbBvhNode) * (2 * triangleNum - 1));
    CHECK(cudaMemcpy(vkBvhNodeArray, this->bvhNodeArray, sizeof(FzbBvhNode) * (2 * triangleNum - 1), cudaMemcpyDeviceToDevice));

    bvhTriangleInfoArrayMem = importVulkanMemoryObjectFromNTHandle(bvhTriangleInfoArrayHandle, sizeof(FzbBvhNodeTriangleInfo) * triangleNum, false);
    FzbBvhNodeTriangleInfo* vkBvhTriangleInfoArray = (FzbBvhNodeTriangleInfo*)mapBufferOntoExternalMemory(bvhTriangleInfoArrayMem, 0, sizeof(FzbBvhNodeTriangleInfo) * triangleNum);
    CHECK(cudaMemcpy(vkBvhTriangleInfoArray, this->bvhTriangleInfoArray, sizeof(FzbBvhNodeTriangleInfo) * triangleNum, cudaMemcpyDeviceToDevice));
}

void BVHCuda::clean() {
    if(bvhNodeArrayExtMem) CHECK(cudaDestroyExternalMemory(bvhNodeArrayExtMem));
    if(bvhTriangleInfoArrayMem) CHECK(cudaDestroyExternalMemory(bvhTriangleInfoArrayMem));

    if(bvhNodeArray) CHECK(cudaFree(bvhNodeArray));
    if(bvhTriangleInfoArray) CHECK(cudaFree(bvhTriangleInfoArray));

    if (stream) CHECK(cudaStreamDestroy(stream));
}

#endif