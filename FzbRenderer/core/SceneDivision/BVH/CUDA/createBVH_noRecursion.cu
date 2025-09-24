#include "./createBVH.cuh"
#include <random>
#include <glm/ext/matrix_transform.hpp>
//#include <curand_kernel.h>

#include <unordered_map>
#include <sstream>
#include <string>
#include <iostream>
#include <functional>

#ifndef CREATE_BVH_NO_RECURSION_CU
#define CREATE_BVH_NO_RECURSION_CU

struct FzbTriangleTempInfo_nr {
    float3 pos0;
    float3 pos1;
    float3 pos2;
    uint32_t nodeIndex;
    FzbAABB AABB;
    uint32_t triangleIndex;
    //uint32_t depth;
};

struct FzbBvhNodeTempInfo {
    uint32_t triangleNum;   //当达到最大深度时，triangleNum没用了，则用来存放node中triangleInfoArray的起始索引
    int divideAxis;    //-1表示当前node划分不开；当达到最大深度时，divideAxis没用了，则置0，让三角形原子累加，得到偏移
    float3 sumPos;
    uint leftTriangleNum[3];     //记录的是三角形数，那么左子树节点数为三角形数 * 2 - 1
    float SAHCost[6];
};

//-----------------------------------------------------------核函数------------------------------------------------
__global__ void initTriangle_nr(float* vertices, FzbTriangleTempInfo_nr* triangleTempInfoArray, FzbBvhNodeTriangleInfo* bvhTriangleInfoArray, uint32_t triangleNum) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= triangleNum) {
        return;
    }
    FzbBvhNodeTriangleInfo triangleInfo = bvhTriangleInfoArray[threadIndex];

    uint32_t vertexStrip = 3 + 3 * (triangleInfo.vertexFormat & 1) + 2 * ((triangleInfo.vertexFormat >> 1) & 1) + 3 * ((triangleInfo.vertexFormat >> 2) & 1);
    uint32_t vertexIndex = triangleInfo.indices0 * vertexStrip;
    float3 pos0 = make_float3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
    vertexIndex = triangleInfo.indices1 * vertexStrip;
    float3 pos1 = make_float3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
    vertexIndex = triangleInfo.indices2 * vertexStrip;
    float3 pos2 = make_float3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
    //printf("%f %f %f  %f %f %f  %f %f %f\n", pos0.x, pos0.y, pos0.z, pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z);

    //现在可能出现三角形完全重叠的情况，解决方法就是，每个三角形往逆法线方向移动一点点，偏移量随机
    //float3 edge1 = normalize(pos1 - pos0);
    //float3 edge2 = normalize(pos2 - pos0);
    //float3 normal = normalize(cross(edge1, edge2));
    //uint32_t randomSeed = threadIndex;
    //float offset = rand(randomSeed) * 0.0001f;
    FzbTriangleTempInfo_nr triangleTempInfo;
    triangleTempInfo.pos0 = pos0;// - offset * normal;
    triangleTempInfo.pos1 = pos1;// - offset * normal;
    triangleTempInfo.pos2 = pos2;// - offset * normal;

    triangleTempInfo.AABB.leftX = fminf(pos0.x, fminf(pos1.x, pos2.x));
    triangleTempInfo.AABB.rightX = fmaxf(pos0.x, fmaxf(pos1.x, pos2.x));
    triangleTempInfo.AABB.leftY = fminf(pos0.y, fminf(pos1.y, pos2.y));
    triangleTempInfo.AABB.rightY = fmaxf(pos0.y, fmaxf(pos1.y, pos2.y));
    triangleTempInfo.AABB.leftZ = fminf(pos0.z, fminf(pos1.z, pos2.z));
    triangleTempInfo.AABB.rightZ = fmaxf(pos0.z, fmaxf(pos1.z, pos2.z));
    //如果是平面片，则给定一个小的偏差
    //if (triangleTempInfo.AABB.leftX == triangleTempInfo.AABB.rightX) {
    //    triangleTempInfo.AABB.leftX -= 0.01f;
    //    triangleTempInfo.AABB.rightX += 0.01f;
    //}
    //if (triangleTempInfo.AABB.leftY == triangleTempInfo.AABB.rightY) {
    //    triangleTempInfo.AABB.leftY -= 0.01f;
    //    triangleTempInfo.AABB.rightY += 0.01f;
    //}
    //if (triangleTempInfo.AABB.leftZ == triangleTempInfo.AABB.rightZ) {
    //    triangleTempInfo.AABB.leftZ -= 0.01f;
    //    triangleTempInfo.AABB.rightZ += 0.01f;
    //}

    //bvhTriangleInfoArray[threadIndex].AABB = triangleTempInfo.AABB;

    //triangleTempInfo.depth = 0;
    triangleTempInfo.nodeIndex = 0;
    triangleTempInfo.triangleIndex = threadIndex;

    //放大pos，也就放大了三角形之间的差距，方便划分
    triangleTempInfo.pos0 *= 100.0f;
    triangleTempInfo.pos1 *= 100.0f;
    triangleTempInfo.pos2 *= 100.0f;

    triangleTempInfoArray[threadIndex] = triangleTempInfo;
}

template<bool isFirst>
__global__ void createNode(FzbBvhNode* bvhNodeArray, FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr, uint32_t* newTriangleIndices, uint32_t* newTriangleNum_ptr) {
    __shared__ uint32_t groupDivideTriangleNum;  //还要继续划分的三角形数量
    __shared__ uint32_t groupStartIndex;
    
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) groupDivideTriangleNum = 0;
    __syncthreads();

    uint32_t warpIndex = threadIndex / 32;
    uint32_t warpLane = threadIndex & 31;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) return;

    uint32_t triangleIndex = isFirst ? threadIndex : triangleIndices[threadIndex];
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[triangleIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;
    uint32_t nodeTriangleNum = bvhNodeTempInfoArray[nodeIndex].triangleNum;
    int nodeCanDivide = bvhNodeTempInfoArray[nodeIndex].divideAxis;

    uint32_t activeMask = __ballot_sync(0xffffffff, nodeTriangleNum != 1 && nodeCanDivide != -1);
    if (nodeTriangleNum == 1) {     //对于叶节点
        bvhNodeArray[nodeIndex].AABB = triangleTempInfo.AABB;
        bvhNodeArray[nodeIndex].rightNodeIndex = triangleTempInfo.triangleIndex;
    }
    //else if (nodeCanDivide == -1) {    //对于划分不开的节点
    //    bvhNodeArray[nodeIndex].triangleCount = nodeTriangleNum;
    //}
    //triangleTempInfoArray[triangleIndex].depth += 1;
    //if (nodeCanDivide == -1) printf("%d %d\n", triangleTempInfo.depth + 1, nodeIndex);
    //if (triangleTempInfo.triangleIndex == 3332) printf("%d %d\n", triangleTempInfo.depth + 1, nodeCanDivide);
    //if (nodeTriangleNum == 0) printf("%d\n", triangleTempInfo.depth);

    if (activeMask == 0) return;
    uint32_t laneOffset = __popc(activeMask & ((1u << warpLane) - 1));
    uint32_t firstActiveLane = __ffs(activeMask) - 1;
    /*
    * 这里我尽量将一个group、warp中的三角形放在一起，即在newTriangleNum_ptr中连续，这有什么好处呢
    * 好处就是一个group、warp中的三角形较为聚集，那么自然更有可能分到一个node中，那么后续的分支就可能更少了。
    */
    uint32_t warpStartIndex;
    if (warpLane == firstActiveLane) {
        uint32_t devideTriangleNum = __popc(activeMask);
        warpStartIndex = atomicAdd(&groupDivideTriangleNum, devideTriangleNum);
    }
    __syncthreads();

    if (threadIdx.x == 0) groupStartIndex = atomicAdd(newTriangleNum_ptr, groupDivideTriangleNum);
    __syncthreads();

    if (nodeTriangleNum != 1 && nodeCanDivide != -1) {
        warpStartIndex = __shfl_sync(activeMask, warpStartIndex, firstActiveLane);    //srcLane 必须在 mask，且只有activeMask中的线程能拿到结果
        newTriangleIndices[groupStartIndex + warpStartIndex + laneOffset] = triangleIndex;
        float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3;

        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.x, meanPos.x);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.y, meanPos.y);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.z, meanPos.z);

        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftX, triangleTempInfo.AABB.leftX);   //为当前节点创造AABB
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightX, triangleTempInfo.AABB.rightX);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftY, triangleTempInfo.AABB.leftY);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightY, triangleTempInfo.AABB.rightY);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftZ, triangleTempInfo.AABB.leftZ);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightZ, triangleTempInfo.AABB.rightZ);
    }
}
/*
template<bool isFirst>
__global__ void createNode(FzbBvhNode* bvhNodeArray, FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr, uint32_t* newTriangleIndices, uint32_t* newTriangleNum_ptr) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) {
        return;
    }

    uint32_t triangleIndex = isFirst ? threadIndex : triangleIndices[threadIndex];
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[triangleIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;

    //printf("%d %d\n", nodeIndex, bvhNodeTempInfoArray[nodeIndex].triangleNum);
    if (bvhNodeTempInfoArray[nodeIndex].triangleNum == 1) {
        bvhNodeArray[nodeIndex].AABB = triangleTempInfo.AABB;
        bvhNodeArray[nodeIndex].rightNodeIndex = triangleTempInfo.triangleIndex;
    }
    else {
        uint32_t newTriangleIndex = atomicAdd(newTriangleNum_ptr, 1);
        newTriangleIndices[newTriangleIndex] = triangleIndex;   //说明该三角形还需要继续划分

        float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3;
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.x, meanPos.x);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.y, meanPos.y);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].sumPos.z, meanPos.z);

        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftX, triangleTempInfo.AABB.leftX);   //为当前节点创造AABB
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightX, triangleTempInfo.AABB.rightX);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftY, triangleTempInfo.AABB.leftY);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightY, triangleTempInfo.AABB.rightY);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftZ, triangleTempInfo.AABB.leftZ);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightZ, triangleTempInfo.AABB.rightZ);
    }
}
*/

__global__ void preDivideNode(FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) {
        return;
    }

    uint32_t triangleIndex = triangleIndices[threadIndex];
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[triangleIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;
    FzbBvhNodeTempInfo nodeTempInfo = bvhNodeTempInfoArray[nodeIndex];

    float3 edge1 = triangleTempInfo.pos1 - triangleTempInfo.pos0;
    float3 edge2 = triangleTempInfo.pos2 - triangleTempInfo.pos0;
    float3 crossResult = cross(edge1, edge2);
    float triangleArea = length(crossResult);   //面积的平方的两倍

    float3 nodeMeanPos = nodeTempInfo.sumPos / nodeTempInfo.triangleNum;
    //if (nodeIndex == 1) {
    //    printf("%d  %d   %f %f %f   %f %f %f\n", triangleIndex, nodeTempInfo.triangleNum, nodeTempInfo.meanPos.x, nodeTempInfo.meanPos.y, nodeTempInfo.meanPos.z, nodeMeanPos.x, nodeMeanPos.y, nodeMeanPos.z);
    //}
    //bvhNodeTempInfoArray[nodeIndex].meanPos = nodeMeanPos;
    float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3;
    //printf("nodeMeanPos: %f %f %f meanPos: %f %f %f \n", nodeTempInfo.sumPos.x, nodeTempInfo.sumPos.y, nodeTempInfo.sumPos.z, meanPos.x, meanPos.y, meanPos.z);

    if (meanPos.x < nodeMeanPos.x) {
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].leftTriangleNum[0], 1);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[0], triangleArea);
    }
    else {
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[1], triangleArea);
    }

    if (meanPos.y < nodeMeanPos.y) {
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].leftTriangleNum[1], 1);
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[2], triangleArea);
    }
    else {
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[3], triangleArea);
    }

    if (meanPos.z < nodeMeanPos.z) {
       atomicAdd(&bvhNodeTempInfoArray[nodeIndex].leftTriangleNum[2], 1);
       atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[4], triangleArea);
    }
    else {
        atomicAdd(&bvhNodeTempInfoArray[nodeIndex].SAHCost[5], triangleArea);
    }
}

__global__ void divideNode(FzbBvhNode* bvhNodeArray, FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr, bool* canDivide) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) return;

    uint32_t triangleIndex = triangleIndices[threadIndex];
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[triangleIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;
    if (nodeIndex >= 38547) printf("%d\n", nodeIndex);
    FzbBvhNodeTempInfo nodeTempInfo = bvhNodeTempInfoArray[nodeIndex];

    float sahCost_X;
    if (nodeTempInfo.leftTriangleNum[0] == 0 || nodeTempInfo.leftTriangleNum[0] == nodeTempInfo.triangleNum) sahCost_X = FLT_MAX;
    else sahCost_X = nodeTempInfo.SAHCost[0] * nodeTempInfo.leftTriangleNum[0] + nodeTempInfo.SAHCost[1] * (nodeTempInfo.triangleNum - nodeTempInfo.leftTriangleNum[0]);
    float sahCost_Y;
    if (nodeTempInfo.leftTriangleNum[1] == 0 || nodeTempInfo.leftTriangleNum[1] == nodeTempInfo.triangleNum) sahCost_Y = FLT_MAX;
    else sahCost_Y = nodeTempInfo.SAHCost[2] * nodeTempInfo.leftTriangleNum[1] + nodeTempInfo.SAHCost[3] * (nodeTempInfo.triangleNum - nodeTempInfo.leftTriangleNum[1]);
    float sahCost_Z;
    if (nodeTempInfo.leftTriangleNum[2] == 0 || nodeTempInfo.leftTriangleNum[2] == nodeTempInfo.triangleNum) sahCost_Z = FLT_MAX;
    else sahCost_Z = nodeTempInfo.SAHCost[4] * nodeTempInfo.leftTriangleNum[2] + nodeTempInfo.SAHCost[5] * (nodeTempInfo.triangleNum - nodeTempInfo.leftTriangleNum[2]);

    //完全分不开
    if (sahCost_X == FLT_MAX && sahCost_Y == FLT_MAX && sahCost_Z == FLT_MAX) { 
        bvhNodeTempInfoArray[nodeIndex].divideAxis = -1;
        *canDivide = false;
        return;
    }
    uint32_t divideAxis = sahCost_X < sahCost_Y ? sahCost_X < sahCost_Z ? 0 : 2 : sahCost_Y < sahCost_Z ? 1 : 2;
    bvhNodeTempInfoArray[nodeIndex].divideAxis = divideAxis;

    float3 nodeMeanPos = nodeTempInfo.sumPos / nodeTempInfo.triangleNum;
    float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3;

    bool isLeft = false;
    if (divideAxis == 0) isLeft = meanPos.x < nodeMeanPos.x;
    else if (divideAxis == 1) isLeft = meanPos.y < nodeMeanPos.y;
    else isLeft = meanPos.z < nodeMeanPos.z;

    uint32_t fatherNodeIndex = nodeIndex;
    if (isLeft) {
        nodeIndex = fatherNodeIndex + 1;   //左节点是父节点数+1
        bvhNodeArray[fatherNodeIndex].leftNodeIndex = nodeIndex;
        bvhNodeTempInfoArray[nodeIndex].triangleNum = nodeTempInfo.leftTriangleNum[divideAxis];
    }
    else {
        nodeIndex = fatherNodeIndex + nodeTempInfo.leftTriangleNum[divideAxis] * 2;     //右节点是父节点左子树节点数+1
        bvhNodeArray[fatherNodeIndex].rightNodeIndex = nodeIndex;
        bvhNodeTempInfoArray[nodeIndex].triangleNum = nodeTempInfo.triangleNum - nodeTempInfo.leftTriangleNum[divideAxis];
    }
    //bvhNodeArray[nodeIndex].depth = bvhNodeArray[fatherNodeIndex].depth + 1;
    //if (triangleTempInfo.triangleIndex == 3332) printf("%f %f %f %f %f %f\n", meanPos.x, meanPos.y, meanPos.z, nodeMeanPos.x, nodeMeanPos.y, nodeMeanPos.z);
    //if (triangleTempInfo.triangleIndex == 3332) printf("%d %d\n", fatherNodeIndex, nodeIndex);
    //if (sahCost_X == FLT_MAX && sahCost_Y == FLT_MAX && sahCost_Z == FLT_MAX) {
    //    if (triangleTempInfo.triangleIndex == 3332) printf("%d %d\n", triangleTempInfo.depth, fatherNodeIndex);
    //    //if(meanPos.z == nodeMeanPos.z) printf("%d %d \n", triangleTempInfo.depth, triangleTempInfo.triangleIndex);
    //}
    //if (nodeIndex == 2) {
    //    float3 nodeMeanPos = nodeTempInfo.meanPos;
    //    //printf("划分后节点: %d 三角形meanPos: %f %f %f  nodeMeanPos: %f %f %f\n",
    //    //    nodeIndex, meanPos.x, meanPos.y, meanPos.z, nodeMeanPos.x, nodeMeanPos.y, nodeMeanPos.z);
    //    printf("三角系数: %d nodeMeanPos: %f %f %f\n", bvhNodeTempInfoArray[nodeIndex].triangleNum, nodeMeanPos.x, nodeMeanPos.y, nodeMeanPos.z);
    //}

    triangleTempInfoArray[triangleIndex].nodeIndex = nodeIndex;
}

__global__ void checkNodeTriangleNum(FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr, uint32_t* isPrintfs) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) {
        return;
    }
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[threadIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;
    FzbBvhNodeTempInfo nodeTempInfo = bvhNodeTempInfoArray[nodeIndex];

    uint32_t isPrintf = atomicAdd(&isPrintfs[nodeIndex], 1);
    if (isPrintf == 0) {
        printf("node索引为：%d   三角形数量为: %d\n", nodeIndex, nodeTempInfo.triangleNum);
    }
}

__global__ void nodeApplyTriangleInfoArraySpace(FzbBvhNode* bvhNodeArray, FzbBvhNodeTempInfo* bvhNodeTempInfoArray, uint32_t nodeCount, uint32_t* triangleOffset) {
    __shared__ uint32_t groupTriangleCount;
    __shared__ uint32_t groupTriangleOffset;
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIdx.x == 0) {
        groupTriangleCount = 0;
        groupTriangleOffset = 0;
    }
    __syncthreads();
    if (threadIndex >= nodeCount) return;

    uint32_t triangleNum = bvhNodeTempInfoArray[threadIndex].triangleNum;
    //if (triangleNum == 0) return;  //对于还没到达的节点，直接返回
    //if (nodeInfo.leftNodeIndex > 0) return;     //对于之前层的非叶节点，直接返回
    FzbBvhNode nodeInfo = bvhNodeArray[threadIndex];
    uint32_t groupStartIndex;
    if (triangleNum != 0 && nodeInfo.leftNodeIndex == 0) {  //对于叶节点、maxDepth的节点和划不开的节点
        nodeInfo.triangleCount = triangleNum;
        groupStartIndex = atomicAdd(&groupTriangleCount, triangleNum);
    }
    __syncthreads();
    if(threadIdx.x == 0) groupTriangleOffset = atomicAdd(triangleOffset, groupTriangleCount);
    __syncthreads();

    if (triangleNum != 0 && nodeInfo.leftNodeIndex == 0) {
        nodeInfo.rightNodeIndex = groupTriangleOffset + groupStartIndex;
        bvhNodeArray[threadIndex] = nodeInfo;
        bvhNodeTempInfoArray[threadIndex].leftTriangleNum[0] = nodeInfo.rightNodeIndex; //表示node的三角形的起始索引
        if (triangleNum > 1) {    //maxDepth的非叶节点或无法划分的节点
            bvhNodeTempInfoArray[threadIndex].divideAxis = 0;
        }
    }
}
__global__ void moveTriangleInfo(FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, FzbBvhNodeTriangleInfo* triangleInfoArray, FzbBvhNodeTriangleInfo* newTriangleInfoArray, uint32_t triangleNum) {
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= triangleNum) return;

    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[threadIndex];
    FzbBvhNodeTempInfo nodeTempInfo = bvhNodeTempInfoArray[triangleTempInfo.nodeIndex];
    uint32_t nodeTriangle = nodeTempInfo.triangleNum;
    uint32_t triangleStartIndex = nodeTempInfo.leftTriangleNum[0];
    if (nodeTriangle == 1) {    //叶节点
        newTriangleInfoArray[triangleStartIndex] = triangleInfoArray[triangleTempInfo.triangleIndex];
    }
    else {  //maxDepth的node和无法划分的node
        uint32_t offset = atomicAdd(&bvhNodeTempInfoArray[triangleTempInfo.nodeIndex].divideAxis, 1);
        newTriangleInfoArray[triangleStartIndex + offset] = triangleInfoArray[triangleTempInfo.triangleIndex];
    }
    //if (nodeTriangle > 1 && nodeTempInfo.test == -1) printf("%d %d\n", triangleTempInfo.depth, nodeTriangle);
}
//------------------------------------------------------------函数-------------------------------------------------
void BVHCuda::createBvhCuda_noRecursion(VkPhysicalDevice vkPhysicalDevice, FzbMainScene* scene, HANDLE bvhFinishedSemaphoreHandle, FzbBVHSetting setting) {
    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }
    extBVHSemaphore = importVulkanSemaphoreObjectFromNTHandle(bvhFinishedSemaphoreHandle);

    this->setting = setting;
    CHECK(cudaStreamCreate(&stream));
    cudaExternalMemory_t vertexExtMem = importVulkanMemoryObjectFromNTHandle(scene->vertexBuffer.handle, scene->vertexBuffer.size, false);
    this->vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, scene->vertexBuffer.size);
    //-------------------------------------------------初始化triangleInfoArray----------------------------------------------
    this->triangleNum = scene->indexBuffer.size / sizeof(uint32_t) / 3;
    createTriangleInfoArray(scene, stream);
    //-------------------------------------------初始化bvhNodeArray----------------------------------
    
    //如果每个叶节点只代表一个顶点，那么bvh数的节点总数固定，为2 * 叶节点数 - 1，因为没有度为1的节点
    //但是呢，必须要确定一个最大深度，这样RayTracing的时候才能搞一个数组用来碰撞检测
    uint32_t maxDepth = setting.maxDepth;
    uint32_t curDepth = 1;  //每次调用核函数就是划分一次，那么就是向下一次，所以整体记录一个即可

    //不能截取，因为现在右子树的节点是默认左子树中一个叶节点一个三角形来设置的，即右子树索引为父节点索引 + 左子树三角形数 * 2
    //uint32_t* nodeCount_device;    //当前叶节点树，总结点树为 2 * nodeCount - 1
    //CHECK(cudaMalloc((void**)&nodeCount_device, sizeof(uint32_t)));
    //CHECK(cudaMemset(nodeCount_device, 0, sizeof(uint32_t)));

    uint32_t bvhNodeNum = 2 * triangleNum - 1;
    CHECK(cudaMalloc((void**)&bvhNodeArray, sizeof(FzbBvhNode) * bvhNodeNum));
    uint32_t gridSize = ceil((float)bvhNodeNum / 1024);
    uint32_t blockSize = bvhNodeNum > 1024 ? 1024 : bvhNodeNum;
    initBvhNode << < gridSize, blockSize, 0, stream >> > (bvhNodeArray, bvhNodeNum);
    //------------------------------------------初始化bvhNodeTempInfoArray---------------------------
    FzbBvhNodeTempInfo* nodeTempInfoArray;
    CHECK(cudaMalloc((void**)&nodeTempInfoArray, sizeof(FzbBvhNodeTempInfo) * bvhNodeNum));
    CHECK(cudaMemset(nodeTempInfoArray, 0, sizeof(FzbBvhNodeTempInfo) * bvhNodeNum));
    CHECK(cudaMemcpy(&nodeTempInfoArray[0].triangleNum, &triangleNum, sizeof(uint32_t), cudaMemcpyHostToDevice));
    //------------------------------------------初始化triangleTempInfoArray-----------------------------
    FzbTriangleTempInfo_nr* triangleTempInfoArray;
    CHECK(cudaMalloc((void**)&triangleTempInfoArray, sizeof(FzbTriangleTempInfo_nr) * triangleNum));
    CHECK(cudaMemset(triangleTempInfoArray, 0, sizeof(FzbTriangleTempInfo_nr) * triangleNum));
    initTriangle_nr << < gridSize, blockSize, 0, stream >> > (vertices, triangleTempInfoArray, bvhTriangleInfoArray, triangleNum);
    //-------------------------------------------划分bvh-----------------------------------------
    uint32_t* triangleIndices0;
    CHECK(cudaMalloc((void**)&triangleIndices0, sizeof(uint32_t) * triangleNum));

    uint32_t* triangleIndices1;
    CHECK(cudaMalloc((void**)&triangleIndices1, sizeof(uint32_t) * triangleNum));

    uint32_t* triangleNum_ptr0;
    CHECK(cudaMalloc((void**)&triangleNum_ptr0, sizeof(uint32_t)));
    CHECK(cudaMemcpy(triangleNum_ptr0, &triangleNum, sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t* triangleNum_ptr1;
    CHECK(cudaMalloc((void**)&triangleNum_ptr1, sizeof(uint32_t)));
    CHECK(cudaMemset(triangleNum_ptr1, 0, sizeof(uint32_t)));

    bool* canDivide;
    CHECK(cudaMalloc((void**)&canDivide, sizeof(bool)));
    bool canDivide_host = true;
    CHECK(cudaMemcpy(canDivide, &canDivide_host, sizeof(bool), cudaMemcpyHostToDevice));

    gridSize = (triangleNum + 511) / 512;
    blockSize = triangleNum > 512 ? 512 : triangleNum;
    createNode<true> << <gridSize, blockSize, 0, stream >> > (bvhNodeArray, nodeTempInfoArray, triangleTempInfoArray, nullptr, triangleNum_ptr0, triangleIndices1, triangleNum_ptr1);
    uint32_t nonLeafTriangleNum;
    CHECK(cudaMemcpy(&nonLeafTriangleNum, triangleNum_ptr1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    //uint32_t* isPrintfs;
    //CHECK(cudaMalloc((void**)&isPrintfs, sizeof(uint32_t) * bvhNodeNum));
    //CHECK(cudaMemset(isPrintfs, 0, sizeof(uint32_t) * bvhNodeNum));
    //int time = 0;
    while (nonLeafTriangleNum > 0 && curDepth < maxDepth) {
        ++curDepth;
        gridSize = (nonLeafTriangleNum + 511) / 512;
        blockSize = nonLeafTriangleNum > 512 ? 512 : nonLeafTriangleNum;

        preDivideNode << <gridSize, blockSize, 0, stream >> > (nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1);
        divideNode << <gridSize, blockSize, 0, stream >> > (bvhNodeArray, nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1, canDivide);
        //checkNodeTriangleNum<<<gridSize, blockSize, 0, stream>>>(nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1, isPrintfs);
        //CHECK(cudaDeviceSynchronize());
        //CHECK(cudaMemset(isPrintfs, 0, sizeof(uint32_t) * bvhNodeNum));
        //std::cout << "\n\n";

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(triangleNum_ptr0, 0, sizeof(uint32_t)));
        uint32_t* temp = triangleNum_ptr0;
        triangleNum_ptr0 = triangleNum_ptr1;
        triangleNum_ptr1 = temp;

        temp = triangleIndices1;
        triangleIndices1 = triangleIndices0;
        triangleIndices0 = temp;

        createNode<false> << <gridSize, blockSize, 0, stream >> > (bvhNodeArray, nodeTempInfoArray, triangleTempInfoArray, triangleIndices0, triangleNum_ptr0, triangleIndices1, triangleNum_ptr1);
        CHECK(cudaMemcpy(&nonLeafTriangleNum, triangleNum_ptr1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        /*
        ++time;
        if (time == 20) {   //把三角形拿出来看看，到底是什么情况
            std::vector<uint32_t> nonLeafTriangles(nonLeafTriangleNum);
            CHECK(cudaMemcpy(nonLeafTriangles.data(), triangleIndices1, sizeof(uint32_t) * nonLeafTriangleNum, cudaMemcpyDeviceToHost));
            std::vector<FzbTriangleTempInfo_nr> triangleTempInfoArray_host(triangleNum);
            CHECK(cudaMemcpy(triangleTempInfoArray_host.data(), triangleTempInfoArray, sizeof(FzbTriangleTempInfo_nr) * triangleNum, cudaMemcpyDeviceToHost));
            std::unordered_map<uint32_t, std::vector<std::string>> nodeTriangles;
            for (int i = 0; i < nonLeafTriangleNum; ++i) {
                FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray_host[nonLeafTriangles[i]];

                std::ostringstream oss;
                oss << "pos0: " << triangleTempInfo.pos0.x << " "
                    << triangleTempInfo.pos0.y << " " << triangleTempInfo.pos0.z
                    << "  pos1: " << triangleTempInfo.pos1.x << " "
                    << triangleTempInfo.pos1.y << " " << triangleTempInfo.pos1.z
                    << "  pos2: " << triangleTempInfo.pos2.x << " "
                    << triangleTempInfo.pos2.y << " " << triangleTempInfo.pos2.z;

                float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3.0f;
                oss << "  meanPos: " << meanPos.x << " " << meanPos.y << " " << meanPos.z;
                std::string outStr = oss.str();

                nodeTriangles[triangleTempInfo.nodeIndex].push_back(outStr);
            }
            for (auto& pair : nodeTriangles) {
                std::vector<std::string>& strings = pair.second;
                std::cout << "对于节点：" << pair.first << " 其具有如下三角形: \n";
                for (int i = 0; i < strings.size(); ++i) {
                    std::cout << strings[i] << std::endl;
                }
                std::cout << "\n\n";
            }
        }
        */
    }
    /*
    bvh还有一点问题，就是我现在如果node划分不开，那么只有在nodeApplyTriangleInfoArraySpace中才能将其划分不开的triangle放在一起
    可是，如果最后到达不了maxDepth，则node的triangleCount只是1，且triangle的index不连续
    那么，我应该一定执行moveTriangleInfo，让他为每个node去申请newTriangleInfoArray，搬运数据
    */
    CHECK(cudaMemcpy(&canDivide_host, canDivide, sizeof(bool), cudaMemcpyDeviceToHost));
    //到达了最大深度或者存在划分不开的node，则也需要重新搬运，使得一个node中的triangle连续
    if ((nonLeafTriangleNum > 0 && curDepth == maxDepth) || !canDivide_host) {     
        //node申请newBvhTriangleInfoArray空间
        uint32_t* triangleOffset;
        CHECK(cudaMalloc((void**)&triangleOffset, sizeof(uint32_t)));
        CHECK(cudaMemset(triangleOffset, 0, sizeof(uint32_t)));
        gridSize = (bvhNodeNum + 511) / 512;
        blockSize = bvhNodeNum > 512 ? 512 : bvhNodeNum;
        nodeApplyTriangleInfoArraySpace<<<gridSize, blockSize, 0, stream>>>(bvhNodeArray, nodeTempInfoArray, bvhNodeNum, triangleOffset);
        CHECK(cudaFree(triangleOffset));

        //triangleInfo搬运
        FzbBvhNodeTriangleInfo* newBvhTriangleInfoArray;
        CHECK(cudaMalloc((void**)&newBvhTriangleInfoArray, sizeof(FzbBvhNodeTriangleInfo) * triangleNum));
        gridSize = (triangleNum + 128) / 128;
        blockSize = triangleNum > 128 ? 128 : triangleNum;
        moveTriangleInfo<<<gridSize, blockSize, 0, stream>>>(nodeTempInfoArray, triangleTempInfoArray, bvhTriangleInfoArray, newBvhTriangleInfoArray, triangleNum);
        CHECK(cudaFree(this->bvhTriangleInfoArray));
        this->bvhTriangleInfoArray = newBvhTriangleInfoArray;

        //截取nodeInfoArray
        //CHECK(cudaMemcpy(&this->nodeCount, nodeCount_device, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        //FzbBvhNode* newBvhNodeArray;
        //CHECK(cudaMalloc((void**)&newBvhNodeArray, sizeof(FzbBvhNode) * this->nodeCount));
        //CHECK(cudaMemcpy(newBvhNodeArray, this->bvhNodeArray, sizeof(FzbBvhNode) * this->nodeCount, cudaMemcpyDeviceToDevice));
        //CHECK(cudaFree(this->bvhNodeArray));
        //this->bvhNodeArray = newBvhNodeArray;
    }
    //std::vector<FzbBvhNode> node_host(bvhNodeNum);
    //CHECK(cudaMemcpy(node_host.data(), this->bvhNodeArray, sizeof(FzbBvhNode) * bvhNodeNum, cudaMemcpyDeviceToHost));
    //std::cout << node_host[38546].rightNodeIndex << std::endl;
    /*
    const uint32_t NIL = 0xffffffff;
    std::vector<uint8_t> color(bvhNodeNum, 0); // 0=white 1=gray 2=black
    uint32_t maxDepth_x = 0;
    std::function<bool(uint32_t, uint32_t)> dfs =
        [&](uint32_t idx, uint32_t depth) -> bool
    {
        if (idx >= node_host.size())          return true;   // 越界也算坏节点
        if (color[idx] == 1)              return false;// 回到灰色→有环
        if (color[idx] == 2)              return true; // 已安全

        color[idx] = 1;                    // 灰
        maxDepth_x = std::max(maxDepth_x, depth);

        const FzbBvhNode& n = node_host[idx];
        if (n.leftNodeIndex == 0) {        // 叶子
            color[idx] = 2;
            return true;
        }
        // 内部节点
        bool ok = dfs(n.leftNodeIndex, depth + 1) &&
            dfs(n.rightNodeIndex, depth + 1);
        color[idx] = 2;                    // 黑
        return ok;
    };
    bool ok = dfs(0, 0);
    std::cout << ok << std::endl;
    */

    signalExternalSemaphore(extBVHSemaphore, stream);

    CHECK(cudaFree(vertices));
    CHECK(cudaDestroyExternalMemory(vertexExtMem));
    CHECK(cudaDestroyExternalSemaphore(extBVHSemaphore));
    //CHECK(cudaFree(nodeCount_device));
    CHECK(cudaFree(triangleTempInfoArray));
    CHECK(cudaFree(nodeTempInfoArray));
    CHECK(cudaFree(triangleIndices0));
    CHECK(cudaFree(triangleIndices1));
    CHECK(cudaFree(triangleNum_ptr0));
    CHECK(cudaFree(triangleNum_ptr1));
    CHECK(cudaFree(canDivide));
}

#endif