#include "./createBVH.cuh"
#include <random>
#include <glm/ext/matrix_transform.hpp>
//#include <curand_kernel.h>

#include <unordered_map>
#include <sstream>
#include <string>
#include <iostream>

#ifndef CREATE_BVH_NO_RECURSION_CU
#define CREATE_BVH_NO_RECURSION_CU

struct FzbTriangleTempInfo_nr {
    float3 pos0;
    float3 pos1;
    float3 pos2;
    uint32_t nodeIndex;
    FzbAABB_BVH AABB;
    uint32_t triangleIndex;
};

struct FzbBvhNodeTempInfo {
    uint32_t triangleNum;
    uint32_t divideAxis;
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
    if (triangleTempInfo.AABB.leftX == triangleTempInfo.AABB.rightX) {
        triangleTempInfo.AABB.leftX -= 0.01f;
        triangleTempInfo.AABB.rightX += 0.01f;
    }
    if (triangleTempInfo.AABB.leftY == triangleTempInfo.AABB.rightY) {
        triangleTempInfo.AABB.leftY -= 0.01f;
        triangleTempInfo.AABB.rightY += 0.01f;
    }
    if (triangleTempInfo.AABB.leftZ == triangleTempInfo.AABB.rightZ) {
        triangleTempInfo.AABB.leftZ -= 0.01f;
        triangleTempInfo.AABB.rightZ += 0.01f;
    }

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

    uint32_t activeMask = __ballot_sync(0xffffffff, nodeTriangleNum != 1);
    uint32_t laneOffset = __popc(activeMask & ((1u << warpLane) - 1));
    uint32_t firstActiveLane = __ffs(activeMask) - 1;

    if (nodeTriangleNum == 1) {
        bvhNodeArray[nodeIndex].AABB = triangleTempInfo.AABB;
        bvhNodeArray[nodeIndex].rightNodeIndex = triangleTempInfo.triangleIndex;
    }

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

    if (nodeTriangleNum != 1) {
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

__global__ void divideNode(FzbBvhNode* bvhNodeArray, FzbBvhNodeTempInfo* bvhNodeTempInfoArray, FzbTriangleTempInfo_nr* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t* triangleNum_ptr) {
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t triangleNum = *triangleNum_ptr;
    if (threadIndex >= triangleNum) {
        return;
    }

    uint32_t triangleIndex = triangleIndices[threadIndex];
    FzbTriangleTempInfo_nr triangleTempInfo = triangleTempInfoArray[triangleIndex];
    uint32_t nodeIndex = triangleTempInfo.nodeIndex;
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

    uint32_t divideAxis = sahCost_X < sahCost_Y ? sahCost_X < sahCost_Z ? 0 : 2 : sahCost_Y < sahCost_Z ? 1 : 2;
    bvhNodeTempInfoArray[nodeIndex].divideAxis = divideAxis;

    float3 meanPos = (triangleTempInfo.pos0 + triangleTempInfo.pos1 + triangleTempInfo.pos2) / 3;
    float3 nodeMeanPos = nodeTempInfo.sumPos / nodeTempInfo.triangleNum;
    bool isLeft = false;
    if (divideAxis == 0) {
        isLeft = meanPos.x < nodeMeanPos.x;
    }
    else if (divideAxis == 1) {
        isLeft = meanPos.y < nodeMeanPos.y;
    }
    else {
        isLeft = meanPos.z < nodeMeanPos.z;
    }

    uint32_t fatherNodeIndex = nodeIndex;
    if (isLeft) {
        bvhNodeArray[nodeIndex].leftNodeIndex = nodeIndex + 1;
        nodeIndex = nodeIndex + 1;   //左节点是父节点数+1
        bvhNodeTempInfoArray[nodeIndex].triangleNum = nodeTempInfo.leftTriangleNum[divideAxis];
    }
    else {
        bvhNodeArray[nodeIndex].rightNodeIndex = nodeIndex + nodeTempInfo.leftTriangleNum[divideAxis] * 2;
        nodeIndex = nodeIndex + nodeTempInfo.leftTriangleNum[divideAxis] * 2;     //右节点是父节点左子树节点数+1
        bvhNodeTempInfoArray[nodeIndex].triangleNum = nodeTempInfo.triangleNum - nodeTempInfo.leftTriangleNum[divideAxis];
    }
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
//------------------------------------------------------------函数-------------------------------------------------
void BVHCuda::createBvhCuda_noRecursion(VkPhysicalDevice vkPhysicalDevice, FzbMainScene* scene, HANDLE bvhFinishedSemaphoreHandle, uint32_t maxDepth) {
    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }
    extBVHSemaphore = importVulkanSemaphoreObjectFromNTHandle(bvhFinishedSemaphoreHandle);

    CHECK(cudaStreamCreate(&stream));
    //-------------------------------------------------初始化triangleInfoArray----------------------------------------------
    cudaExternalMemory_t vertexExtMem = importVulkanMemoryObjectFromNTHandle(scene->vertexBuffer.handle, scene->vertexBuffer.size, false);
    this->vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, scene->vertexBuffer.size);
    this->triangleNum = scene->indexBuffer.size / sizeof(uint32_t) / 3;
    createTriangleInfoArray(scene, stream);
    //-------------------------------------------初始化bvhNodeArray----------------------------------
   
    //如果每个叶节点只代表一个顶点，那么bvh数的节点总数固定，为2 * 叶节点数 - 1，因为没有度为1的节点
    //uint32_t maxNodeNum = uint32_t(pow(2, bvhDepth) - 1);
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

    gridSize = ceil((float)triangleNum / 512);
    blockSize = triangleNum > 512 ? 512 : triangleNum;
    createNode<true> << <gridSize, blockSize, 0, stream >> > (bvhNodeArray, nodeTempInfoArray, triangleTempInfoArray, nullptr, triangleNum_ptr0, triangleIndices1, triangleNum_ptr1);
    uint32_t nonLeafTriangleNum;
    CHECK(cudaMemcpy(&nonLeafTriangleNum, triangleNum_ptr1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t* isPrintfs;
    CHECK(cudaMalloc((void**)&isPrintfs, sizeof(uint32_t) * bvhNodeNum));
    CHECK(cudaMemset(isPrintfs, 0, sizeof(uint32_t) * bvhNodeNum));
    int time = 0;
    while (nonLeafTriangleNum > 0) {
        gridSize = ceil((float)nonLeafTriangleNum / 512);
        blockSize = nonLeafTriangleNum > 512 ? 512 : nonLeafTriangleNum;

        preDivideNode << <gridSize, blockSize, 0, stream >> > (nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1);
        divideNode << <gridSize, blockSize, 0, stream >> > (bvhNodeArray, nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1);
        //checkNodeTriangleNum<<<gridSize, blockSize, 0, stream>>>(nodeTempInfoArray, triangleTempInfoArray, triangleIndices1, triangleNum_ptr1, isPrintfs);
        //CHECK(cudaDeviceSynchronize());
        //CHECK(cudaMemset(isPrintfs, 0, sizeof(uint32_t) * bvhNodeNum));
        //std::cout << "\n\n";

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

    CHECK(cudaDestroyExternalMemory(vertexExtMem));
    CHECK(cudaFree(triangleTempInfoArray));
    CHECK(cudaFree(nodeTempInfoArray));
    CHECK(cudaFree(triangleIndices0));
    CHECK(cudaFree(triangleIndices1));
    CHECK(cudaFree(triangleNum_ptr0));
    CHECK(cudaFree(triangleNum_ptr1));
}

#endif