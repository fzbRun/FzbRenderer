#include "./createBVH.cuh"
#include <helper_math.h>
#include <random>
#include <glm/ext/matrix_transform.hpp>

#ifndef CREATE_BVH_RECURSION_CU
#define CREATE_BVH_RECURSION_CU

struct FzbTriangleTempInfo {
    float3 pos0;
    float3 pos1;
    float3 pos2;
    uint32_t fatherNodeIndex;
    FzbAABB AABB;
    uint32_t triangleIndex;
};

//----------------------------------------------------------核函数-------------------------------------------------------
__global__ void initTriangle(float* vertices, FzbTriangleTempInfo* triangleTempInfoArray, FzbBvhNodeTriangleInfo* bvhTriangleInfoArray, uint32_t triangleNum) {
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

    FzbTriangleTempInfo triangleTempInfo;
    triangleTempInfo.pos0 = pos0;
    triangleTempInfo.pos1 = pos1;
    triangleTempInfo.pos2 = pos2;

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

    triangleTempInfo.fatherNodeIndex = 0;
    triangleTempInfo.triangleIndex = threadIndex;
    //__syncwarp();

    triangleTempInfoArray[threadIndex] = triangleTempInfo;
    //printf("%d:   %f %f %f\n", threadIndex, triangleTempInfoArray[threadIndex].pos0.x, triangleTempInfoArray[threadIndex].pos0.y, triangleTempInfoArray[threadIndex].pos0.z);
}

__global__ void createNodeAABB(bool isLeft, FzbBvhNode* bvhNodeArray, uint32_t nodeIndex, FzbTriangleTempInfo* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t triangleNum, float3* sumPos) {
    //每个线程组先获得各自的AABB，然后获得整个节点的AABB
    __shared__ FzbAABB groupVoxelNum;
    __shared__ float groupSumX;
    __shared__ float groupSumY;
    __shared__ float groupSumZ;

    uint32_t threadGroupIndex = threadIdx.x;
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warpLane = threadIndex & 31u;

    uint32_t mask = __ballot_sync(0xffffffff, threadIndex < triangleNum);
    if (threadIndex >= triangleNum) {
        return;
    }

    uint32_t triangleIndex = triangleIndices == nullptr ? threadIndex : triangleIndices[threadIndex];
    FzbTriangleTempInfo triangleTempInfo = triangleTempInfoArray[triangleIndex];
    //如果三角形数小于32，则无需共享内存
    if (triangleNum == 1) {
        bvhNodeArray[nodeIndex].AABB = triangleTempInfo.AABB;
        if (isLeft == 1) {
            bvhNodeArray[triangleTempInfo.fatherNodeIndex].leftNodeIndex = nodeIndex; //将父节点的左右节点赋值为当前节点
        }
        else {
            bvhNodeArray[triangleTempInfo.fatherNodeIndex].rightNodeIndex = nodeIndex;
        }
        bvhNodeArray[nodeIndex].rightNodeIndex = triangleTempInfo.triangleIndex;
        return;
    }

    if (threadGroupIndex == 0) {
        groupVoxelNum.leftX = FLT_MAX;
        groupVoxelNum.rightX = -FLT_MAX;
        groupVoxelNum.leftY = FLT_MAX;
        groupVoxelNum.rightY = -FLT_MAX;
        groupVoxelNum.leftZ = FLT_MAX;
        groupVoxelNum.rightZ = -FLT_MAX;

        groupSumX = 0.0f;
        groupSumY = 0.0f;
        groupSumZ = 0.0f;
    }
    float sumX = (triangleTempInfo.pos0.x + triangleTempInfo.pos1.x + triangleTempInfo.pos2.x) * 0.333f;
    float sumY = (triangleTempInfo.pos0.y + triangleTempInfo.pos1.y + triangleTempInfo.pos2.y) * 0.333f;
    float sumZ = (triangleTempInfo.pos0.z + triangleTempInfo.pos1.z + triangleTempInfo.pos2.z) * 0.333f;
    __syncwarp();
    float warpSumX = warpReduce(sumX);
    float warpSumY = warpReduce(sumY);
    float warpSumZ = warpReduce(sumZ);
    __syncthreads();

    if (warpLane == 0) {
        atomicAddFloat(&groupSumX, warpSumX);
        atomicAddFloat(&groupSumY, warpSumY);
        atomicAddFloat(&groupSumZ, warpSumZ);
    }

    //如果warp是全激活的，则warp内先取最大最小值；否则直接对共享内存进行操作。因为不激活的线程的__shfl_down_sync结果会返回0，导致取最大最小值可能出错。
    if (mask == 0xffffffff) {
        float leftX = warpMin(triangleTempInfo.AABB.leftX);
        float rightX = warpMax(triangleTempInfo.AABB.rightX);
        float leftY = warpMin(triangleTempInfo.AABB.leftY);
        float rightY = warpMax(triangleTempInfo.AABB.rightY);
        float leftZ = warpMin(triangleTempInfo.AABB.leftZ);
        float rightZ = warpMax(triangleTempInfo.AABB.rightZ);
        __syncwarp();
        if (warpLane == 0) {
            atomicMinFloat(&groupVoxelNum.leftX, leftX);
            atomicMaxFloat(&groupVoxelNum.rightX, rightX);
            atomicMinFloat(&groupVoxelNum.leftY, leftY);
            atomicMaxFloat(&groupVoxelNum.rightY, rightY);
            atomicMinFloat(&groupVoxelNum.leftZ, leftZ);
            atomicMaxFloat(&groupVoxelNum.rightZ, rightZ);
        }
    }
    else {
        atomicMinFloat(&groupVoxelNum.leftX, triangleTempInfo.AABB.leftX);
        atomicMaxFloat(&groupVoxelNum.rightX, triangleTempInfo.AABB.rightX);
        atomicMinFloat(&groupVoxelNum.leftY, triangleTempInfo.AABB.leftY);
        atomicMaxFloat(&groupVoxelNum.rightY, triangleTempInfo.AABB.rightY);
        atomicMinFloat(&groupVoxelNum.leftZ, triangleTempInfo.AABB.leftZ);
        atomicMaxFloat(&groupVoxelNum.rightZ, triangleTempInfo.AABB.rightZ);
    }
    __syncthreads();

    if (threadGroupIndex == 0) {
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftX, groupVoxelNum.leftX);   //为当前节点创造AABB
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightX, groupVoxelNum.rightX);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftY, groupVoxelNum.leftY);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightY, groupVoxelNum.rightY);
        atomicMinFloat(&bvhNodeArray[nodeIndex].AABB.leftZ, groupVoxelNum.leftZ);
        atomicMaxFloat(&bvhNodeArray[nodeIndex].AABB.rightZ, groupVoxelNum.rightZ);

        atomicAddFloat(&(sumPos->x), groupSumX);
        atomicAddFloat(&(sumPos->y), groupSumY);
        atomicAddFloat(&(sumPos->z), groupSumZ);

        if (threadIndex == 0) {
            if (isLeft) {
                bvhNodeArray[triangleTempInfo.fatherNodeIndex].leftNodeIndex = nodeIndex; //将父节点的左右节点赋值为当前节点
            }
            else {
                bvhNodeArray[triangleTempInfo.fatherNodeIndex].rightNodeIndex = nodeIndex;
            }
        }
    }

    triangleTempInfoArray[triangleIndex].fatherNodeIndex = nodeIndex; //更新三角形的父节点。
}

/*
__global__ void getTriangleSumPos(FzbTriangleTempInfo* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t triangleNum, float3* sumPos) {
    __shared__ float groupSumX;
    __shared__ float groupSumY;
    __shared__ float groupSumZ;

    uint32_t threadGroupIndex = threadIdx.x;
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warpLane = threadIndex & 31;
    if (threadGroupIndex == 0) {
        groupSumX = 0.0f;
        groupSumY = 0.0f;
        groupSumZ = 0.0f;
    }
    if (threadIndex >= triangleNum) {
        return;
    }
    uint32_t triangleIndex = triangleIndices == nullptr ? threadIndex : triangleIndices[threadIndex];
    FzbTriangleTempInfo triangleTempInfo = triangleTempInfoArray[triangleIndex];
    float sumX = (triangleTempInfo.pos0.x + triangleTempInfo.pos1.x + triangleTempInfo.pos2.x) * 0.333f;
    float sumY = (triangleTempInfo.pos0.y + triangleTempInfo.pos1.y + triangleTempInfo.pos2.y) * 0.333f;
    float sumZ = (triangleTempInfo.pos0.z + triangleTempInfo.pos1.z + triangleTempInfo.pos2.z) * 0.333f;
    __syncwarp();
    float warpSumX = warpReduce(sumX);
    float warpSumY = warpReduce(sumY);
    float warpSumZ = warpReduce(sumZ);
    __syncthreads();
    if (warpLane == 0) {
        atomicAddFloat(&groupSumX, warpSumX);
        atomicAddFloat(&groupSumY, warpSumY);
        atomicAddFloat(&groupSumZ, warpSumZ);
    }
    __syncthreads();

    if (threadGroupIndex == 0) {
        atomicAddFloat(&(sumPos->x), groupSumX);
        atomicAddFloat(&(sumPos->y), groupSumY);
        atomicAddFloat(&(sumPos->z), groupSumZ);
    }
}
*/
__global__ void getNodeDivideInfo(FzbTriangleTempInfo* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t triangleNum, float3* sumPos,
    uint32_t* leftNodeX, uint32_t* leftNodeTriangleNumX, uint32_t* rightNodeX, uint32_t* rightNodeTriangleNumX, float* leftNodeCostX, float* rightNodeCostX,
    uint32_t* leftNodeY, uint32_t* leftNodeTriangleNumY, uint32_t* rightNodeY, uint32_t* rightNodeTriangleNumY, float* leftNodeCostY, float* rightNodeCostY,
    uint32_t* leftNodeZ, uint32_t* leftNodeTriangleNumZ, uint32_t* rightNodeZ, uint32_t* rightNodeTriangleNumZ, float* leftNodeCostZ, float* rightNodeCostZ) {
    __shared__ float groupLeftNodeCostX;
    __shared__ float groupLeftNodeCostY;
    __shared__ float groupLeftNodeCostZ;
    __shared__ float groupRightNodeCostX;
    __shared__ float groupRightNodeCostY;
    __shared__ float groupRightNodeCostZ;

    uint32_t threadGroupIndex = threadIdx.x;
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warpLane = threadIndex & 31;
    if (threadIndex >= triangleNum) {
        return;
    }

    uint32_t triangleIndex = triangleIndices == nullptr ? threadIndex : triangleIndices[threadIndex];
    if (threadGroupIndex == 0) {
        groupLeftNodeCostX = 0.0f;
        groupLeftNodeCostY = 0.0f;
        groupLeftNodeCostZ = 0.0f;
        groupRightNodeCostX = 0.0f;
        groupRightNodeCostY = 0.0f;
        groupRightNodeCostZ = 0.0f;
    }

    float meanX = sumPos->x / triangleNum;
    float meanY = sumPos->y / triangleNum;
    float meanZ = sumPos->z / triangleNum;
    //其实我可以批处理，以一个warp为单位进行划分。这个后面再说吧，其实先按mesh划分再划分mesh也行。后面再说
    FzbTriangleTempInfo triangleTempInfo = triangleTempInfoArray[triangleIndex];
    float sumX = (triangleTempInfo.pos0.x + triangleTempInfo.pos1.x + triangleTempInfo.pos2.x) * 0.333f;
    float sumY = (triangleTempInfo.pos0.y + triangleTempInfo.pos1.y + triangleTempInfo.pos2.y) * 0.333f;
    float sumZ = (triangleTempInfo.pos0.z + triangleTempInfo.pos1.z + triangleTempInfo.pos2.z) * 0.333f;

    float3 edge1 = triangleTempInfo.pos1 - triangleTempInfo.pos0;
    float3 edge2 = triangleTempInfo.pos2 - triangleTempInfo.pos0;
    float3 crossResult = cross(edge1, edge2);
    float triangleArea = length(crossResult);   //面积的平方的两倍
    __syncthreads();

    //if (triangleNum == 19274) {  //11562 19274
    //    //printf("%d:  ", triangleIndex);
    //    printf("%d:   pos0: %f %f %f  pos1: %f %f %f  pos2: %f %f %f\n", triangleIndex, triangleTempInfo.pos0.x, triangleTempInfo.pos0.y, triangleTempInfo.pos0.z, 
    //        triangleTempInfo.pos1.x, triangleTempInfo.pos1.y, triangleTempInfo.pos1.z, 
    //        triangleTempInfo.pos2.x, triangleTempInfo.pos2.y, triangleTempInfo.pos2.z);
    //}

    float triangleLeftCost = 0.0f;
    float triangleRightCost = 0.0f;
    float warpLeftNodeCost;
    float warpRightNodeCost;
    //按x轴划分
    if (sumX <= meanX) {
        uint32_t leftNodeTriangleIndex = atomicAdd(leftNodeTriangleNumX, 1);
        leftNodeX[leftNodeTriangleIndex] = triangleIndex;
        triangleLeftCost = triangleArea;
    }
    else {
        uint32_t rightNodeTriangleIndex = atomicAdd(rightNodeTriangleNumX, 1);
        rightNodeX[rightNodeTriangleIndex] = triangleIndex;
        triangleRightCost = triangleArea;
    }
    warpLeftNodeCost = warpReduce(triangleLeftCost);
    warpRightNodeCost = warpReduce(triangleRightCost);
    if (warpLane == 0) {
        atomicAddFloat(&groupLeftNodeCostX, warpLeftNodeCost);
        atomicAddFloat(&groupRightNodeCostX, warpRightNodeCost);
    }
    __syncwarp();

    triangleLeftCost = 0.0f;
    triangleRightCost = 0.0f;
    //按y轴划分
    if (sumY <= meanY) {
        uint32_t leftNodeTriangleIndex = atomicAdd(leftNodeTriangleNumY, 1);
        leftNodeY[leftNodeTriangleIndex] = triangleIndex;
        triangleLeftCost = triangleArea;
    }
    else {
        uint32_t rightNodeTriangleIndex = atomicAdd(rightNodeTriangleNumY, 1);
        rightNodeY[rightNodeTriangleIndex] = triangleIndex;
        triangleRightCost = triangleArea;
    }
    warpLeftNodeCost = warpReduce(triangleLeftCost);
    warpRightNodeCost = warpReduce(triangleRightCost);
    if (warpLane == 0) {
        atomicAddFloat(&groupLeftNodeCostY, warpLeftNodeCost);
        atomicAddFloat(&groupRightNodeCostY, warpRightNodeCost);
    }
    __syncwarp();

    triangleLeftCost = 0.0f;
    triangleRightCost = 0.0f;
    //按z轴划分
    if (sumZ <= meanZ) {
        uint32_t leftNodeTriangleIndex = atomicAdd(leftNodeTriangleNumZ, 1);
        leftNodeZ[leftNodeTriangleIndex] = triangleIndex;
        triangleLeftCost = triangleArea;
    }
    else {
        uint32_t rightNodeTriangleIndex = atomicAdd(rightNodeTriangleNumZ, 1);
        rightNodeZ[rightNodeTriangleIndex] = triangleIndex;
        triangleRightCost = triangleArea;
    }
    warpLeftNodeCost = warpReduce(triangleLeftCost);
    warpRightNodeCost = warpReduce(triangleRightCost);
    if (warpLane == 0) {
        atomicAddFloat(&groupLeftNodeCostZ, warpLeftNodeCost);
        atomicAddFloat(&groupRightNodeCostZ, warpRightNodeCost);
    }
    __syncthreads();
    
    if (threadGroupIndex == 0) {
        atomicAddFloat(leftNodeCostX, groupLeftNodeCostX);
        atomicAddFloat(rightNodeCostX, groupRightNodeCostX);
        atomicAddFloat(leftNodeCostY, groupLeftNodeCostY);
        atomicAddFloat(rightNodeCostY, groupRightNodeCostY);
        atomicAddFloat(leftNodeCostZ, groupLeftNodeCostZ);
        atomicAddFloat(rightNodeCostZ, groupRightNodeCostZ);
    }
}
//----------------------------------------------------------递归的方式---------------------------------------------------------

//每个divideScene中的triangleTempInfoArray内的三角形必然是同一个父节点。
//其实，如果按CPU这样同步的来的话，我是知道每次递归时节点的索引的，这里的索引是左子树优先的
void divideScene(bool isLeft, FzbBvhNode* bvhNodeArray, uint32_t& nodeIndex, FzbTriangleTempInfo* triangleTempInfoArray, uint32_t* triangleIndices, uint32_t triangleNum, cudaStream_t stream) {

    uint32_t gridSize = ceil((float)triangleNum / 512);
    uint32_t blockSize = triangleNum > 512 ? 512 : triangleNum;

    float3* sumPos = nullptr;
    if (triangleNum > 1) {
        CHECK(cudaMalloc((void**)&sumPos, sizeof(float3)));
        CHECK(cudaMemset(sumPos, 0, sizeof(float3)));
    }

    createNodeAABB << < gridSize, blockSize, 0, stream >> > (isLeft, bvhNodeArray, nodeIndex, triangleTempInfoArray, triangleIndices, triangleNum, sumPos);
    //CHECK(cudaStreamSynchronize(stream));
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    printf("====== Kernel launch failed: %s ======\n",
    //        cudaGetErrorString(err));
    //}
    if (triangleNum == 1) {
        return;
    }

    std::vector<uint32_t*> leftNode(3);
    std::vector<uint32_t*> rightNode(3);
    std::vector<uint32_t*> leftNodeTriangleNum(3);
    std::vector<uint32_t*> rightNodeTriangleNum(3);
    std::vector<float*> leftNodeCost(3);
    std::vector<float*> rightNodeCost(3);
    for (int i = 0; i < 3; i++) {
        CHECK(cudaMalloc((void**)&leftNode[i], sizeof(uint32_t) * triangleNum));
        CHECK(cudaMalloc((void**)&rightNode[i], sizeof(uint32_t) * triangleNum));

        CHECK(cudaMalloc((void**)&leftNodeTriangleNum[i], sizeof(uint32_t)));
        CHECK(cudaMemset(leftNodeTriangleNum[i], 0, sizeof(uint32_t)));
        CHECK(cudaMalloc((void**)&rightNodeTriangleNum[i], sizeof(uint32_t)));
        CHECK(cudaMemset(rightNodeTriangleNum[i], 0, sizeof(uint32_t)));

        CHECK(cudaMalloc((void**)&leftNodeCost[i], sizeof(float)));
        CHECK(cudaMemset(leftNodeCost[i], 0, sizeof(float)));
        CHECK(cudaMalloc((void**)&rightNodeCost[i], sizeof(float)));
        CHECK(cudaMemset(rightNodeCost[i], 0, sizeof(float)));
    }

    gridSize = ceil((float)triangleNum / 512);
    blockSize = triangleNum > 512 ? 512 : triangleNum;
    getNodeDivideInfo << <gridSize, blockSize, 0, stream >> > (triangleTempInfoArray, triangleIndices, triangleNum, sumPos,
        leftNode[0], leftNodeTriangleNum[0], rightNode[0], rightNodeTriangleNum[0], leftNodeCost[0], rightNodeCost[0],
        leftNode[1], leftNodeTriangleNum[1], rightNode[1], rightNodeTriangleNum[1], leftNodeCost[1], rightNodeCost[1],
        leftNode[2], leftNodeTriangleNum[2], rightNode[2], rightNodeTriangleNum[2], leftNodeCost[2], rightNodeCost[2]);
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    printf("====== Kernel launch failed: %s ======\n",
    //        cudaGetErrorString(err));
    //}
    CHECK(cudaStreamSynchronize(stream));

    std::vector<uint32_t> leftNodeTriangleNum_host(3);
    std::vector<uint32_t> rightNodeTriangleNum_host(3);
    std::vector<float> leftNodeCost_host(3);
    std::vector<float> rightNodeCost_host(3);
    std::vector<float> nodeCost(3);
    bool isUnDivide = true;
    std::vector<std::string> axisString = { "x", "y", "z" };
    for (int i = 0; i < 3; i++) {
        //std::cout << "当前轴为 " << axisString[i] << std::endl;
        CHECK(cudaMemcpy(&leftNodeTriangleNum_host[i], leftNodeTriangleNum[i], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&rightNodeTriangleNum_host[i], rightNodeTriangleNum[i], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&leftNodeCost_host[i], leftNodeCost[i], sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&rightNodeCost_host[i], rightNodeCost[i], sizeof(float), cudaMemcpyDeviceToHost));

        if (leftNodeTriangleNum_host[i] == 0 || rightNodeTriangleNum_host[i] == 0) {
            nodeCost[i] = FLT_MAX;
            continue;
        }
        isUnDivide = false;
        nodeCost[i] = leftNodeTriangleNum_host[i] * leftNodeCost_host[i] + rightNodeTriangleNum_host[i] * rightNodeCost_host[i];
        //std::cout << "SAH cost: " << nodeCost[i] << std::endl;
    }
    if (isUnDivide) {
        std::cout << "输入三角形数为" << triangleNum << std::endl;
        //std::vector<FzbTriangleTempInfo> triangleTempInfoArray_host(triangleNumAll);
        //CHECK(cudaMemcpy(triangleTempInfoArray_host.data(), triangleTempInfoArray, sizeof(FzbTriangleTempInfo) * triangleNumAll, cudaMemcpyDeviceToHost));
        //for (int i = 0; i < 3; i++) {
        //    std::cout << "左子树数量为 " << leftNodeTriangleNum_host[i] << std::endl;
        //    std::vector<uint32_t> leftNode_host(leftNodeTriangleNum_host[i]);
        //    CHECK(cudaMemcpy(leftNode_host.data(), leftNode[i], sizeof(uint32_t) * leftNodeTriangleNum_host[i], cudaMemcpyDeviceToHost));
        //    for (int j = 0; j < leftNodeTriangleNum_host[i]; j++) {
        //        std::cout << "三角形:" << leftNode_host[j];
        //        std::cout << " pos0 : ";
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos0.x;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos0.y;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos0.z;
        //        std::cout << " pos1 : ";
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos1.x;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos1.y;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos1.z;
        //        std::cout << " pos2 : ";
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos2.x;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos2.y;
        //        std::cout << " " << triangleTempInfoArray_host[leftNode_host[j]].pos2.z;
        //        std::cout << std::endl;
        //    }
        //    std::vector<uint32_t> rightNode_host(rightNodeTriangleNum_host[i]);
        //    CHECK(cudaMemcpy(rightNode_host.data(), rightNode[i], sizeof(uint32_t) * rightNodeTriangleNum_host[i], cudaMemcpyDeviceToHost));
        //    std::cout << "右子树数量为 " << rightNodeTriangleNum_host[i] << std::endl;
        //    for (int j = 0; j < rightNodeTriangleNum_host[i]; j++) {
        //        std::cout << "三角形:" << rightNode_host[j];
        //        std::cout << " pos0 : ";
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos0.x;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos0.y;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos0.z;
        //        std::cout << " pos1 : ";
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos1.x;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos1.y;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos1.z;
        //        std::cout << " pos2 : ";
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos2.x;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos2.y;
        //        std::cout << " " << triangleTempInfoArray_host[rightNode_host[j]].pos2.z;
        //        std::cout << std::endl;
        //    }
        //    std::cout << "左子树cost为 " << leftNodeCost_host[i] << std::endl;
        //    std::cout << "右子树cost为 " << rightNodeCost_host[i] << std::endl;;
        //}
        throw std::runtime_error("有分不开的node");
    }
    uint32_t axis = nodeCost[0] < nodeCost[1] ? nodeCost[0] < nodeCost[2] ? 0 : 2 : nodeCost[1] < nodeCost[2] ? 1 : 2;
    //std::cout << "划分的轴为" << axis << std::endl << std::endl;
    for (int i = 0; i < 3; i++) {
        if (i != axis) {
            CHECK(cudaFree(leftNode[i]));
            CHECK(cudaFree(rightNode[i]));
        }
        CHECK(cudaFree(leftNodeTriangleNum[i]));
        CHECK(cudaFree(rightNodeTriangleNum[i]));
        CHECK(cudaFree(leftNodeCost[i]));
        CHECK(cudaFree(rightNodeCost[i]));
    }
    CHECK(cudaFree(sumPos));

    divideScene(true, bvhNodeArray, ++nodeIndex, triangleTempInfoArray, leftNode[axis], leftNodeTriangleNum_host[axis], stream);
    divideScene(false, bvhNodeArray, ++nodeIndex, triangleTempInfoArray, rightNode[axis], rightNodeTriangleNum_host[axis], stream);

    CHECK(cudaFree(leftNode[axis]));
    CHECK(cudaFree(rightNode[axis]));
}

void BVHCuda::createBvhCuda_recursion(VkPhysicalDevice vkPhysicalDevice, FzbMainScene& scene, HANDLE bvhFinishedSemaphoreHandle, uint32_t maxDepth) {
    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }
    extBVHSemaphore = importVulkanSemaphoreObjectFromNTHandle(bvhFinishedSemaphoreHandle);

    CHECK(cudaStreamCreate(&stream));

    std::vector<float>& vertices_host = scene.sceneVertices;       //压缩后的顶点数据
    std::vector<uint32_t>& sceneIndices_host = scene.sceneIndices;
    this->triangleNum = sceneIndices_host.size() / 3;
    std::vector<FzbBvhNodeTriangleInfo> triangleInfoArray_host(triangleNum);     //三角形信息
    
    //uint32_t triangleIndex = 0;
    //for (int i = 0; i < scene.sceneMeshIndices.size(); i++) {
    //    FzbMesh& mesh = scene.sceneMeshSet[scene.sceneMeshIndices[i]];
    //    FzbVertexFormat vertexFormat = mesh.vertexFormat;
    //    uint32_t vertexStrip = 3 + 3 * vertexFormat.useNormal + 2 * vertexFormat.useTexCoord + 3 * vertexFormat.useTangent;
    //    for (int j = 0; j < mesh.indices.size(); j += 3) {
    //        triangleInfoArray_host[triangleIndex + j / 3].vertexFormat = (vertexFormat.useTangent << 2) | (vertexFormat.useTexCoord << 1) | (vertexFormat.useNormal);
    //        triangleInfoArray_host[triangleIndex + j / 3].indices0 = sceneIndices_host[mesh.indexArrayOffset + j];
    //        triangleInfoArray_host[triangleIndex + j / 3].indices1 = sceneIndices_host[mesh.indexArrayOffset + j + 1];
    //        triangleInfoArray_host[triangleIndex + j / 3].indices2 = sceneIndices_host[mesh.indexArrayOffset + j + 2];
    //    }
    //    triangleIndex += mesh.indices.size() / 3;
    //}
    //CHECK(cudaMalloc((void**)&bvhTriangleInfoArray, sizeof(FzbBvhNodeTriangleInfo) * triangleNum));
    //CHECK(cudaMemcpy(bvhTriangleInfoArray, triangleInfoArray_host.data(), sizeof(FzbBvhNodeTriangleInfo) * triangleNum, cudaMemcpyHostToDevice));
    createTriangleInfoArray(scene, stream);

    FzbTriangleTempInfo* triangleTempInfoArray;
    CHECK(cudaMalloc((void**)&triangleTempInfoArray, sizeof(FzbTriangleTempInfo) * triangleNum));
    CHECK(cudaMemset(triangleTempInfoArray, 0, sizeof(FzbTriangleTempInfo) * triangleNum));

    float* vertices;
    CHECK(cudaMalloc((void**)&vertices, sizeof(float) * vertices_host.size()));
    CHECK(cudaMemcpy(vertices, vertices_host.data(), sizeof(float) * vertices_host.size(), cudaMemcpyHostToDevice));

    uint32_t bvhDepth = 1;
    uint32_t triangleNum_temp = triangleNum;
    while (triangleNum_temp > 1) {
        bvhDepth++;
        triangleNum_temp >>= 1;
    }
    if (bvhDepth > maxDepth) {
        //这里应该处理一下如果超过最大深度，则一个叶节点要代表多个顶点了。
        //算了，还是按实际的来吧，如果要处理多个顶点，则又要给shader传额外的信息
    }
    //如果每个叶节点只代表一个顶点，那么bvh数的节点总数固定，为2 * 叶节点数 - 1，因为没有度为1的节点
    //uint32_t maxNodeNum = uint32_t(pow(2, bvhDepth) - 1);
    uint32_t bvhNodeNum = 2 * triangleNum - 1;

    CHECK(cudaMalloc((void**)&bvhNodeArray, sizeof(FzbBvhNode) * bvhNodeNum));
    uint32_t gridSize_node = ceil((float)bvhNodeNum / 1024);
    uint32_t blockSize_node = bvhNodeNum > 1024 ? 1024 : bvhNodeNum;
    initBvhNode << < gridSize_node, blockSize_node, 0, stream >> > (bvhNodeArray, bvhNodeNum);

    uint32_t gridSize = ceil((float)triangleNum / 1024);
    uint32_t blockSize = triangleNum > 1024 ? 1024 : triangleNum;
    initTriangle << < gridSize, blockSize, 0, stream >> > (vertices, triangleTempInfoArray, bvhTriangleInfoArray, triangleNum);
   //CHECK(cudaDeviceSynchronize());
    
    //std::vector<FzbTriangleTempInfo> triangleTemInfoArray_host(triangleNum);
    //CHECK(cudaMemcpy(triangleTemInfoArray_host.data(), triangleTempInfoArray, sizeof(FzbTriangleTempInfo) * triangleNum, cudaMemcpyDeviceToHost));
    //for (int i = 5000; i < 10000; i++) {
    //    FzbTriangleTempInfo triangleTempInfo_host = triangleTemInfoArray_host[i];
    //    printf("%d:   pos0: %f %f %f  pos1: %f %f %f  pos2: %f %f %f\n", i, triangleTempInfo_host.pos0.x, triangleTempInfo_host.pos0.y, triangleTempInfo_host.pos0.z,
    //        triangleTempInfo_host.pos1.x, triangleTempInfo_host.pos1.y, triangleTempInfo_host.pos1.z,
    //        triangleTempInfo_host.pos2.x, triangleTempInfo_host.pos2.y, triangleTempInfo_host.pos2.z);
    //}

    //CHECK(cudaStreamSynchronize(stream));
    //std::vector<FzbTriangleTempInfo> triangleTempInfoArray_host(triangleNum);
    //CHECK(cudaMemcpy(triangleTempInfoArray_host.data(), triangleTempInfoArray, sizeof(FzbTriangleTempInfo) * triangleNum, cudaMemcpyDeviceToHost));
    //for (int i = 0; i < triangleNum; i++) {
    //    uint32_t vertexStrip = 3 + 3 * (triangleInfoArray_host[i].vertexFormat & 1) + 2 * ((triangleInfoArray_host[i].vertexFormat >> 1) & 1) + 3 * ((triangleInfoArray_host[i].vertexFormat >> 2) & 1);
    //    uint32_t indices = triangleInfoArray_host[i].indices0;
    //    float3 pos0 = make_float3(vertices_host[indices * vertexStrip], vertices_host[indices * vertexStrip + 1], vertices_host[indices * vertexStrip + 2]);
    //    indices = triangleInfoArray_host[i].indices1;
    //    float3 pos1 = make_float3(vertices_host[indices * vertexStrip], vertices_host[indices * vertexStrip + 1], vertices_host[indices * vertexStrip + 2]);
    //    indices = triangleInfoArray_host[i].indices2;
    //    float3 pos2 = make_float3(vertices_host[indices * vertexStrip], vertices_host[indices * vertexStrip + 1], vertices_host[indices * vertexStrip + 2]);
    //    if (pos0.x != triangleTempInfoArray_host[i].pos0.x || pos0.y != triangleTempInfoArray_host[i].pos0.y || pos0.z != triangleTempInfoArray_host[i].pos0.z ||
    //        pos1.x != triangleTempInfoArray_host[i].pos1.x || pos1.y != triangleTempInfoArray_host[i].pos1.y || pos1.z != triangleTempInfoArray_host[i].pos1.z ||
    //        pos2.x != triangleTempInfoArray_host[i].pos2.x || pos2.y != triangleTempInfoArray_host[i].pos2.y || pos2.z != triangleTempInfoArray_host[i].pos2.z) {
    //        printf("三角形pos: x: %f %f %f, y: %f %f %f, z: %f %f %f \n", pos0.x, pos0.y, pos0.z, pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z);
    //        printf("三角形pos: x: %f %f %f, y: %f %f %f, z: %f %f %f \n", triangleTempInfoArray_host[i].pos0.x, triangleTempInfoArray_host[i].pos0.y, triangleTempInfoArray_host[i].pos0.z, triangleTempInfoArray_host[i].pos1.x, triangleTempInfoArray_host[i].pos1.y, triangleTempInfoArray_host[i].pos1.z, triangleTempInfoArray_host[i].pos2.x, triangleTempInfoArray_host[i].pos2.y, triangleTempInfoArray_host[i].pos2.z);
    //        std::cout << i << std::endl;
    //        break;
    //    }
    //}
    //for (int i = 0; i < triangleNum; i++) {
    //    printf("三角形pos: pos0: %f %f %f, pos1: %f %f %f, pos2: %f %f %f \n", triangleTempInfoArray_host[i].pos0.x, triangleTempInfoArray_host[i].pos0.y, triangleTempInfoArray_host[i].pos0.z, triangleTempInfoArray_host[i].pos1.x, triangleTempInfoArray_host[i].pos1.y, triangleTempInfoArray_host[i].pos1.z, triangleTempInfoArray_host[i].pos2.x, triangleTempInfoArray_host[i].pos2.y, triangleTempInfoArray_host[i].pos2.z);
    //}
    //std::cout << std::endl;

    uint32_t nodeIndex = 0;
    divideScene(true, bvhNodeArray, nodeIndex, triangleTempInfoArray, nullptr, triangleNum, stream);
    //CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(triangleTempInfoArray));
    CHECK(cudaFree(vertices));

}


#endif