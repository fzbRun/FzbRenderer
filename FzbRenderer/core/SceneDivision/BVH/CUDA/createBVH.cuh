#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"
#include "../../../common/FzbMesh/FzbMesh.h"
#include "../../../common/FzbScene/FzbScene.h"

#ifndef CREATE_BVH_CUH
#define CREATE_BVH_CUH

struct FzbAABB_BVH {
	float leftX = FLT_MAX;
	float rightX = -FLT_MAX;
	float leftY = FLT_MAX;
	float rightY = -FLT_MAX;
	float leftZ = FLT_MAX;
	float rightZ = -FLT_MAX;

	__host__ __device__  FzbAABB_BVH() {};
	__host__ __device__ FzbAABB_BVH(float leftX, float rightX, float leftY, float rightY, float leftZ, float rightZ) {
		this->leftX = leftX;
		this->rightX = rightX;
		this->leftY = leftY;
		this->rightY = rightY;
		this->leftZ = leftZ;
		this->rightZ = rightZ;
	}
};

//-------------------------------------------------------------------------------------------------------------
struct FzbBvhNodeTriangleInfo {
	int materialIndex;
	uint32_t vertexFormat;	//3位，第一为表示是否使用法线，第二位表示是否使用uv，第三位表示是否使用tangent
	uint32_t indices0;
	uint32_t indices1;
	uint32_t indices2;
	//uint32_t meshVertexOffset;	//由于顶点格式不同，因此需要记录每个mesh顶点在数组中的偏移
};
struct FzbBvhNode {
	uint32_t leftNodeIndex;
	uint32_t rightNodeIndex;
	FzbAABB_BVH AABB;
};

/*
我们来讲述一下BVHCuda的工作原理，这里以非递归的方式，即createBvhCuda_noRecursion为准（递归的太慢了，不再更新）
1.	首先进入createBvhCuda_noRecursion函数，他会
 	1.1 判断CUDA和vulkan是不是使用一个GPU，然后创建stream
 	1.2 根据顶点缓冲和索引缓冲的 handle得到相应的GPU数组
 	1.3 调用createTriangleInfoArray函数，这个函数会重建最终所需的三角形信息FzbBvhNodeTriangleInfo数组，FzbBvhNodeTriangleInfo包括
 		a) 三角形所在mesh对应的materialIndex
 		b) 三角形的顶点格式
 		c) 三角形的三个顶点的索引
 		这是方便我们后续在光线追踪中使用这些数据得到AABB碰撞检测后的三角形数据
 		1.3.1 因为我们的sceneIndices是按照相同顶点格式的顶点索引聚集排放的，所以我们计算所有相同顶点格式的indexNum和起始索引
		1.3.2 在和函数中将每个三角形对应的索引从sceneIndices中搬到bvhTriangleInfoArray数组中
		1.3.3 然后将该顶点格式index的偏移和对应的三角形偏移保存，用于下一个顶点格式的index的搬运
		这样，我们得到了存有三角形信息的数组
	1.4  然后，我们初始化节点临时数组bvhNodeTempInfoArray，他是FzbBvhNodeTempInfo类型的，有
		a) triangleNum，该节点的三角形数量，用于后续决定时候继续划分
		b)  divideAxis，该节点的划分轴，是按x划分的还是yz
			c)  sumPos，float3，该节点中三角形的累加世界坐标，后续除以三角形数得到平均坐标，用于判断三角形位于左边还是右边
		d)  leftRiangleNum，一个大小为3的uint数组，记录按每个轴划分后，左节点的数量，用于后续计算SAH决定按哪个轴划分最优
		e)  SAHCost，一个大小为6的float数组，记录每个轴左右节点中三角形的表面积，用于后续计算SAH决定按哪个轴划分最优
		创建后，对根节点的triangleNum赋值三角形数
	1.5 然后，初始化triangleTempInfoArray，这是三角形临时信息数组，是FzbTriangleTempInfo类型的
		a)  三个顶点的世界坐标。之所以在createTriangleInfoArray中只存索引，有几点原因
			a.1 到时候光追索要使用的顶点数组是一个全格式的数组，所以只存pos不够，但是全存太大了
			a.2 同时只存index的好处是只需要三个uint，而存pos就需要9个float了
		b)  nodeIndex，三角形所处的节点索引
		c)  AABB，三角形的AABB，为后续node的AABB创建提供遍历
		d) triangleIndex， 三角形的索引，对应于FzbBvhNodeTriangleInfo中的索引。最后FzbBvhNodeArray中的右节点存储该索引，在光追碰撞检测后找到三角形信息
		1.5.1 initTriangle函数为每个三角形开辟一个线程，会现根据bvhTriangleInfoArray，去sceneVertices中找到相应的三角形顶点pos
			1.5.2 然后处理三角形可能重叠的问题，就是逆着法线方向偏移一点。
		1.5.3 然后计算三角形的AABB，将nodeIndex设为0，并将得到triangleIndex（就是线程索引）
	1.6 创建2个索引数组triangleIndices1 2，大小为三角形数，用于存储没达到bvh叶节点的三角形索引；创建两个uint，用于存储未达到bvh叶节点的三角形数triangleNum1 2
	我们划分bvh共分为3步
	1.7 通过createNode核函数，为每个需要没达到叶节点的三角形开辟一个线程
		1.7.1 每个线程从triangleIndices1找到相应的三角形索引（第一次的话就是线程索引），得到triangleTempInfo
		1.7.2 判断三角形所处的bvhNodeTempInfoArray的node的三角形数是不是1，如果是1，则直接将当前三角形的AABB赋给node，并将FzbBvhNodeArray的node的右子节点索引存为三角形索引（注意这里是前后node不是一个数组的）
		1.7.3 如果node的三角形数不为1，则继续划分，我们triangleNum2原子加，然后根据返回的索引，将三角形信息存入triangleIndices2对应位置
		1.7.4 计算三角形的平均pos，并为bvhNodeTempInfoArray的sumPos继续原子加该平均Pos（float原子加是自己实现）。同理，对node的AABB进行原子float加
	1.8 调用preDivideNode核函数，为每个三角形开辟一个线程
		1.8.1 每个线程从triangleIndices2中获得三角形索引，从bvhNodeTempInfoArray得到triangleTempInfo
		1.8.2 计算三角形的面积，然后计算所处node的sumPos的平均。
		1.8.3 然后每个三角形将自己的meanPos与node的meanPos进行比较，判断自己处于xyz轴的左边还是右边，将面积原子加入node相应的SAHCost中
	1.9 调用divideNode核函数，为每个三角形开辟一个线程
		1.9.1 同理找到triangleTempInfo，然后根据所处node的SAHCost和leftTriangleNum计算出sahCost，判断哪个轴的sahCos最小，将该轴作为划分轴
		1.9.2 计算三角形meanPos，判断位于划分轴的左边还是右边，为所处子节点的triangleNum原子加
			1.9.2.1 如果是左节点，那么左节点的索引就是当前nodeIndex+1，其triangleNum就是leftTriangleNum
				  1.9.2.2 如果是右节点，右节点索引=nodeIndex + leftTriangleNum*2 ，其triangleNum就是当前triangleNum - leftTriangleNum
	交换triangleIndices11和2、triangleNum11和2，然后重复1.7-1.9，直至triangleNum12 == 0
2. getBvhCuda函数将bvhNodeArray和bvhTriangleInfoArray传给vulkan
*/
struct BVHCuda {

public:
	cudaExternalMemory_t bvhNodeArrayExtMem = nullptr;
	cudaExternalMemory_t bvhTriangleInfoArrayMem = nullptr;
	uint32_t triangleNum = 0;

	float* vertices = nullptr;
	FzbBvhNode* bvhNodeArray = nullptr;
	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray = nullptr;

	cudaStream_t stream = nullptr;
	cudaExternalSemaphore_t extBVHSemaphore = nullptr;

	BVHCuda() {};

	void createTriangleInfoArray(FzbMainScene* scene, cudaStream_t stream);
	void createBvhCuda_recursion(VkPhysicalDevice vkPhysicalDevice, FzbMainScene& scene, HANDLE bvhFinishedSemaphoreHandle, uint32_t maxDepth);	//太慢了，废弃了，但是写了很多，不舍得删，放着吧
	void createBvhCuda_noRecursion(VkPhysicalDevice vkPhysicalDevice, FzbMainScene* scene, HANDLE bvhFinishedSemaphoreHandle, uint32_t maxDepth);	//不使用迭代
	void getBvhCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE bvhNodeArrayHandle, HANDLE bvhTriangleInfoArrayMem);
	void clean();
};

__global__ void initBvhNode(FzbBvhNode* bvhNodeArray, uint32_t nodeNum);


#endif
