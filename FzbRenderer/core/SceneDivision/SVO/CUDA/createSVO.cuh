#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//前24为表示归一化的世界坐标，后8位表示体素内三角形包含的像素数量
};

struct FzbSVONode {
	uint32_t shuffleKey;	//当前节点在八叉树中的各级三维索引, z1y1x1……z7y7x7；前4位额外表示层级
	uint32_t voxelNum;	//该节点所包含的叶子节点数
	uint32_t label;		//在八叉树同一层的节点中是第几个有值节点，从1开始
	uint32_t hasSubNode;
	
	__device__ __host__ FzbSVONode() {
		shuffleKey = 0;
		voxelNum = 0;
		label = 1;
		hasSubNode = 0;
	}

};

struct FzbNodePoolBlock {
	uint32_t startIndex;
	uint32_t blockNum;
	uint32_t nodeNum;

	__device__ FzbNodePoolBlock(uint32_t startIndex, uint32_t blockNum, uint32_t nodeNum) {
		this->startIndex = startIndex;
		this->blockNum = blockNum;
		this->nodeNum = nodeNum;
	}

	__device__ FzbNodePoolBlock() {
		this->startIndex = 0;
		this->blockNum = 0;
	}
};

/*
这里描述一下整个Cuda创建SVO的过程
1. vulkan传入物理设备、vgm、vgm创建完成的信号量以及cuda完成后应该唤醒的信号量
2. createSVOCuda函数是创建SVO函数的主要函数，其主要分为几步
	2.1 判断cuda和vulkan使用的物理设备（GPU）是否相同，并创建vgm的纹理映射
	2.2 根据vgm的size判断八叉树的最大深度，创建stream，创建各种所需的buffer：
		a) vgm中有值体素数量voxelNum：uint32_t
		b) 满八叉树buffer svoNodeArray：FzbSVONode
		c) vgm体素数量的buffer，用于存储临时有值体素value的svoVoxelValueArray：FzbVoxelValue
		d) 非空八叉树节点数subArrayNum：uint32_t
		e) 八叉树数组tempNodePool: FzbSVONode
		d) 最终有序的八叉树数组nodePool: FzbSVONode
		f) 组内有序，组件无序的八叉树数组的临时信息，二维数组，每一维数组大小为 八叉树层数/线程组大小 ，threadBlockInfos: FzbNodePoolBlock
		g) 有序排序时需要一个每层第一个块的起始索引blockStartIndex：uint32_t
	2.3 调用getSVONum_step1函数，为每个vgm的体素开辟一个线程（8x8x8为一个线程组）
		2.3.1 从vgm中获取体素值，如果体素有值，则对共享内存groupVoxelNum进行原子累加，并得到线程在线程组有值线程中的索引voxelLocalIndex
		2.3.2 线程组第一个线程通过原子累加将groupVoxelNum加到voxelNum中，表明当前线程组有几个有值体素，得到线程组在voxelNum中的偏移groupVoxelOffset
		2.3.3 对于有值体素，我们在后续需要获得他的voxelIndex，所以这个int3压缩成一个uint（因为规定最大128，所以3 * 7 = 21 < 32）
			  存于暂时没用的svoNodeArray的shuffleKey中。并且为了防止影响非叶节点的写入，需要写在所有叶节点的shuffleKey中
		2.3.4 需要将有值体素value存于svoVoxelValueArray中，我们知道svoVoxelValueArray大小与有值voxelNum相同，则可以直接通过
			  voxelLocalIndex + groupVoxelOffset存入体素的value。
		2.3.5 如果线程组有值，那么线程组第一个线程自上而下对一路上的父节点的voxelNum赋值，并对其hasSubNode进行原子或运算，这有助于后续我们判断其
			  哪个子节点有值
	2.4 根据getSVONum_step1获得的voxelNum，为每个有值voxel开辟线程（线程组大小为512），调用getSVONum_step2
		2.4.1 从svoNodeArray的shuffleKey解压出体素的三维索引
		2.4.2 自上而下找自己的父节点，找到所处的8x8x8的体素块（getSVONum_step1的线程组）即的根节点
		2.4.3 从体素块的根节点自上而下的找到各个父节点，对其hasSubNode和voxelNum进行原子运算
		2.4.4 最后找到自身所在节点，对其voxelNum进行赋值1

	现在获得了一个满八叉树数组，八叉树上有值的节点均写在其中。现在需要对其进行压缩，我们可以按层写入，即为八叉树的每层开辟线程
	并且，对于第0层和第一层，我们可以直接使用写入，而不需要复杂判断，可以第0、1、2三层同时写入。
	然后第3层是512个线程，也可以使用一个线程组写入。但是之后的每层都需要多个线程组，这样我们无法保证有序，所以还需要一个核函数去排序。

	2.5 compressSVO_Step1的思路就是每8个节点为一个块，计算出有几个块有值（有一个系欸但有值就有值），将有值块写入
		2.5.1 对于0，1，2，3层
			2.5.1.1 我们创建4个共享变量
				a) blockHasValue：uint64_t，每一位表示一个8个节点的体素块是否有值
				b) blockIndexOfGroup：当前体素块在线程组中的偏移，通过原子加得到
				c) nodeNum：线程组节点数，用于多个线程组计算label
				d) blockNodeNum：64个uint的数组，用于存储每个体素块有几个有值体素
			2.5.1.2 每个节点先判断自己有没有值，如果有值，则对blockNodeNum相应的索引原子加
			2.5.1.3 每个有值体素块对blockHasValue的相应位进行赋1
			2.5.1.4 线程组获取blockHasValue中1的数量blockNum，表明了线程组有几个有值体素块，然后对subArrayNum进行原子加，得到线程组的体素块在nodePool中
					的起始索引blockIndexOfGroup
			2.5.1.5 每个线程获取自身所在体素块在该层所有体素块中的索引blockIndex
			2.5.1.6 如果节点有值，则开始计算节点的label，label=所有在前面的有值体素块的有值节点数+自身在体素块中的有值体素的索引
			2.5.1.7 计算shuffleKey，最后将节点赋给nodePool中[1 + blockIndex * 8 + threadOffsetInBlock（体素块中的索引）
			2.5.1.8 如果是第一次启动核函数，则对第0、1层进行赋值
		2.5.2 对于剩下的层，大部分都与2.5.1相同，下面说一下少量不同
			2.5.2.1 如果节点有值，则对nodeNum进行原子加，然后线程组对threadBlockInfos对于位置存储FzbNodePoolBlock(blockIndexOfGroup, blockNum, nodeNum)
			2.5.2.2 每个线程组对blockStartIndex使用原子最小，将最小的blockIndexOfGroup写入，得到该层的块的起始索引
			2.5.2.3 最后将节点存于temNodePool，用于下一个函数使用
	2.6 对于多个线程组，我们还需要对其进行排序，compressSVO_Step2，为每个节点开辟一个线程
		2.6.1 创建两个共享变量 
			a) threadBlockInfo：用于存储该线程组在compressSVO_Step1中存储的threadBlockInfos
			b) firstBlockIndex：blockStartIndex
		2.6.2 每个线程组通过threadBlockInfos对两个共享变量进行赋值
		2.6.3 因为每个线程组只有一部分体素块是有值的，那么我们对于不是这些体素块的无值节点就直接return就好了
		2.6.4 有值节点先找到存于tempNodePool中的数据
		2.6.5 对于每个之前的线程组，得到他们的有值体素块的数量blockNum*8累加，然后根据firstBlockIndex*8+1得到当前线程组的起始块索引
			  那么，该节点在nodePool中的索引就是1 + firstBlockIndex * 8 + 累加(blockNum * 8) + threadLocalIndex了，将node复制到nodePool的对应位置即可
3. 然后getSVOCuda函数将得到的nodePool和svoVoxelValueArray复制到vulkan传进来的buffer中，删除各种buffer和变量。
*/
class SVOCuda {

public:

	cudaExternalMemory_t vgmExtMem;
	cudaMipmappedArray_t vgmMipmap;
	cudaTextureObject_t vgm = 0;
	cudaExternalSemaphore_t extVgmSemaphore;
	cudaExternalSemaphore_t extSvoSemaphore;

	cudaExternalMemory_t nodePoolExtMem;
	cudaExternalMemory_t voxelValueArrayExtMem;

	cudaStream_t stream;

	uint32_t nodeBlockNum;	//8个数据的block的数量，需要额外加1表示根节点
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//后续所需要的节点数组
	FzbVoxelValue* svoVoxelValueArray;	//后续所需要的体素数据


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif