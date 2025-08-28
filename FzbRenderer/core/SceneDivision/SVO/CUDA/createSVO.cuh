#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//ǰ24Ϊ��ʾ��һ�����������꣬��8λ��ʾ�����������ΰ�������������
};

struct FzbSVONode {
	uint32_t shuffleKey;	//��ǰ�ڵ��ڰ˲����еĸ�����ά����, z1y1x1����z7y7x7��ǰ4λ�����ʾ�㼶
	uint32_t voxelNum;	//�ýڵ���������Ҷ�ӽڵ���
	uint32_t label;		//�ڰ˲���ͬһ��Ľڵ����ǵڼ�����ֵ�ڵ㣬��1��ʼ
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
��������һ������Cuda����SVO�Ĺ���
1. vulkan���������豸��vgm��vgm������ɵ��ź����Լ�cuda��ɺ�Ӧ�û��ѵ��ź���
2. createSVOCuda�����Ǵ���SVO��������Ҫ����������Ҫ��Ϊ����
	2.1 �ж�cuda��vulkanʹ�õ������豸��GPU���Ƿ���ͬ��������vgm������ӳ��
	2.2 ����vgm��size�жϰ˲����������ȣ�����stream���������������buffer��
		a) vgm����ֵ��������voxelNum��uint32_t
		b) ���˲���buffer svoNodeArray��FzbSVONode
		c) vgm����������buffer�����ڴ洢��ʱ��ֵ����value��svoVoxelValueArray��FzbVoxelValue
		d) �ǿհ˲����ڵ���subArrayNum��uint32_t
		e) �˲�������tempNodePool: FzbSVONode
		d) ��������İ˲�������nodePool: FzbSVONode
		f) ���������������İ˲����������ʱ��Ϣ����ά���飬ÿһά�����СΪ �˲�������/�߳����С ��threadBlockInfos: FzbNodePoolBlock
		g) ��������ʱ��Ҫһ��ÿ���һ�������ʼ����blockStartIndex��uint32_t
	2.3 ����getSVONum_step1������Ϊÿ��vgm�����ؿ���һ���̣߳�8x8x8Ϊһ���߳��飩
		2.3.1 ��vgm�л�ȡ����ֵ�����������ֵ����Թ����ڴ�groupVoxelNum����ԭ���ۼӣ����õ��߳����߳�����ֵ�߳��е�����voxelLocalIndex
		2.3.2 �߳����һ���߳�ͨ��ԭ���ۼӽ�groupVoxelNum�ӵ�voxelNum�У�������ǰ�߳����м�����ֵ���أ��õ��߳�����voxelNum�е�ƫ��groupVoxelOffset
		2.3.3 ������ֵ���أ������ں�����Ҫ�������voxelIndex���������int3ѹ����һ��uint����Ϊ�涨���128������3 * 7 = 21 < 32��
			  ������ʱû�õ�svoNodeArray��shuffleKey�С�����Ϊ�˷�ֹӰ���Ҷ�ڵ��д�룬��Ҫд������Ҷ�ڵ��shuffleKey��
		2.3.4 ��Ҫ����ֵ����value����svoVoxelValueArray�У�����֪��svoVoxelValueArray��С����ֵvoxelNum��ͬ�������ֱ��ͨ��
			  voxelLocalIndex + groupVoxelOffset�������ص�value��
		2.3.5 ����߳�����ֵ����ô�߳����һ���߳����϶��¶�һ·�ϵĸ��ڵ��voxelNum��ֵ��������hasSubNode����ԭ�ӻ����㣬�������ں��������ж���
			  �ĸ��ӽڵ���ֵ
	2.4 ����getSVONum_step1��õ�voxelNum��Ϊÿ����ֵvoxel�����̣߳��߳����СΪ512��������getSVONum_step2
		2.4.1 ��svoNodeArray��shuffleKey��ѹ�����ص���ά����
		2.4.2 ���϶������Լ��ĸ��ڵ㣬�ҵ�������8x8x8�����ؿ飨getSVONum_step1���߳��飩���ĸ��ڵ�
		2.4.3 �����ؿ�ĸ��ڵ����϶��µ��ҵ��������ڵ㣬����hasSubNode��voxelNum����ԭ������
		2.4.4 ����ҵ��������ڽڵ㣬����voxelNum���и�ֵ1

	���ڻ����һ�����˲������飬�˲�������ֵ�Ľڵ��д�����С�������Ҫ�������ѹ�������ǿ��԰���д�룬��Ϊ�˲�����ÿ�㿪���߳�
	���ң����ڵ�0��͵�һ�㣬���ǿ���ֱ��ʹ��д�룬������Ҫ�����жϣ����Ե�0��1��2����ͬʱд�롣
	Ȼ���3����512���̣߳�Ҳ����ʹ��һ���߳���д�롣����֮���ÿ�㶼��Ҫ����߳��飬���������޷���֤�������Ի���Ҫһ���˺���ȥ����

	2.5 compressSVO_Step1��˼·����ÿ8���ڵ�Ϊһ���飬������м�������ֵ����һ��ϵ�G����ֵ����ֵ��������ֵ��д��
		2.5.1 ����0��1��2��3��
			2.5.1.1 ���Ǵ���4���������
				a) blockHasValue��uint64_t��ÿһλ��ʾһ��8���ڵ�����ؿ��Ƿ���ֵ
				b) blockIndexOfGroup����ǰ���ؿ����߳����е�ƫ�ƣ�ͨ��ԭ�Ӽӵõ�
				c) nodeNum���߳���ڵ��������ڶ���߳������label
				d) blockNodeNum��64��uint�����飬���ڴ洢ÿ�����ؿ��м�����ֵ����
			2.5.1.2 ÿ���ڵ����ж��Լ���û��ֵ�������ֵ�����blockNodeNum��Ӧ������ԭ�Ӽ�
			2.5.1.3 ÿ����ֵ���ؿ��blockHasValue����Ӧλ���и�1
			2.5.1.4 �߳����ȡblockHasValue��1������blockNum���������߳����м�����ֵ���ؿ飬Ȼ���subArrayNum����ԭ�Ӽӣ��õ��߳�������ؿ���nodePool��
					����ʼ����blockIndexOfGroup
			2.5.1.5 ÿ���̻߳�ȡ�����������ؿ��ڸò��������ؿ��е�����blockIndex
			2.5.1.6 ����ڵ���ֵ����ʼ����ڵ��label��label=������ǰ�����ֵ���ؿ����ֵ�ڵ���+���������ؿ��е���ֵ���ص�����
			2.5.1.7 ����shuffleKey����󽫽ڵ㸳��nodePool��[1 + blockIndex * 8 + threadOffsetInBlock�����ؿ��е�������
			2.5.1.8 ����ǵ�һ�������˺�������Ե�0��1����и�ֵ
		2.5.2 ����ʣ�µĲ㣬�󲿷ֶ���2.5.1��ͬ������˵һ��������ͬ
			2.5.2.1 ����ڵ���ֵ�����nodeNum����ԭ�Ӽӣ�Ȼ���߳����threadBlockInfos����λ�ô洢FzbNodePoolBlock(blockIndexOfGroup, blockNum, nodeNum)
			2.5.2.2 ÿ���߳����blockStartIndexʹ��ԭ����С������С��blockIndexOfGroupд�룬�õ��ò�Ŀ����ʼ����
			2.5.2.3 ��󽫽ڵ����temNodePool��������һ������ʹ��
	2.6 ���ڶ���߳��飬���ǻ���Ҫ�����������compressSVO_Step2��Ϊÿ���ڵ㿪��һ���߳�
		2.6.1 ��������������� 
			a) threadBlockInfo�����ڴ洢���߳�����compressSVO_Step1�д洢��threadBlockInfos
			b) firstBlockIndex��blockStartIndex
		2.6.2 ÿ���߳���ͨ��threadBlockInfos����������������и�ֵ
		2.6.3 ��Ϊÿ���߳���ֻ��һ�������ؿ�����ֵ�ģ���ô���Ƕ��ڲ�����Щ���ؿ����ֵ�ڵ��ֱ��return�ͺ���
		2.6.4 ��ֵ�ڵ����ҵ�����tempNodePool�е�����
		2.6.5 ����ÿ��֮ǰ���߳��飬�õ����ǵ���ֵ���ؿ������blockNum*8�ۼӣ�Ȼ�����firstBlockIndex*8+1�õ���ǰ�߳������ʼ������
			  ��ô���ýڵ���nodePool�е���������1 + firstBlockIndex * 8 + �ۼ�(blockNum * 8) + threadLocalIndex�ˣ���node���Ƶ�nodePool�Ķ�Ӧλ�ü���
3. Ȼ��getSVOCuda�������õ���nodePool��svoVoxelValueArray���Ƶ�vulkan��������buffer�У�ɾ������buffer�ͱ�����
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

	uint32_t nodeBlockNum;	//8�����ݵ�block����������Ҫ�����1��ʾ���ڵ�
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//��������Ҫ�Ľڵ�����
	FzbVoxelValue* svoVoxelValueArray;	//��������Ҫ����������


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif