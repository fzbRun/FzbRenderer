#pragma once

#include "../../../common/StructSet.h"
#include "../../../common/FzbBuffer.h"
#include "vma/vk_mem_alloc.h"
#include <map>
#include <stdexcept>

#ifndef AS_H
#define AS_H

VkDeviceAddress GetAccelerationStructureDeviceAddressKHR(VkDevice device, VkAccelerationStructureDeviceAddressInfoKHR* info) {
	auto func = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
	if (func != nullptr) {
		func(device, info);
	}
}

class FzbAccelerationStructure {

public:

	FzbAccelerationStructure(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkAccelerationStructureTypeKHR type) {
		this->logicalDevice = logicalDevice;
		this->physicalDevice = physicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
		this->type = type;
	}

	void clean() {
		if (accelerationStructure != VK_NULL_HANDLE) {
			vkDestroyAccelerationStructureKHR(logicalDevice, accelerationStructure, nullptr);
		}
	}

	//�ײ���ٽṹ
	uint64_t addTriangleGeometry(FzbBuffer& vertexBuffer, FzbBuffer& indexBuffer, FzbBuffer& transformBuffer,
		uint32_t triangleCount, uint32_t maxVertexCount, VkDeviceSize vertexStride, uint32_t transformOffset = 0,
		VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, VkIndexType indexType = VK_INDEX_TYPE_UINT32,
		VkGeometryFlagBitsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR, uint64_t vertexBufferDataAddress = 0,
		uint64_t indexBufferDataAddress = 0, uint64_t transformBufferDataAddress = 0) {

		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.flags = flags;
		geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry.geometry.triangles.vertexFormat = vertexFormat;	//�����ʾ���㻺������Ҫ�����ݸ�ʽ�����vertexStrideʹ�á��綥�㻺���ж�����pos��normal��
		geometry.geometry.triangles.vertexStride = vertexStride;	//��ôvertexFormat����VK_FORMAT_R32G32B32_SFLOAT��vertexStride����sizeof(���㣩�����ֻȡpos
		geometry.geometry.triangles.maxVertex = maxVertexCount;		//��󶥵�����ֵ����������-1
		geometry.geometry.triangles.indexType = indexType;
		geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferDataAddress == 0 ? vertexBuffer.deviceAddress : vertexBufferDataAddress;
		geometry.geometry.triangles.indexData.deviceAddress = indexBufferDataAddress == 0 ? indexBuffer.deviceAddress : indexBufferDataAddress;
		geometry.geometry.triangles.transformData.deviceAddress = transformBufferDataAddress == 0 ? transformBuffer.deviceAddress : transformBufferDataAddress;

		uint64_t index = geometries.size();
		geometries.insert({ index, {geometry, triangleCount, transformOffset} });
		return index;

	}

	void updateTriangleGeometry(uint64_t triangleUUID, FzbBuffer& vertexBuffer, FzbBuffer& indexBuffer, FzbBuffer& transformBuffer,
		uint32_t triangleCount, uint32_t maxVertexCount, VkDeviceSize vertexStride, uint32_t transformOffset = 0,
		VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, VkIndexType indexType = VK_INDEX_TYPE_UINT32,
		VkGeometryFlagBitsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR, uint64_t vertexBufferDataAddress = 0,
		uint64_t indexBufferDataAddress = 0, uint64_t transformBufferDataAddress = 0) {

		VkAccelerationStructureGeometryKHR* geometry = &geometries[triangleUUID].geometry;	//����UUIDʵ���Ͼ���index
		geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry->flags = flags;
		geometry->geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry->geometry.triangles.vertexFormat = vertexFormat;
		geometry->geometry.triangles.maxVertex = maxVertexCount;
		geometry->geometry.triangles.vertexStride = vertexStride;
		geometry->geometry.triangles.indexType = indexType;
		geometry->geometry.triangles.vertexData.deviceAddress = vertexBufferDataAddress == 0 ? vertexBuffer.deviceAddress : vertexBufferDataAddress;
		geometry->geometry.triangles.indexData.deviceAddress = indexBufferDataAddress == 0 ? indexBuffer.deviceAddress : indexBufferDataAddress;
		geometry->geometry.triangles.transformData.deviceAddress = transformBufferDataAddress == 0 ? transformBuffer.deviceAddress : transformBufferDataAddress;
		
		geometries[triangleUUID].primitiveCount = triangleCount;
		geometries[triangleUUID].transformOffset = transformOffset;
		geometries[triangleUUID].updated = true;

	}

	//������ٽṹ
	uint64_t addInstanceGeometry(FzbBuffer& instanceBuffer, uint32_t instanceCount,
		uint32_t transformOffset = 0, VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR) {

		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.flags = flags;		//VK_GEOMETRY_OPAQUE_BIT_KHR ��ʾʵ���Ƿ�͸��
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.arrayOfPointers = VK_FALSE;	//��ʾʵ��ֱ�Ӵ洢���ݻ��Ǵ洢ָ�룬ָ��ʵ�����ݣ�����������Ҫָ��ײ���ٽṹ
		geometry.geometry.instances.data.deviceAddress = instanceBuffer.deviceAddress;

		uint64_t index = geometries.size();
		geometries.insert({ index, {geometry, instanceCount, transformOffset} });
		return index;

	}

	void updateInstanceGeometry(uint64_t instanceUUID, FzbBuffer& instanceBuffer, uint32_t instanceCount,
		uint32_t transformOffset = 0, VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR) {
		VkAccelerationStructureGeometryKHR* geometry = &geometries[instanceUUID].geometry;
		geometry->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry->flags = flags;
		geometry->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry->geometry.instances.arrayOfPointers = VK_FALSE;
		geometry->geometry.instances.data.deviceAddress = instanceBuffer.deviceAddress;
		geometries[instanceUUID].primitiveCount = instanceCount;
		geometries[instanceUUID].transformOffset = transformOffset;
		geometries[instanceUUID].updated = true;
	}

	/*
	������Ҫͨ��vkCmdBuildAccelerationStructuresKHR�����ٽṹ�����������Ҫ�õ���������
	1. ���ٽṹ�Ĵ���������ϢbuildGeometryInfo
	2. ���ٽṹ�����ݴ�����ΧaccelerationStructureBuildRangeInfos

	����buildGeometryInfo������ṹ��ʵ���൱���ӣ�������Ҫ��Ҫ
	1. �յļ��ٽṹ����Ҫ���ٽṹ���������Լ���������������������������ͼ����������Ļ�Ԫ����������
	2. ��ʱ���ݻ���

	Ϊ�˻�ü��ٽṹ�������ݣ�������Ҫ��������Ҫ��������µļ�����ӵ�accelerationStructureGeometries������
	���ҽ�ÿ�����ε�ƬԪ������ӵ�primitiveCounts������
	�������ܵõ����ٽṹ���������Լ������ˡ�
	
	ͬʱ����Ӽ���ʱ����Ҳ���Խ����εķ�Χ��Ϣ��ӵ�accelerationStructureBuildRangeInfos�У�������˼��ٽṹ�ڶ��㡢�����ͱ任�����еķ�Χ��ƫ�ơ�
	
	Ȼ������ͨ��vkGetAccelerationStructureBuildSizesKHR�����е�buildGeometryInfo�õ����ٽṹ����Ҫ�Ļ�������С������ʵ�����ݺ���ʱ���ݣ����洢��buildSizesInfo��
	����buildSizesInfo��ʵ�����ݻ����С�����������ٽṹ������asBuffer
	Ȼ�����asBuffer������ʵ�ʵļ��ٽṹaccelerationStructure��ע���ʱ���ٽṹaccelerationStructure�ǿյģ���Ҫ����ͨ����ʱ���ݺ�vkCmdBuildAccelerationStructuresKHR���
	
	������Ҫ����buildSizesInfo�е���ʱ�����С��������ʱ�����������������ٽṹ����Ҫ��
	������ʱ����scratchBuffer�ĵ�ַ����buildGeometryInfo��
	
	���������ǾͿ��Ե���vkCmdBuildAccelerationStructuresKHR���õ����ٽṹ�ˡ�
	*/
	void build(
		VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
		VkBuildAccelerationStructureModeKHR mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {

		if(geometries.empty())
			throw std::runtime_error("���ٽṹ�ǿյģ�û��������");

		std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometries;	//��Ҫ���»򴴽��ļ��μ���
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos;	//��Ҫ���µļ������ݷ�Χ
		std::vector<uint32_t> primitiveCounts;
		for (auto& geometry : geometries) {
			//���ֻ��Ҫ���¶�̬���壬��ʹ�ø���ģʽ�����Լ����ڷַ��䣬�ӿ���ʮ��
			if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && !geometry.second.updated) {
				continue;
			}

			accelerationStructureGeometries.push_back(geometry.second.geometry);

			VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo;
			buildRangeInfo.primitiveCount = geometry.second.primitiveCount;
			buildRangeInfo.primitiveOffset = 0;
			buildRangeInfo.firstVertex = 0;
			buildRangeInfo.transformOffset = geometry.second.transformOffset;
			accelerationStructureBuildRangeInfos.push_back(buildRangeInfo);

			primitiveCounts.push_back(geometry.second.primitiveCount);

			geometry.second.updated = false;
		}

		VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
		buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildGeometryInfo.type = type;
		buildGeometryInfo.flags = flags;
		buildGeometryInfo.mode = mode;
		//�����ʱ���ٽṹ�Ѿ����ڣ���Ҫ����������ͬʱ������ΪԴ��Ŀ��
		if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && accelerationStructure != VK_NULL_HANDLE) {
			buildGeometryInfo.srcAccelerationStructure = accelerationStructure;
			buildGeometryInfo.dstAccelerationStructure = accelerationStructure;
		}
		buildGeometryInfo.geometryCount = static_cast<uint32_t>(accelerationStructureGeometries.size());
		buildGeometryInfo.pGeometries = accelerationStructureGeometries.data();

		buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		//�������ٽṹ���������ռ���ڴ����ʱ�ڴ棬��Ϊδ������������Ԥ������ռ䡣
		vkGetAccelerationStructureBuildSizesKHR(logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, primitiveCounts.data(), &buildSizesInfo);

		//�����ǰ���ٽṹbuffer�����ڻ��ߴ�С�������µĴ�С��ͬ����˵����Ҫ���´���buffer
		if (!asBuffer.buffer || asBuffer.size != buildSizesInfo.accelerationStructureSize) {
			fzbCreateASBuffer(buildSizesInfo.accelerationStructureSize);

			VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
			accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			accelerationStructureCreateInfo.buffer = asBuffer.buffer;
			accelerationStructureCreateInfo.size = asBuffer.size;
			accelerationStructureCreateInfo.type = type;

			if (vkCreateAccelerationStructureKHR(logicalDevice, &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS) {
				throw std::runtime_error("�������ٽṹʧ��");
			}
		}

		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
		accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddressInfo.accelerationStructure = accelerationStructure;
		deviceAddress = GetAccelerationStructureDeviceAddressKHR(logicalDevice, &accelerationDeviceAddressInfo);

		//�������ٽṹ��ҪһЩ��ʱ���ݣ������Ҫ������ʱ������
		fzbCreateASScratchBuffer(buildSizesInfo.buildScratchSize);
		fzbGetBufferDeviceAddress(logicalDevice, scratchBuffer);	//�����ʱ�������豸��ַ

		buildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;
		buildGeometryInfo.dstAccelerationStructure = accelerationStructure;

		VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);
		auto asBuildRangeInfos = &*accelerationStructureBuildRangeInfos.data();
		vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildGeometryInfo, &asBuildRangeInfos);
		endSingleTimeCommands(logicalDevice, commandPool, commandBuffer, graphicsQueue);

		fzbCleanBuffer(logicalDevice, &scratchBuffer);
	}

	VkAccelerationStructureKHR getAccelerationStructure() {
		return accelerationStructure;
	};

	const VkAccelerationStructureKHR* getAccelerationStructurePoint() {
		return &accelerationStructure;
	};

	uint64_t getDeviceAddress() {
		deviceAddress;
	};

private:

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;	//ͼ�����Ĭ��֧�ֹ���׷��

	VkAccelerationStructureKHR accelerationStructure{ VK_NULL_HANDLE };
	uint64_t deviceAddress{ 0 };	//���ٽṹ�ĵ�ַ��������ɫ����ʹ��
	VkAccelerationStructureTypeKHR type{};	//��ǰ���ٽṹ��top����bottom
	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo{};	//��ǰ���ٽṹ�������Ĵ�С���������ݻ����С����ʱ�����С
	
	struct Geometry {
		VkAccelerationStructureGeometryKHR geometry{};
		uint32_t primitiveCount;
		uint32_t transformOffset;
		bool updated = false;
	};

	FzbBuffer scratchBuffer;
	std::map<uint32_t, Geometry> geometries{};

	FzbBuffer asBuffer;

	void fzbCreateASBuffer(VkDeviceSize bufferSize, bool UseExternal = false) {
		asBuffer.size = bufferSize;
		//VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT��ʾ������shader��ʹ�û������ĵ�ַ������һЩ��������ԭ�����㣬������Ҫͨ��������
		//���������е�����Դ��С�̶������Ӳ���߶��Ż������ҿ��Լ�����������Դ�Ĵ�С��������������������Ҫ���´����������豸��ַ�������Ӧ��̬��Դ��
		fzbCreateBuffer(physicalDevice, logicalDevice, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asBuffer, UseExternal);
	}

	void fzbCreateASScratchBuffer(VkDeviceSize scratchBufferSize) {
		scratchBuffer.size = scratchBufferSize;
		fzbCreateBuffer(physicalDevice, logicalDevice,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			scratchBuffer);
	}

};

#endif