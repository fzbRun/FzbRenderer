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

	//底层加速结构
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
		geometry.geometry.triangles.vertexFormat = vertexFormat;	//这里表示顶点缓冲中需要的数据格式，配合vertexStride使用。如顶点缓冲中顶点有pos和normal，
		geometry.geometry.triangles.vertexStride = vertexStride;	//那么vertexFormat设置VK_FORMAT_R32G32B32_SFLOAT，vertexStride设置sizeof(顶点）则可以只取pos
		geometry.geometry.triangles.maxVertex = maxVertexCount;		//最大顶点索引值，即顶点数-1
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

		VkAccelerationStructureGeometryKHR* geometry = &geometries[triangleUUID].geometry;	//这里UUID实际上就是index
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

	//顶层加速结构
	uint64_t addInstanceGeometry(FzbBuffer& instanceBuffer, uint32_t instanceCount,
		uint32_t transformOffset = 0, VkGeometryFlagsKHR flags = VK_GEOMETRY_OPAQUE_BIT_KHR) {

		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.flags = flags;		//VK_GEOMETRY_OPAQUE_BIT_KHR 表示实例是否不透明
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.arrayOfPointers = VK_FALSE;	//表示实例直接存储数据还是存储指针，指向实际数据；这里我们需要指向底层加速结构
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
	我们需要通过vkCmdBuildAccelerationStructuresKHR填充加速结构，这个函数需要用到两个东西
	1. 加速结构的创建数据信息buildGeometryInfo
	2. 加速结构的数据创建范围accelerationStructureBuildRangeInfos

	对于buildGeometryInfo，这个结构体实例相当复杂，但是主要需要
	1. 空的加速结构（需要加速结构具体数据以及数量（这个数量包括几何数量和几何所包含的基元的数量））
	2. 临时数据缓冲

	为了获得加速结构具体数据，我们需要将所有需要创建或更新的几何添加到accelerationStructureGeometries数组中
	并且将每个几何的片元数量添加到primitiveCounts数组中
	这样就能得到加速结构具体数据以及数量了。
	
	同时在添加几何时我们也可以将几何的范围信息添加到accelerationStructureBuildRangeInfos中，这表明了加速结构在顶点、索引和变换数据中的范围和偏移。
	
	然后我们通过vkGetAccelerationStructureBuildSizesKHR和现有的buildGeometryInfo得到加速结构所需要的缓冲区大小（包括实际数据和临时数据），存储在buildSizesInfo中
	根据buildSizesInfo的实际数据缓冲大小，创建出加速结构缓冲区asBuffer
	然后根据asBuffer创建出实际的加速结构accelerationStructure，注意此时加速结构accelerationStructure是空的，需要我们通过临时数据和vkCmdBuildAccelerationStructuresKHR填充
	
	我们需要根据buildSizesInfo中的临时缓冲大小创建出临时缓冲区，满足填充加速结构的需要。
	并将临时缓冲scratchBuffer的地址传给buildGeometryInfo。
	
	这样，我们就可以调用vkCmdBuildAccelerationStructuresKHR填充得到加速结构了。
	*/
	void build(
		VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
		VkBuildAccelerationStructureModeKHR mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {

		if(geometries.empty())
			throw std::runtime_error("加速结构是空的，没法创建！");

		std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometries;	//需要更新或创建的几何集合
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos;	//需要更新的几何数据范围
		std::vector<uint32_t> primitiveCounts;
		for (auto& geometry : geometries) {
			//如果只需要更新动态物体，则使用更新模式，可以减少内分分配，加快数十倍
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
		//如果此时加速结构已经存在，需要更新他，则同时将其设为源和目的
		if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && accelerationStructure != VK_NULL_HANDLE) {
			buildGeometryInfo.srcAccelerationStructure = accelerationStructure;
			buildGeometryInfo.dstAccelerationStructure = accelerationStructure;
		}
		buildGeometryInfo.geometryCount = static_cast<uint32_t>(accelerationStructureGeometries.size());
		buildGeometryInfo.pGeometries = accelerationStructureGeometries.data();

		buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		//创建加速结构所需的最终占用内存和临时内存，会为未来的增量更新预留额外空间。
		vkGetAccelerationStructureBuildSizesKHR(logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, primitiveCounts.data(), &buildSizesInfo);

		//如果当前加速结构buffer不存在或者大小与计算的新的大小不同，则说明需要重新创建buffer
		if (!asBuffer.buffer || asBuffer.size != buildSizesInfo.accelerationStructureSize) {
			fzbCreateASBuffer(buildSizesInfo.accelerationStructureSize);

			VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
			accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			accelerationStructureCreateInfo.buffer = asBuffer.buffer;
			accelerationStructureCreateInfo.size = asBuffer.size;
			accelerationStructureCreateInfo.type = type;

			if (vkCreateAccelerationStructureKHR(logicalDevice, &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS) {
				throw std::runtime_error("创建加速结构失败");
			}
		}

		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
		accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddressInfo.accelerationStructure = accelerationStructure;
		deviceAddress = GetAccelerationStructureDeviceAddressKHR(logicalDevice, &accelerationDeviceAddressInfo);

		//创建加速结构需要一些临时数据，因此需要创建临时缓冲区
		fzbCreateASScratchBuffer(buildSizesInfo.buildScratchSize);
		fzbGetBufferDeviceAddress(logicalDevice, scratchBuffer);	//获得临时缓冲区设备地址

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
	VkQueue graphicsQueue;	//图像管线默认支持光线追踪

	VkAccelerationStructureKHR accelerationStructure{ VK_NULL_HANDLE };
	uint64_t deviceAddress{ 0 };	//加速结构的地址，用于着色器中使用
	VkAccelerationStructureTypeKHR type{};	//当前加速结构是top还是bottom
	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo{};	//当前加速结构缓冲区的大小，包括数据缓冲大小和临时缓冲大小
	
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
		//VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT表示可以在shader中使用缓冲区的地址来进行一些操作，如原子运算，而不需要通过描述符
		//描述符的有点是资源大小固定，因此硬件高度优化，并且可以减少描述符资源的大小（不能增大，如需增大，则要重新创建）；而设备地址则可以适应动态资源。
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