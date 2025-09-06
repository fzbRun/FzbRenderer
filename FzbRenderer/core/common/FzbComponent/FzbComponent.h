#pragma once

#include "../StructSet.h"
#include "../FzbBuffer/FzbBuffer.h"

#ifndef FZB_COMPONENT
#define FZB_COMPONENT

struct FzbComponent {
public:
	std::vector<VkCommandBuffer> commandBuffers;
	VkDescriptorPool descriptorPool = nullptr;
//---------------------------------------º¯Êý-----------------------------
	void fzbCreateCommandBuffers(uint32_t bufferNum = 1);
	virtual void clean();
};
#endif