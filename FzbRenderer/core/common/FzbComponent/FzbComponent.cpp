#include "./FzbComponent.h"
#include "../FzbRenderer.h"
#include <stdexcept>

void FzbComponent::fzbCreateCommandBuffers(uint32_t bufferNum) {
	//我们现在想像流水线一样绘制，所以需要多个指令缓冲区
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = FzbRenderer::globalData.commandPool;
	//VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以提交到队列执行，但不能从其他命令缓冲区调用。
	//VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接提交，但可以从主命令缓冲区调用
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = bufferNum;	//指定分配的命令缓冲区是主命令缓冲区还是辅助命令缓冲区的大小

	//这个函数的第三个参数可以是单个命令缓冲区指针也可以是数组
	this->commandBuffers.resize(bufferNum);
	if (vkAllocateCommandBuffers(FzbRenderer::globalData.logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate shadow command buffers!");
	}
}
void FzbComponent::clean() {
	if(descriptorPool) vkDestroyDescriptorPool(FzbRenderer::globalData.logicalDevice, descriptorPool, nullptr);
}


