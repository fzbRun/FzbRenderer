#include "./FzbComponent.h"
#include "../FzbRenderer.h"
#include <stdexcept>

void FzbComponent::fzbCreateCommandBuffers(uint32_t bufferNum) {
	//��������������ˮ��һ�����ƣ�������Ҫ���ָ�����
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = FzbRenderer::globalData.commandPool;
	//VK_COMMAND_BUFFER_LEVEL_PRIMARY�������ύ������ִ�У������ܴ���������������á�
	//VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ���ύ�������Դ��������������
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = bufferNum;	//ָ����������������������������Ǹ�����������Ĵ�С

	//��������ĵ��������������ǵ����������ָ��Ҳ����������
	this->commandBuffers.resize(bufferNum);
	if (vkAllocateCommandBuffers(FzbRenderer::globalData.logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate shadow command buffers!");
	}
}
void FzbComponent::clean() {
	if(descriptorPool) vkDestroyDescriptorPool(FzbRenderer::globalData.logicalDevice, descriptorPool, nullptr);
}


