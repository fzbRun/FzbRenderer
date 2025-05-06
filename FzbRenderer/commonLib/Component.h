#pragma once

#include "StructSet.h"

#ifndef COMPONENT_H
#define COMPONENT_H

struct ComponentSetting{};

class Component {

public:

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkQueue graphicsQueue;
	VkExtent2D swapChainExtent;
	MyModel* model;
	std::unique_ptr<FzbImage> fzbImage;
	std::unique_ptr<FzbBuffer> fzbBuffer;
	std::unique_ptr<FzbDescriptor> fzbDescriptor;
	std::unique_ptr<FzbPipeline> fzbPipeline;
	std::unique_ptr<FzbSync> fzbSync;

	virtual void activate() = 0;

	virtual void clean() = 0;

};

/*
class ComponentSet {

public:
	std::vector<const char*> instanceExtensions;
	std::vector<const char*> deviceExtensions;
	VkPhysicalDeviceFeatures deviceFeatures;
	//std::map<std::string, Component*> components;
	std::unique_ptr<FzbSVO> fzbSVO;

	template<typename T>
	void addInstanceExtensions(ComponentSetting* componentSetting) {
		T::getInstanceExtensions(componentSetting, this->instanceExtensions);
	}

	template<typename T>
	void addDeviceExtensions(ComponentSetting* componentSetting) {
		T::getDeviceExtensions(componentSetting, this->deviceExtensions);
	}

	template<typename T>
	void addDeviceFeatures(ComponentSetting* componentSetting) {
		T::getDeviceFeatures(componentSetting, this->deviceFeatures);
	}

	template<typename T>
	void addComponents(std::unique_ptr<FzbDevice>& fzbDevice, std::unique_ptr<FzbSwapchain>& fzbSwapchain, VkCommandPool& commandPool, MyModel* model, ComponentSetting* componentSetting) {
		std::string componentName = typeid(T).name();
		componentName = componentName.substr(componentName.find(" ") + 1);
		this->components.insert({ componentName, new T(fzbDevice, fzbSwapchain, commandPool, model, componentSetting) });
	}

	void activateComponents() {
		fzbSVO->activate();
	}

	void cleanComponents() {
		fzbSVO->clean();
	}

private:

};
*/
#endif
