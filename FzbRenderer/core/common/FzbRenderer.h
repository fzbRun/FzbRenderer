#pragma once
#include "./FzbComponent/FzbComponent.h"
#include "./FzbComponent/FzbMainComponent.h"
#include "./FzbComponent/FzbFeatureComponent.h"
#include "./FzbComponent/FzbFeatureComponentManager.h"

#ifndef FZB_APPLICATION_H
#define FZB_APPLICATION_H

const uint32_t MAXFRAMECOUNT = 4096;

std::shared_ptr<FzbFeatureComponent> createFzbComponent(std::string componentName, pugi::xml_node& node);
class FzbRenderer {
public:
	std::string rendererName = "";
	inline static FzbMainComponent globalData = FzbMainComponent();
	inline static 	FzbFeatureComponentManager componentManager = FzbFeatureComponentManager();
	inline static bool useImageAvailableSemaphoreHandle = false;

	FzbRenderer(std::string rendererXML);
	void run();

private:
	FzbSemaphore imageAvailableSemaphore;
	VkFence fence;
	void initRendererFromXMLInfo(std::string rendererXML);
	void mainLoop();
	void drawFrame();
	void updateGlobalData();
	void clean();
};

#endif