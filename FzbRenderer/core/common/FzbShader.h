#pragma once

#include "StructSet.h"
#include "FzbPipeline.h"

#include <pugixml/src/pugixml.hpp>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <regex>

#ifndef FZB_SHADER_H
#define FZB_SHADER_H

//------------------------------------------------------------------------------------------------
glm::vec2 getfloat2FromString(std::string str) {
	std::vector<float> float2_array;
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, ' ')) {
		float2_array.push_back(std::stof(token));
	}
	return glm::vec2(float2_array[0], float2_array[1]);
}

glm::vec4 getRGBAFromString(std::string str) {
	std::vector<float> float4_array;
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, ',')) {
		float4_array.push_back(std::stof(token));
	}
	return glm::vec4(float4_array[0], float4_array[1], float4_array[2], float4_array[3]);
}
//------------------------------------------------------------------------------------------------
struct FzbTexture {
	std::string path = "";
	VkFilter filter = VK_FILTER_LINEAR;

	FzbTexture() {};
	FzbTexture(std::string path, VkFilter filter) {
		this->path = path;
		this->filter = filter;
	}

	bool operator==(const FzbTexture& other) const {	//只需要这两个就行
		return path == other.path && filter == other.filter;
	}
};

struct FzbNumberProperty {
	glm::vec4 value = glm::vec4(0.0f);

	FzbNumberProperty() {};
	FzbNumberProperty(glm::vec4 value) {
		this->value = value;
	}

	bool operator==(const FzbNumberProperty& other) const {	//只需要这两个就行
		return value == other.value;
	}
};

struct FzbShaderProperty {
	std::map<std::string, FzbTexture> textureProperties;
	std::map<std::string, FzbNumberProperty> numberProperties;

	bool operator==(const FzbShaderProperty& other) const {
		return textureProperties == other.textureProperties && numberProperties == other.numberProperties;
	}
};

struct FzbShader {

	VkDevice logicalDevice;

	std::string path;
	FzbShaderProperty properties;
	FzbVertexFormat vertexFormat;
	std::map<std::string, bool> macros;
	std::map<VkShaderStageFlagBits, std::string> shaders;

	FzbPipelineCreateInfo pipelineCreateInfo;
	VkPipelineLayout pipelineLayout = nullptr;
	VkPipeline pipeline = nullptr;

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;

	FzbShader() {}

	FzbShader(VkDevice logicalDevice, bool useNormal, bool useTexCoord, bool useTangent) {
		this->logicalDevice = logicalDevice;
	}

	FzbShader(VkDevice logicalDevice, std::string path) {
		this->logicalDevice = logicalDevice;
		this->path = path;
		pugi::xml_document doc;
		if (!doc.load_file((path + "/shaderInfo.xml").c_str())) {
			throw std::runtime_error("pugixml打开文件失败");
		}

		pugi::xml_node shaderInfos = doc.document_element();	//获取根节点，即<ShaderInfo>
		if (pugi::xml_node properties = shaderInfos.child("Properties")) {	//只有usability=true，才会加入map，其实可以用vector存储，但是不好使用string检索。
			for (pugi::xml_node property : properties.children("property")) {
				bool usability = std::string(property.attribute("usability").value()) == "true";
				std::string propertyName = property.attribute("name").value();
				std::string type = property.attribute("type").value();
				if (type == "texture") this->properties.textureProperties.insert({ propertyName, FzbTexture() });
				else if (type == "rgba") {
					glm::vec4 numberValue = getRGBAFromString(property.attribute("value").value());
					this->properties.numberProperties.insert({ propertyName, FzbNumberProperty(numberValue) });
				} 
				else {
					throw std::runtime_error("有新的shader属性类型，快些");
				}
			}
			/*
			for (pugi::xml_node texture : properties.children("texture")) {
				if (texture.attribute("usability").value() == "false") {
					continue;
				}
				FzbTexture texturePorperty;

				std::string type = texture.attribute("filter").value();
				if (type == "linear") {
					texturePorperty.filter = VK_FILTER_LINEAR;
				}
				else {
					texturePorperty.filter = VK_FILTER_NEAREST;
					throw std::runtime_error("不是linear的texture");
				}

				std::string stage = texture.attribute("stage").value();
				if (stage == "fragmentShader") {
					texturePorperty.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
				}
				else {
					texturePorperty.stage = VK_SHADER_STAGE_ALL;
					throw std::runtime_error("不是片元阶段的texture");
				}
				
				this->properties.textureProperties[texture.attribute("name").value()] = texturePorperty;
			}
			for (pugi::xml_node number : properties.children("rgba")) {
				if (number.attribute("usability").value() == "false") {
					continue;
				}
				FzbNumberProperty numberPorperty;

				std::string stage = number.attribute("stage").value();
				if (stage == "fragmentShader") {
					numberPorperty.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
				}
				else {
					numberPorperty.stage = VK_SHADER_STAGE_ALL;
					std::cout << "不是片元阶段的texture" << std::endl;
				}

				this->properties.numberProperties[number.attribute("name").value()] = numberPorperty;
			}
			*/
		}
		if (pugi::xml_node macros = shaderInfos.child("Macros")) {
			for (pugi::xml_node macro : macros.children("macro")) {
				this->macros.insert({ macro.attribute("name").value(),  std::string(macro.attribute("usability").value()) == "true" });
			}
		}
		initVertexFormat();
			
		if (pugi::xml_node shaders = shaderInfos.child("Shaders")) {
			for (pugi::xml_node shader : shaders.children("shader")) {
				std::string shaderStage = shader.attribute("name").value();
				if (shaderStage == "vertexShader") {
					this->shaders[VK_SHADER_STAGE_VERTEX_BIT] = shader.attribute("path").value();
					//this->shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, shader.attribute("path").value() });
				}
				else if (shaderStage == "fragmentShader") {
					this->shaders[VK_SHADER_STAGE_FRAGMENT_BIT] = shader.attribute("path").value();
				}
				else {
					throw std::runtime_error("新的shader阶段，快来写");
				}
			}
		}

		if (pugi::xml_node pipelineInfoNode = shaderInfos.child("Pipeline")) {
			if (pugi::xml_node screenSpaceNode = pipelineInfoNode.child("screenSpace")) {
				pipelineCreateInfo.screenSpace = true;
			}
			if (pugi::xml_node inputAssemblyInfoNode = pipelineInfoNode.child("inputAssemblyInfo")) {
				if (pugi::xml_node primitiveTopologyNode = inputAssemblyInfoNode.child("primitiveTopology")) {
					std::string primitiveTopologyValue = primitiveTopologyNode.attribute("value").value();
					if(primitiveTopologyValue == "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST") {
						pipelineCreateInfo.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
					}
					else if (primitiveTopologyValue == "VK_PRIMITIVE_TOPOLOGY_LINE_LIST") {
						pipelineCreateInfo.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
					}
					else {
						throw std::runtime_error("新的基元类型，快来写");
					}
				}
			}
			if (pugi::xml_node rasterizerInfoNode = pipelineInfoNode.child("rasterizerInfo")) {
				if (pugi::xml_node cullModeNode = rasterizerInfoNode.child("cullMode")) {
					std::string cullModeValue = cullModeNode.attribute("value").value();
					if (cullModeValue == "VK_CULL_MODE_BACK_BIT") {
						pipelineCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
					}
					else if (cullModeValue == "VK_CULL_MODE_FRONT_BIT") {
						pipelineCreateInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
					}
					else if (cullModeValue == "VK_CULL_MODE_NONE") {
						pipelineCreateInfo.cullMode = VK_CULL_MODE_NONE;
					}
					else {
						throw std::runtime_error("新的剔除类型，快来写");
					}
				}
				if (pugi::xml_node frontFaceNode = rasterizerInfoNode.child("frontFace")) {
					std::string frontFaceValue = frontFaceNode.attribute("value").value();
					if (frontFaceValue == "VK_FRONT_FACE_COUNTER_CLOCKWISE") {
						pipelineCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
					}
					else {
						pipelineCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
					}
				}
				for (pugi::xml_node rasterizerExtensionsNode : rasterizerInfoNode.children("rasterizerExtensions")) {
					std::string rasterizerExtensionsValue = rasterizerExtensionsNode.attribute("value").value();
					if (rasterizerExtensionsValue == "nullptr") {
						continue;
					}
					else if (rasterizerExtensionsValue == "保守光栅化") {
						float OverestimationSize = 0.5f;
						if (rasterizerExtensionsNode.attribute("OverestimationSize"))
							OverestimationSize = std::stof(rasterizerExtensionsNode.attribute("OverestimationSize").value());
						pipelineCreateInfo.conservativeState = getRasterizationConservativeState(OverestimationSize, pipelineCreateInfo.rasterizerExtensions);
						pipelineCreateInfo.rasterizerExtensions = &pipelineCreateInfo.conservativeState;
					}
					else {
						throw std::runtime_error("新的光栅化扩展，快来写");
					}
				}
				if (pugi::xml_node polyModeNode = rasterizerInfoNode.child("polyMode")) {
					std::string polyModeValue = polyModeNode.attribute("value").value();
					if (polyModeValue == "VK_POLYGON_MODE_FILL") {
						pipelineCreateInfo.polyMode = VK_POLYGON_MODE_FILL;
					}
					else if (polyModeValue == "VK_POLYGON_MODE_LINE") {
						pipelineCreateInfo.polyMode = VK_POLYGON_MODE_LINE;
					}
					else {
						throw std::runtime_error("新的填充类型，快来写");
					}
				}
				if (pugi::xml_node lineWidthNode = rasterizerInfoNode.child("lineWidth")) {
					pipelineCreateInfo.lineWidth = std::stof(lineWidthNode.attribute("value").value());
				}
			}
			if (pugi::xml_node multisamplingInfoNode = pipelineInfoNode.child("multisamplingInfo")) {
				if (pugi::xml_node sampleShadingEnableNode = multisamplingInfoNode.child("multisamplingInfo")) {
					std::string sampleShadingEnableValue = sampleShadingEnableNode.attribute("value").value();
					if (sampleShadingEnableValue == "true") pipelineCreateInfo.sampleShadingEnable = VK_TRUE;
				}
				if (pugi::xml_node sampleCountNode = multisamplingInfoNode.child("sampleCount")) {
					uint32_t sampleCountValue = std::stoul(sampleCountNode.attribute("value").value());
					switch (sampleCountValue) {
						case 1: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT; break;
						case 2: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_2_BIT; break;
						case 4: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_4_BIT; break;
						case 8: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_8_BIT; break;
						case 16: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_16_BIT; break;
						case 32: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_32_BIT; break;
						case 64: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_64_BIT; break;
						//case 4294967295: pipelineCreateInfo.sampleCount = VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM; break;
					}
				}
			}
			if (pugi::xml_node colorBlendingInfoNoe = pipelineInfoNode.child("colorBlendingInfo")) {
				for (pugi::xml_node colorBlendAttachmentNode : colorBlendingInfoNoe.children("colorBlendAttachment")) {
					if (colorBlendAttachmentNode.first_child()) {
						throw std::runtime_error("新的颜色附件类型，快来写");
					}
					else {
						pipelineCreateInfo.colorBlendAttachments.push_back(fzbCreateColorBlendAttachmentState());
					}
				}
			}
			if (pugi::xml_node depthStencilInfoNode = pipelineInfoNode.child("depthStencilInfo")) {
				if (pugi::xml_node depthTestEnableNode = depthStencilInfoNode.child("depthTestEnable")) {
					pipelineCreateInfo.depthTestEnable = depthTestEnableNode.attribute("value").value() == "true" ? VK_TRUE : VK_FALSE;
				}
				if (pugi::xml_node depthWriteEnableNode = depthStencilInfoNode.child("depthWriteEnable")) {
					pipelineCreateInfo.depthWriteEnable = depthWriteEnableNode.attribute("value").value() == "true" ? VK_TRUE : VK_FALSE;
				}
				if (pugi::xml_node depthCompareOpNode = depthStencilInfoNode.child("depthCompareOp")) {
					std::string depthCompareOpValue = depthCompareOpNode.attribute("value").value();
					if (depthCompareOpValue == "VK_COMPARE_OP_LESS") {
						pipelineCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
					}
					else {
						throw std::runtime_error("新的深度比较类型，快来写");
					}
				}
			}
			if (pugi::xml_node dynamicStateInfoNode = pipelineInfoNode.child("dynamicStateInfo")) {	//目前动态只关于视口
				if (pugi::xml_node dynamicViewNode = dynamicStateInfoNode.child("dynamicView")) {
					pipelineCreateInfo.dynamicView = VK_TRUE;
					pipelineCreateInfo.dynamicStates = {
						VK_DYNAMIC_STATE_VIEWPORT,
						VK_DYNAMIC_STATE_SCISSOR
					};
				}   
				else {
					pipelineCreateInfo.dynamicView = VK_FALSE;
				}
			}
			else if (pugi::xml_node viewportStateInfoNode = pipelineInfoNode.child("viewportStateInfo")) {
				for (pugi::xml_node viewportInfoNode : viewportStateInfoNode.children("viewport")) {
					VkViewport viewport;
					viewport.x = std::stof(viewportInfoNode.attribute("x").value());
					viewport.y = std::stof(viewportInfoNode.attribute("y").value());
					viewport.width = std::stof(viewportInfoNode.attribute("width").value());
					viewport.height = std::stof(viewportInfoNode.attribute("height").value());
					pipelineCreateInfo.viewports.push_back(viewport);
				}
				for (pugi::xml_node scissorInfoNode : viewportStateInfoNode.children("scissor")) {
					VkRect2D scissor;
					glm::vec2 offset = getfloat2FromString(scissorInfoNode.attribute("offset").value());
					scissor.offset = { (int)offset.x, (int)offset.y };
					glm::vec2 extent = getfloat2FromString(scissorInfoNode.attribute("extent").value());
					scissor.extent = { (uint32_t)extent.x, (uint32_t)extent.y };
					pipelineCreateInfo.scissors.push_back(scissor);
				}
				for (pugi::xml_node viewportExtensionNode : viewportStateInfoNode.children("viewportExtensions")) {
					std::string viewportExtensionValue = viewportExtensionNode.attribute("value").value();
					if (viewportExtensionValue == "nullptr") {
						continue;
					}
					else if (viewportExtensionValue == "swizzle") {
						for (pugi::xml_node swizzleNode : viewportExtensionNode.children("swizzles")) {
							std::vector<std::string> axis = {"x", "y", "z", "w"};
							std::vector<VkViewportCoordinateSwizzleNV> swizzleAxis(4);
							for (int i = 0; i < 4; i++) {
								switch (std::stoi(swizzleNode.attribute(axis[i]).value())) {
									case 0: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV; break;
									case 1: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV; break;
									case 2: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV; break;
									case 3: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV; break;
									case 4: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV; break;
									case 5: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV; break;
									case 6: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV; break;
									case 7: swizzleAxis[i] = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_W_NV; break;
								}
							}
							VkViewportSwizzleNV swizzle = { swizzleAxis[0], swizzleAxis[1], swizzleAxis[2], swizzleAxis[3] };
							pipelineCreateInfo.swizzles.push_back(swizzle);
						}
						pipelineCreateInfo.viewportSwizzleState = getViewportSwizzleState(pipelineCreateInfo.swizzles, pipelineCreateInfo.viewportExtensions);
						pipelineCreateInfo.viewportExtensions = &pipelineCreateInfo.viewportSwizzleState;
					}
					else {
						throw std::runtime_error("新的视口扩展，快来写");
					}
				}
			}
		}

	}

	void initVertexFormat() {
		if (macros["useVertexNormal"]) this->vertexFormat.useNormal = true;
		if (macros["useVertexTexCoords"]) this->vertexFormat.useTexCoord = true;
		if (macros["useVertexTangent"]) this->vertexFormat.useTangent = true;
	}
	
	void changeVertexFormat(FzbVertexFormat newFzbVertexFormat) {
		this->vertexFormat = newFzbVertexFormat;
		if (this->vertexFormat.useNormal) macros["useVertexNormal"] = true;
		if (this->vertexFormat.useTexCoord) macros["useVertexTexCoords"] = true;
		if (this->vertexFormat.useTangent) macros["useVertexTangent"] = true;
	}

	void clean() {
		if (pipeline) {
			vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
			vkDestroyPipeline(logicalDevice, pipeline, nullptr);
		}
	}

	void clear() {

		clean();
	}

//-----------------------------------------------------shader编译---------------------------------------------------------
	
// 递归处理包含文件
	std::string preprocessGLSL(
		const std::string& source,
		const std::filesystem::path& parentPath,
		std::set<std::filesystem::path>& includedFiles,
		int depth = 0)
	{
		if (depth > 32) {
			throw std::runtime_error("Include depth exceeded maximum limit (32)");
		}

		std::istringstream sourceStream(source);
		std::ostringstream output;
		std::string line;

		// 正则匹配 #include "filename" 或 #include <filename>
		std::regex includeRegex(R"(#include\s+["<]([^">]+)[">])");

		while (std::getline(sourceStream, line)) {
			std::smatch match;
			if (std::regex_search(line, match, includeRegex)) {
				std::string includeFilename = match[1].str();
				std::filesystem::path includePath;

				// 处理相对路径和绝对路径
				if (includeFilename[0] == '/') {
					// 绝对路径
					includePath = includeFilename;
				}
				else {
					// 相对路径 - 相对于父文件所在目录
					includePath = parentPath / includeFilename;
				}

				// 防止循环包含
				if (includedFiles.find(includePath) != includedFiles.end()) {
					output << "// [Skip already included: " << includeFilename << "]\n";
					continue;
				}

				// 标记为已包含
				includedFiles.insert(includePath);

				// 读取包含文件
				std::ifstream includeFile(includePath);
				if (!includeFile.is_open()) {
					throw std::runtime_error("Failed to open included file: " + includePath.string());
				}

				std::string includeContent(
					(std::istreambuf_iterator<char>(includeFile)),
					std::istreambuf_iterator<char>()
				);

				// 递归处理包含文件中的包含指令
				includeContent = preprocessGLSL(
					includeContent,
					includePath.parent_path(),
					includedFiles,
					depth + 1
				);

				// 添加包含文件内容
				output << "// [Begin include: " << includeFilename << "]\n";
				output << includeContent << "\n";
				output << "// [End include: " << includeFilename << "]\n";
			}
			else {
				// 不是包含指令的行直接输出
				output << line << "\n";
			}
		}

		return output.str();
	}

	// 入口函数
	std::string preprocessShaderFile(const std::string& filePath, uint32_t& version) {
		// 读取主文件
		std::ifstream mainFile(filePath);
		if (!mainFile.is_open()) {
			throw std::runtime_error("Failed to open shader file: " + filePath);
		}

		std::string mainContent(
			(std::istreambuf_iterator<char>(mainFile)),
			std::istreambuf_iterator<char>()
		);

		std::string version_str = mainContent.substr(0, mainContent.find('\n'));
		std::regex pattern("\\s*#version\\s+(\\d+)(\\s+|$)");	//加括号表示捕获
		std::smatch match;
		if (std::regex_match(version_str, match, pattern)) {
			version = std::stoi(match[1]);	//索引是从 1 开始获取捕获组的，而 0 索引保留给整个匹配的字符串
		}
		else {
			throw std::runtime_error("shader:" + filePath + " 版本不对");
		}

		// 处理包含
		std::set<std::filesystem::path> includedFiles;	//记录已经包含的文件，防止重复包含
		includedFiles.insert(std::filesystem::path(filePath)); // 添加主文件

		return preprocessGLSL(
			mainContent,
			std::filesystem::path(filePath).parent_path(),
			includedFiles
		);
	}

	std::vector<uint32_t> compileGLSL(
		const std::string& filePath,
		VkShaderStageFlagBits stage,
		const std::map<std::string, bool>& macros = {})
	{
		uint32_t version;
		// 1. 预处理（处理包含）
		std::string source = preprocessShaderFile(filePath, version);

		// 添加宏定义
		std::string versionString = source.substr(0, source.find_first_of("\n") + 1);
		std::string shaderMainContent = source.substr(source.find_first_of("\n") + 1);
		std::string processed = "";
		for (const auto& macro : macros) {
			if(macro.second) processed += "#define " + macro.first + "\n";
		}
		processed = versionString + processed + shaderMainContent;	//把宏加入glsl中

		//std::cout << (stage == VK_SHADER_STAGE_VERTEX_BIT ? "顶点着色器glsl：" : "片元着色器glsl：") << std::endl;
		//std::cout << processed << std::endl;

		// 初始化 glslang
		//glslang::InitializeProcess();	//初始化GLSLang库，全局初始化，在程序启动时调用一次

		// 设置着色器阶段
		std::string shaderStage;
		EShLanguage lang = EShLangVertex;
		switch (stage) {
		case VK_SHADER_STAGE_VERTEX_BIT:    lang = EShLangVertex; shaderStage = "顶点着色器";  break;
		case VK_SHADER_STAGE_FRAGMENT_BIT:  lang = EShLangFragment; shaderStage = "片元着色器"; break;
			// 添加其他阶段...
		}

		// 创建着色器对象
		glslang::TShader shader(lang);
		const char* str = processed.c_str();
		shader.setStrings(&str, 1);

		// 设置编译选项
		EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);	//EShMsgSpvRules表示生成spv，EShMsgVulkanRules为使用vulkan规范
		TBuiltInResource resources = {};
		resources.maxDrawBuffers = true;
		if (!shader.parse(&resources, version, false, messages)) {
			std::cout << processed << std::endl;
			throw std::runtime_error(shaderStage + " GLSL compilation failed:\n" +
				std::string(shader.getInfoLog()));
		}

		// 链接程序
		glslang::TProgram program;
		program.addShader(&shader);
		if (!program.link(messages)) {
			throw std::runtime_error("GLSL linking failed:\n" +
				std::string(program.getInfoLog()));
		}

		// 生成 SPIR-V
		std::vector<uint32_t> spirv;
		glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);

		// 清理资源
		//glslang::FinalizeProcess();
		return spirv;
	}

	std::vector<VkPipelineShaderStageCreateInfo> createShaderStates() {
		
		std::vector<VkPipelineShaderStageCreateInfo> shaderStates;
		for (auto& shader : shaders) {
			VkShaderStageFlagBits shaderStage = shader.first;
			std::string shaderPath = this->path + "/" + shader.second;
			std::vector<uint32_t> shaderSpvCode = compileGLSL(shaderPath, shaderStage, this->macros);

			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = shaderSpvCode.size() * sizeof(uint32_t);
			createInfo.pCode = shaderSpvCode.data();

			VkShaderModule shaderModule;
			if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
				throw std::runtime_error("failed to create shader module!");
			}

			VkPipelineShaderStageCreateInfo shaderStageInfo{};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = shaderStage;
			shaderStageInfo.module = shaderModule;
			shaderStageInfo.pName = "main";
			//允许指定着色器常量的值，比起在渲染时指定变量配置更加有效，因为可以通过编译器优化（没搞懂）
			shaderStageInfo.pSpecializationInfo = nullptr;

			shaderStates.push_back(shaderStageInfo);
		}

		return shaderStates;
	}

//-----------------------------------------------------创建pipeline------------------------------------------------------------
	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = this->createShaderStates();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		VkVertexInputBindingDescription inputBindingDescriptor;
		std::vector<VkVertexInputAttributeDescription> inputAttributeDescription;
		if (!pipelineCreateInfo.screenSpace) {
			inputBindingDescriptor = this->vertexFormat.getBindingDescription();
			inputAttributeDescription = this->vertexFormat.getAttributeDescriptions();
			vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		}
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(pipelineCreateInfo.primitiveTopology);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(pipelineCreateInfo);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo(pipelineCreateInfo.sampleShadingEnable, pipelineCreateInfo.sampleCount);
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { pipelineCreateInfo.colorBlendAttachments };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(pipelineCreateInfo);
		
		VkPipelineDynamicStateCreateInfo dynamicState{};
		VkPipelineViewportStateCreateInfo viewportState{};
		if (pipelineCreateInfo.dynamicView) {
			dynamicState = createDynamicStateCreateInfo(pipelineCreateInfo.dynamicStates);
		}
		else {
			viewportState = fzbCreateViewStateCreateInfo(pipelineCreateInfo.viewports, pipelineCreateInfo.scissors, pipelineCreateInfo.viewportExtensions);
		}

		pipelineLayout = fzbCreatePipelineLayout(logicalDevice, &descriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		if (pipelineCreateInfo.dynamicView) {
			pipelineInfo.pDynamicState = &dynamicState;
		}
		else {
			pipelineInfo.pViewportState = &viewportState;
		}
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = subPassIndex;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
		
	}

	bool operator==(const FzbShader& other) const {
		return shaders == other.shaders && macros == other.macros;	//宏不同就是不同的shader变体，属于不同的SRP Batch，其实还应该加上管线设置的，之后再说吧
	}

	//sbool operator=(const FzbShader& other) {
	//s	this->logicalDevice = other.logicalDevice;
	//s	this->vertexShader = other.vertexShader;
	//s	this->tessellationControlShader = other.tessellationControlShader;
	//s	this->tessellationEvaluateShader = other.tessellationEvaluateShader;
	//s	this->geometryShader = other.geometryShader;
	//s	this->fragmentShader = other.fragmentShader;
	//s	this->amplifyShader = other.amplifyShader;
	//s	this->meshShader = other.meshShader;
	//s	this->rayTracingShader = other.rayTracingShader;
	//s	this->vertexFormat = other.vertexFormat;
	//s	this->useFaceNormal = other.useFaceNormal;
	//s	this->albedoTexture = other.albedoTexture;
	//s	this->normalTexture = other.normalTexture;
	//s	this->materialTexture = other.materialTexture;
	//s	this->heightTexture = other.heightTexture;
	//s	this->pipelineLayout = other.pipelineLayout;
	//s	this->pipeline = other.pipeline;
	//s}
};

#endif