#include "./FzbShader.h"

#include <pugixml/src/pugixml.hpp>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <regex>
#include <sstream>
#include <iostream>
#include <filesystem>>

const TBuiltInResource DefaultTBuiltInResource = {
	/* .MaxLights = */ 32,
	/* .MaxClipPlanes = */ 6,
	/* .MaxTextureUnits = */ 32,
	/* .MaxTextureCoords = */ 32,
	/* .MaxVertexAttribs = */ 64,
	/* .MaxVertexUniformComponents = */ 4096,
	/* .MaxVaryingFloats = */ 64,
	/* .MaxVertexTextureImageUnits = */ 32,
	/* .MaxCombinedTextureImageUnits = */ 80,
	/* .MaxTextureImageUnits = */ 32,
	/* .MaxFragmentUniformComponents = */ 4096,
	/* .MaxDrawBuffers = */ 32,
	/* .MaxVertexUniformVectors = */ 128,
	/* .MaxVaryingVectors = */ 8,
	/* .MaxFragmentUniformVectors = */ 16,
	/* .MaxVertexOutputVectors = */ 16,
	/* .MaxFragmentInputVectors = */ 15,
	/* .MinProgramTexelOffset = */ -8,
	/* .MaxProgramTexelOffset = */ 7,
	/* .MaxClipDistances = */ 8,
	/* .MaxComputeWorkGroupCountX = */ 65535,
	/* .MaxComputeWorkGroupCountY = */ 65535,
	/* .MaxComputeWorkGroupCountZ = */ 65535,
	/* .MaxComputeWorkGroupSizeX = */ 1024,
	/* .MaxComputeWorkGroupSizeY = */ 1024,
	/* .MaxComputeWorkGroupSizeZ = */ 64,
	/* .MaxComputeUniformComponents = */ 1024,
	/* .MaxComputeTextureImageUnits = */ 16,
	/* .MaxComputeImageUniforms = */ 8,
	/* .MaxComputeAtomicCounters = */ 8,
	/* .MaxComputeAtomicCounterBuffers = */ 1,
	/* .MaxVaryingComponents = */ 60,
	/* .MaxVertexOutputComponents = */ 64,
	/* .MaxGeometryInputComponents = */ 64,
	/* .MaxGeometryOutputComponents = */ 128,
	/* .MaxFragmentInputComponents = */ 128,
	/* .MaxImageUnits = */ 8,
	/* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
	/* .MaxCombinedShaderOutputResources = */ 8,
	/* .MaxImageSamples = */ 0,
	/* .MaxVertexImageUniforms = */ 0,
	/* .MaxTessControlImageUniforms = */ 0,
	/* .MaxTessEvaluationImageUniforms = */ 0,
	/* .MaxGeometryImageUniforms = */ 0,
	/* .MaxFragmentImageUniforms = */ 8,
	/* .MaxCombinedImageUniforms = */ 8,
	/* .MaxGeometryTextureImageUnits = */ 16,
	/* .MaxGeometryOutputVertices = */ 256,
	/* .MaxGeometryTotalOutputComponents = */ 1024,
	/* .MaxGeometryUniformComponents = */ 1024,
	/* .MaxGeometryVaryingComponents = */ 64,
	/* .MaxTessControlInputComponents = */ 128,
	/* .MaxTessControlOutputComponents = */ 128,
	/* .MaxTessControlTextureImageUnits = */ 16,
	/* .MaxTessControlUniformComponents = */ 1024,
	/* .MaxTessControlTotalOutputComponents = */ 4096,
	/* .MaxTessEvaluationInputComponents = */ 128,
	/* .MaxTessEvaluationOutputComponents = */ 128,
	/* .MaxTessEvaluationTextureImageUnits = */ 16,
	/* .MaxTessEvaluationUniformComponents = */ 1024,
	/* .MaxTessPatchComponents = */ 120,
	/* .MaxPatchVertices = */ 32,
	/* .MaxTessGenLevel = */ 64,
	/* .MaxViewports = */ 16,
	/* .MaxVertexAtomicCounters = */ 0,
	/* .MaxTessControlAtomicCounters = */ 0,
	/* .MaxTessEvaluationAtomicCounters = */ 0,
	/* .MaxGeometryAtomicCounters = */ 0,
	/* .MaxFragmentAtomicCounters = */ 8,
	/* .MaxCombinedAtomicCounters = */ 8,
	/* .MaxAtomicCounterBindings = */ 1,
	/* .MaxVertexAtomicCounterBuffers = */ 0,
	/* .MaxTessControlAtomicCounterBuffers = */ 0,
	/* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
	/* .MaxGeometryAtomicCounterBuffers = */ 0,
	/* .MaxFragmentAtomicCounterBuffers = */ 1,
	/* .MaxCombinedAtomicCounterBuffers = */ 1,
	/* .MaxAtomicCounterBufferSize = */ 16384,
	/* .MaxTransformFeedbackBuffers = */ 4,
	/* .MaxTransformFeedbackInterleavedComponents = */ 64,
	/* .MaxCullDistances = */ 8,
	/* .MaxCombinedClipAndCullDistances = */ 8,
	/* .MaxSamples = */ 4,
	/* .maxMeshOutputVerticesNV = */ 256,
	/* .maxMeshOutputPrimitivesNV = */ 512,
	/* .maxMeshWorkGroupSizeX_NV = */ 32,
	/* .maxMeshWorkGroupSizeY_NV = */ 1,
	/* .maxMeshWorkGroupSizeZ_NV = */ 1,
	/* .maxTaskWorkGroupSizeX_NV = */ 32,
	/* .maxTaskWorkGroupSizeY_NV = */ 1,
	/* .maxTaskWorkGroupSizeZ_NV = */ 1,
	/* .maxMeshViewCountNV = */ 4,
	/* .maxMeshOutputVerticesEXT = */ 256,
	/* .maxMeshOutputPrimitivesEXT = */ 256,
	/* .maxMeshWorkGroupSizeX_EXT = */ 128,
	/* .maxMeshWorkGroupSizeY_EXT = */ 128,
	/* .maxMeshWorkGroupSizeZ_EXT = */ 128,
	/* .maxTaskWorkGroupSizeX_EXT = */ 128,
	/* .maxTaskWorkGroupSizeY_EXT = */ 128,
	/* .maxTaskWorkGroupSizeZ_EXT = */ 128,
	/* .maxMeshViewCountEXT = */ 4,
	/* .maxDualSourceDrawBuffersEXT = */ 1,

	/* .limits = */ {
		/* .nonInductiveForLoops = */ 1,
		/* .whileLoops = */ 1,
		/* .doWhileLoops = */ 1,
		/* .generalUniformIndexing = */ 1,
		/* .generalAttributeMatrixVectorIndexing = */ 1,
		/* .generalVaryingIndexing = */ 1,
		/* .generalSamplerIndexing = */ 1,
		/* .generalVariableIndexing = */ 1,
		/* .generalConstantMatrixVectorIndexing = */ 1,
	} };

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

//--------------------------------------------����shader-----------------------------------
std::string preprocessGLSL(
	const std::string& source,
	const std::filesystem::path& parentPath,
	std::set<std::filesystem::path>& includedFiles,
	int depth)
{
	if (depth > 32) {
		throw std::runtime_error("Include depth exceeded maximum limit (32)");
	}

	std::istringstream sourceStream(source);
	std::ostringstream output;
	std::string line;

	// ����ƥ�� #include "filename" �� #include <filename>
	std::regex includeRegex(R"(#include\s+["<]([^">]+)[">])");

	while (std::getline(sourceStream, line)) {
		std::smatch match;
		if (std::regex_search(line, match, includeRegex)) {
			std::string includeFilename = match[1].str();
			std::filesystem::path includePath;

			// �������·���;���·��
			if (includeFilename[0] == '/') {
				// ����·��
				includePath = includeFilename;
			}
			else {
				// ���·�� - ����ڸ��ļ�����Ŀ¼
				includePath = parentPath / includeFilename;
			}

			// ��ֹѭ������
			if (includedFiles.find(includePath) != includedFiles.end()) {
				output << "// [Skip already included: " << includeFilename << "]\n";
				continue;
			}

			// ���Ϊ�Ѱ���
			includedFiles.insert(includePath);

			// ��ȡ�����ļ�
			std::ifstream includeFile(includePath);
			if (!includeFile.is_open()) {
				throw std::runtime_error("Failed to open included file: " + includePath.string());
			}

			std::string includeContent(
				(std::istreambuf_iterator<char>(includeFile)),
				std::istreambuf_iterator<char>()
			);

			// �ݹ鴦������ļ��еİ���ָ��
			includeContent = preprocessGLSL(
				includeContent,
				includePath.parent_path(),
				includedFiles,
				depth + 1
			);

			// ��Ӱ����ļ�����
			output << "// [Begin include: " << includeFilename << "]\n";
			output << includeContent << "\n";
			output << "// [End include: " << includeFilename << "]\n";
		}
		else {
			// ���ǰ���ָ�����ֱ�����
			output << line << "\n";
		}
	}

	return output.str();
}
std::string preprocessShaderFile(const std::string& filePath, uint32_t& version) {
	// ��ȡ���ļ�
	std::ifstream mainFile(filePath);
	if (!mainFile.is_open()) {
		throw std::runtime_error("Failed to open shader file: " + filePath);
	}

	std::string mainContent(
		(std::istreambuf_iterator<char>(mainFile)),
		std::istreambuf_iterator<char>()
	);

	std::string version_str = mainContent.substr(0, mainContent.find('\n'));
	std::regex pattern("\\s*#version\\s+(\\d+)(\\s+|$)");	//�����ű�ʾ����
	std::smatch match;
	if (std::regex_match(version_str, match, pattern)) {
		version = std::stoi(match[1]);	//�����Ǵ� 1 ��ʼ��ȡ������ģ��� 0 ��������������ƥ����ַ���
	}
	else {
		throw std::runtime_error("shader:" + filePath + " �汾����");
	}

	// �������
	std::set<std::filesystem::path> includedFiles;	//��¼�Ѿ��������ļ�����ֹ�ظ�����
	includedFiles.insert(std::filesystem::path(filePath)); // ������ļ�

	return preprocessGLSL(
		mainContent,
		std::filesystem::path(filePath).parent_path(),
		includedFiles
	);
}
std::vector<uint32_t> compileGLSL(
	const std::string& filePath,
	VkShaderStageFlagBits stage,
	const std::map<std::string, bool>& macros)
{
	uint32_t version;
	// 1. Ԥ�������������
	std::string source = preprocessShaderFile(filePath, version);

	// ��Ӻ궨��
	std::string versionString = source.substr(0, source.find_first_of("\n") + 1);
	std::string shaderMainContent = source.substr(source.find_first_of("\n") + 1);
	std::string processed = "";
	for (const auto& macro : macros) {
		if (macro.second) processed += "#define " + macro.first + "\n";
	}
	processed = versionString + processed + shaderMainContent;	//�Ѻ����glsl��

	//std::cout << (stage == VK_SHADER_STAGE_VERTEX_BIT ? "������ɫ��glsl��" : "ƬԪ��ɫ��glsl��") << std::endl;
	//std::cout << processed << std::endl;

	// ��ʼ�� glslang
	//glslang::InitializeProcess();	//��ʼ��GLSLang�⣬ȫ�ֳ�ʼ�����ڳ�������ʱ����һ��

	// ������ɫ���׶�
	std::string shaderStage;
	EShLanguage lang = EShLangVertex;
	switch (stage) {
	case VK_SHADER_STAGE_VERTEX_BIT:    lang = EShLangVertex; shaderStage = "������ɫ��";  break;
	case VK_SHADER_STAGE_GEOMETRY_BIT:    lang = EShLangGeometry; shaderStage = "������ɫ��";  break;
	case VK_SHADER_STAGE_FRAGMENT_BIT:  lang = EShLangFragment; shaderStage = "ƬԪ��ɫ��"; break;
		// ��������׶�...
	}

	// ������ɫ������
	glslang::TShader shader(lang);
	const char* str = processed.c_str();
	shader.setStrings(&str, 1);

	// ���ñ���ѡ��
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);	//EShMsgSpvRules��ʾ����spv��EShMsgVulkanRulesΪʹ��vulkan�淶
	TBuiltInResource resources = DefaultTBuiltInResource;
	if (!shader.parse(&resources, version, false, messages)) {
		std::cout << processed << std::endl;
		throw std::runtime_error(shaderStage + " GLSL compilation failed:\n" +
			std::string(shader.getInfoLog()));
	}

	// ���ӳ���
	glslang::TProgram program;
	program.addShader(&shader);
	if (!program.link(messages)) {
		throw std::runtime_error("GLSL linking failed:\n" +
			std::string(program.getInfoLog()));
	}

	// ���� SPIR-V
	std::vector<uint32_t> spirv;
	glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);

	// ������Դ
	//glslang::FinalizeProcess();
	return spirv;
}

//--------------------------------------------shader-----------------------------------
FzbShader::FzbShader() {}
FzbShader::FzbShader(VkDevice logicalDevice, bool useNormal, bool useTexCoord, bool useTangent) {
	this->logicalDevice = logicalDevice;
}
FzbShader::FzbShader(VkDevice logicalDevice, std::string path) {
	this->logicalDevice = logicalDevice;
	this->path = path;
	pugi::xml_document doc;
	if (!doc.load_file((path + "/shaderInfo.xml").c_str())) {
		throw std::runtime_error("pugixml���ļ�ʧ��");
	}

	pugi::xml_node shaderInfos = doc.document_element();	//��ȡ���ڵ㣬��<ShaderInfo>
	if (pugi::xml_node properties = shaderInfos.child("Properties")) {	//ֻ��usability=true���Ż����map����ʵ������vector�洢�����ǲ���ʹ��string������
		for (pugi::xml_node property : properties.children("property")) {
			//bool usability = std::string(property.attribute("usability").value()) == "true";
			std::string propertyName = property.attribute("name").value();
			std::string type = property.attribute("type").value();
			if (type == "texture") this->properties.textureProperties.insert({ propertyName, FzbTexture() });
			else if (type == "rgba") {
				glm::vec4 numberValue = getRGBAFromString(property.attribute("value").value());
				this->properties.numberProperties.insert({ propertyName, FzbNumberProperty(numberValue) });
			}
			else {
				throw std::runtime_error("���µ�shader�������ͣ���Щ");
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
				throw std::runtime_error("����linear��texture");
			}

			std::string stage = texture.attribute("stage").value();
			if (stage == "fragmentShader") {
				texturePorperty.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			}
			else {
				texturePorperty.stage = VK_SHADER_STAGE_ALL;
				throw std::runtime_error("����ƬԪ�׶ε�texture");
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
				std::cout << "����ƬԪ�׶ε�texture" << std::endl;
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
	//initVertexFormat();

	if (pugi::xml_node shaders = shaderInfos.child("Shaders")) {
		for (pugi::xml_node shader : shaders.children("shader")) {
			std::string shaderStage = shader.attribute("name").value();
			if (shaderStage == "vertexShader") {
				this->shaders[VK_SHADER_STAGE_VERTEX_BIT] = shader.attribute("path").value();
				//this->shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, shader.attribute("path").value() });
			}
			else if (shaderStage == "geometryShader") {
				this->shaders[VK_SHADER_STAGE_GEOMETRY_BIT] = shader.attribute("path").value();
			}
			else if (shaderStage == "fragmentShader") {
				this->shaders[VK_SHADER_STAGE_FRAGMENT_BIT] = shader.attribute("path").value();
			}
			else {
				throw std::runtime_error("�µ�shader�׶Σ�����д");
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
				if (primitiveTopologyValue == "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST") {
					pipelineCreateInfo.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
				}
				else if (primitiveTopologyValue == "VK_PRIMITIVE_TOPOLOGY_LINE_LIST") {
					pipelineCreateInfo.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
				}
				else {
					throw std::runtime_error("�µĻ�Ԫ���ͣ�����д");
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
					throw std::runtime_error("�µ��޳����ͣ�����д");
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
				else if (rasterizerExtensionsValue == "���ع�դ��") {
					float OverestimationSize = 0.5f;
					if (rasterizerExtensionsNode.attribute("OverestimationSize"))
						OverestimationSize = std::stof(rasterizerExtensionsNode.attribute("OverestimationSize").value());
					pipelineCreateInfo.conservativeState = getRasterizationConservativeState(OverestimationSize, pipelineCreateInfo.rasterizerExtensions);
					pipelineCreateInfo.rasterizerExtensions = &pipelineCreateInfo.conservativeState;
				}
				else {
					throw std::runtime_error("�µĹ�դ����չ������д");
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
					throw std::runtime_error("�µ�������ͣ�����д");
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
					throw std::runtime_error("�µ���ɫ�������ͣ�����д");
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
					throw std::runtime_error("�µ���ȱȽ����ͣ�����д");
				}
			}
		}
		if (pugi::xml_node dynamicStateInfoNode = pipelineInfoNode.child("dynamicStateInfo")) {	//Ŀǰ��ֻ̬�����ӿ�
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
						std::vector<std::string> axis = { "x", "y", "z", "w" };
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
					throw std::runtime_error("�µ��ӿ���չ������д");
				}
			}
		}
	}

}
void FzbShader::clean() {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].clean();
	}
}
void FzbShader::createShaderVariant(FzbMaterial* material, FzbVertexFormat vertexFormat) {
	for (auto& materialPair : material->properties.textureProperties) {
		if (!this->properties.textureProperties.count(materialPair.first)) {
			std::string error = "����" + material->id + "����shaderû�е���Դ: " + materialPair.first;
			throw std::runtime_error(error);
		}
	}
	for (auto& materialPair : material->properties.numberProperties) {
		if (!this->properties.numberProperties.count(materialPair.first)) {
			std::string error = "����" + material->id + "����shaderû�е���Դ: " + materialPair.first;
			throw std::runtime_error(error);
		}
	}

	for (int i = 0; i < shaderVariants.size(); i++) {
		if (shaderVariants[i].properties.keyCompare(material->properties)) {
			shaderVariants[i].materials.push_back(material);
			material->vertexFormat = shaderVariants[i].vertexFormat;
			//material->shader = &shaderVariants[i];
			return;
		}
	}

	this->shaderVariants.push_back(FzbShaderVariant(this, material, vertexFormat));
	//material->shader = &this->shaderVariants[this->shaderVariants.size() - 1];
}
void FzbShader::createMeshBatch(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<FzbMesh>& sceneMeshSet) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].createMeshBatch(physicalDevice, commandPool, graphicsQueue, sceneMeshSet);
	}
}
void FzbShader::createDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage>& sceneImages) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].createDescriptor(sceneDescriptorPool, sceneImages);
	}
}
void FzbShader::createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, VkDescriptorSetLayout meshDescriptorSetLayout, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts_temp = descriptorSetLayouts;
		if (shaderVariants[i].descriptorSetLayout) descriptorSetLayouts_temp.push_back(shaderVariants[i].descriptorSetLayout);
		descriptorSetLayouts_temp.push_back(meshDescriptorSetLayout);
		shaderVariants[i].createPipeline(renderPass, subPassIndex, descriptorSetLayouts_temp);
	}
}
void FzbShader::render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets, VkExtent2D extent) {
	if (pipelineCreateInfo.dynamicView) {
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(extent.width);
		viewport.height = static_cast<float>(extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = extent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
	}
	
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].render(commandBuffer, componentDescriptorSets);
	}
}
bool FzbShader::operator==(const FzbShader& other) const {
	return path == other.path;
}

//--------------------------------------------����shader����-----------------------------------
void FzbShaderVariant::changeVertexFormatAndMacros(FzbVertexFormat vertexFormat) {
	this->vertexFormat.mergeUpward(vertexFormat);	//����˵һ��Ҫ��texCoords����������material��shader����û���������Կ��Բ�������

	if (this->properties.textureProperties.size() > 0) {
		this->vertexFormat.useTexCoord = true;
		this->macros["useTextureProperty"] = true;
		this->macros["useVertexTexCoords"] = true;
	}
	else {
		this->vertexFormat.useTexCoord = false;
		this->macros["useTextureProperty"] = false;
		this->macros["useVertexTexCoords"] = false;
	}
	this->macros["useNumberProperty"] = this->properties.numberProperties.size() == 0 ? false : true;
	if (macros["useVertexNormal"]) this->vertexFormat.useNormal = true;
	if (macros["useVertexTexCoords"]) this->vertexFormat.useTexCoord = true;
	if (macros["useVertexTangent"]) this->vertexFormat.useTangent = true;
}
FzbShaderVariant::FzbShaderVariant() {};
FzbShaderVariant::FzbShaderVariant(FzbShader* publicShader, FzbMaterial* material, FzbVertexFormat vertexFormat) {
	this->logicalDevice = publicShader->logicalDevice;
	this->publicShader = publicShader;
	this->macros = publicShader->macros;
	this->materials.push_back(material);
	this->vertexFormat = FzbVertexFormat();

	//���shader������ĳ����Դ���ͣ�����materialû�д�����Ӧ����Դ����ر�shader����Ӧ����Դ�ࣨ�꣩
	for (auto& shaderPair : publicShader->properties.textureProperties) {
		std::string textureType = shaderPair.first;
		std::string macro = textureType;
		macro[0] = std::toupper(static_cast<unsigned char>(macro[0]));
		macro = "use" + macro;
		if (material->properties.textureProperties.count(textureType)) {
			this->macros[macro] = true;
			this->properties.textureProperties.insert({ textureType, material->properties.textureProperties[textureType] });
		}
		else this->macros[macro] = false;
	}
	//number�е㲻ͬ�����materialû�д��룬��shader�����ˣ���ʹ��Ĭ��ֵ
	for (auto& shaderPair : publicShader->properties.numberProperties) {
		std::string numberType = shaderPair.first;
		std::string macro = numberType;
		macro[0] = std::toupper(static_cast<unsigned char>(macro[0]));
		macro = "use" + macro;
		if (material->properties.numberProperties.count(numberType)) {
			this->macros[macro] = true;
			this->properties.numberProperties.insert({ numberType, material->properties.numberProperties[numberType] });
		}
		else if (this->macros[macro]) this->properties.numberProperties.insert(shaderPair);
	}
	changeVertexFormatAndMacros(vertexFormat);
	material->vertexFormat = this->vertexFormat;
}
void FzbShaderVariant::createDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage>& sceneImages) {
	//����������
	uint32_t textureNum = this->properties.textureProperties.size();
	uint32_t numberNum = this->properties.numberProperties.size() > 0 ? 1 : 0;	//������ֵ������һ��uniformBuffer����
	if (textureNum + numberNum == 0) {
		return;
	}
	std::vector<VkDescriptorType> type(textureNum + numberNum);
	std::vector<VkShaderStageFlags> stage(textureNum + numberNum);
	for (int i = 0; i < textureNum; i++) {
		type[i] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		stage[i] = VK_SHADER_STAGE_ALL;	//shaderStage����Ϊall���ص�shader����Ӱ�����ݵĶ�ȡ�ٶȣ�ֻ���ڱ���ʱ���ⷶΧ��ͬ���ѣ�Ӱ������ٶȶ��ѣ���
	}
	if (numberNum) {
		type[textureNum] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		stage[textureNum] = VK_SHADER_STAGE_ALL;
	}
	this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, type.size(), type, stage);
	for (int i = 0; i < materials.size(); i++) {
		materials[i]->createMaterialDescriptor(sceneDescriptorPool, this->descriptorSetLayout, sceneImages);
	}
}
/*
void FzbShaderVariant::changeVertexFormat(FzbVertexFormat newFzbVertexFormat) {
	this->vertexFormat = newFzbVertexFormat;
	if (this->vertexFormat.useNormal) macros["useVertexNormal"] = true;
	if (this->vertexFormat.useTexCoord) macros["useVertexTexCoords"] = true;
	if (this->vertexFormat.useTangent) macros["useVertexTangent"] = true;
}
*/
void FzbShaderVariant::clean() {
	if (pipeline) {
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, pipeline, nullptr);
	}
	vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	meshBatch.clean();
}
void FzbShaderVariant::createMeshBatch(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<FzbMesh>& sceneMeshSet) {
	/*
	������е�mesh��material����ͬ�Ļ��߸�����materialsֻ��һ��material����meshBatch��materialsֻ��Ҫ�洢һ��material���ɣ�������useSameMaterial=true
	*/
	this->meshBatch = FzbMeshBatch(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	std::unordered_set<FzbMaterial*> materialSet(this->materials.begin(), this->materials.end());
	std::map<FzbMaterial*, FzbMaterial*> meshMaterials;
	std::vector<FzbMaterial*> meshBatchMaterials;
	for (size_t i = 0; i < sceneMeshSet.size(); i++) {
		FzbMesh& mesh = sceneMeshSet[i];
		FzbMaterial* material = mesh.material;
		if (materialSet.find(material) != materialSet.end()) {
			this->meshBatch.meshes.push_back(&mesh);
			meshBatchMaterials.push_back(material);
			if (!meshMaterials.count(material)) {	//�����в��ظ���material�������У�������ֻ��һ������ô˵������mesh����һ��material
				meshMaterials.insert({ material, material });
			}
		}
	}
	if (meshMaterials.size() == 1) {	//˵������mesh����һ��material������Ը���
		this->meshBatch.materials.push_back(meshBatchMaterials[0]);
		this->meshBatch.useSameMaterial = true;
	}
	else this->meshBatch.materials = meshBatchMaterials;
}
std::vector<VkPipelineShaderStageCreateInfo> FzbShaderVariant::createShaderStates() {

	std::vector<VkPipelineShaderStageCreateInfo> shaderStates;
	for (auto& shader : publicShader->shaders) {
		VkShaderStageFlagBits shaderStage = shader.first;
		std::string shaderPath = publicShader->path + "/" + shader.second;
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
		//����ָ����ɫ��������ֵ����������Ⱦʱָ���������ø�����Ч����Ϊ����ͨ���������Ż���û�㶮��
		shaderStageInfo.pSpecializationInfo = nullptr;

		shaderStates.push_back(shaderStageInfo);
	}

	return shaderStates;
}
void FzbShaderVariant::createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
	//descriptorSetLayouts.push_back(meshDescriptorSetLayout);

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages = this->createShaderStates();

	FzbPipelineCreateInfo pipelineCreateInfo = publicShader->pipelineCreateInfo;

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	VkVertexInputBindingDescription inputBindingDescriptor;
	std::vector<VkVertexInputAttributeDescription> inputAttributeDescription;
	if (!publicShader->pipelineCreateInfo.screenSpace) {
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
		viewportState = fzbCreateViewStateCreateInfo(1);
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
	pipelineInfo.pViewportState = &viewportState;
	if (pipelineCreateInfo.dynamicView) {
		pipelineInfo.pDynamicState = &dynamicState;
	}
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;	//�Ƚ������ӣ��������
	pipelineInfo.subpass = subPassIndex;	//��Ӧrenderpass���ĸ��Ӳ���
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//����ֱ��ʹ������pipeline
	pipelineInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
		vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
	}

}
void FzbShaderVariant::render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets) {
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	//0�������uniform,1: material��texture��transform��2����ͬmeshBatch��materialIndex
	for (int i = 0; i < componentDescriptorSets.size(); i++) {
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, i, 1, &componentDescriptorSets[i], 0, nullptr);
	}
	if (meshBatch.meshes.size() > 0) {
		meshBatch.render(commandBuffer, pipelineLayout, componentDescriptorSets.size());
	}
}
bool FzbShaderVariant::operator==(const FzbShaderVariant& other) const {
	return publicShader == other.publicShader && macros == other.macros;	//�겻ͬ���ǲ�ͬ��shader���壬���ڲ�ͬ��SRP Batch����ʵ��Ӧ�ü��Ϲ������õģ�֮����˵��
}