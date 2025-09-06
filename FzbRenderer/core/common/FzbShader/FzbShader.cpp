#include "./FzbShader.h"
#include "../FzbComponent/FzbComponent.h"

#include <pugixml/src/pugixml.hpp>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <regex>
#include <sstream>
#include <iostream>
#include <filesystem>>
#include "../FzbRenderer.h"

std::vector<uint32_t> readSPIRVFile(const std::string& filename) {
	// 以二进制模式从文件末尾开始读取
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + filename);
	}

	// 获取文件大小
	size_t fileSize = static_cast<size_t>(file.tellg());

	// 检查文件大小是否是 4 的倍数（SPIR-V 要求）
	if (fileSize % sizeof(uint32_t) != 0) {
		throw std::runtime_error("SPIR-V file size is not a multiple of 4 bytes");
	}

	// 计算需要多少 uint32_t 元素
	size_t wordCount = fileSize / sizeof(uint32_t);
	std::vector<uint32_t> buffer(wordCount);

	// 回到文件开头并读取所有数据
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

	// 关闭文件并返回数据
	file.close();
	return buffer;
}
void getSource(const std::string& path, std::string& buffer) {
	try {
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			throw std::runtime_error("Unable to open file: " + path);
		}

		// 移动到文件末尾获取大小
		file.seekg(0, std::ios::end);
		size_t size = file.tellg();
		file.seekg(0, std::ios::beg);

		// 读取文件内容
		buffer.resize(size);
		file.read(buffer.data(), size);
	}
	catch (const std::exception& e) {
		throw std::runtime_error("Failed to read file " + path + "\n" + e.what());
	}
}
//--------------------------------------------编译shader-----------------------------------
/*
struct HBE_Includer : glslang::TShader::Includer {
	std::string path;
	std::set<IncludeResult*> results;

	HBE_Includer(const std::string& path) {
		this->path = path.substr(0, path.find_last_of("\\/") + 1);
	}

	// For both include methods below:
	//
	// Resolves an inclusion request by name, current source name,
	// and include depth.
	// On success, returns an IncludeResult containing the resolved name
	// and content of the include.
	// On failure, returns a nullptr, or an IncludeResult
	// with an empty string for the headerName and error details in the
	// header field.
	// The Includer retains ownership of the contents
	// of the returned IncludeResult value, and those contents must
	// remain valid until the releaseInclude method is called on that
	// IncludeResult object.
	//
	// Note "local" vs. "system" is not an "either/or": "local" is an
	// extra thing to do over "system". Both might get called, as per
	// the C++ specification.

	// For the "system" or <>-style includes; search the "system" paths.


	IncludeResult* includeSystem(const char* file_path, const char* includer_name, size_t inclusion_depth) override {
		std::string* source = new std::string();
		source->erase(std::find(source->begin(), source->end(), '\0'), source->end());
		Shader::getSource(file_path, *source);
		IncludeResult* result = new IncludeResult(path, source->c_str(), source->size() - 2, source);
		results.emplace(result);
		return result;
	}

	// For the "local"-only aspect of a "" include. Should not search in the
	// "system" paths, because on returning a failure, the parser will
	// call includeSystem() to look in the "system" locations.
	IncludeResult* includeLocal(const char* file_path, const char* includer_name, size_t inclusion_depth) override {
		std::string* source = new std::string();
		Shader::getSource(path + file_path, *source);
		source->erase(std::find(source->begin(), source->end(), '\0'), source->end());
		IncludeResult* result = new IncludeResult(path + file_path, source->c_str(), source->size(), source);
		results.emplace(result);
		return result;
	}

	void releaseInclude(IncludeResult* result) override {
		if (result != NULL) {
			results.erase(result);
			delete (std::string*)result->userData;
			delete result;
		}
	}

	~HBE_Includer() {
		for (auto r : results) {
			delete[] r->headerData;
			delete r;
		}
	}
};
void FzbShaderCompiler::GLSLToSpirV(const char* source, size_t size, std::vector<uint32_t>& spirv, 
	VkShaderStageFlagBits shadertType, const std::string& shaderPath, const std::string& preamble) {
	EShLanguage stage = EShLangVertex;
	auto spirv_target = glslang::EShTargetSpv_1_3;
	switch (shadertType) {
		case VK_SHADER_STAGE_COMPUTE_BIT:
			stage = EShLangCompute;
			break;
		case VK_SHADER_STAGE_VERTEX_BIT:
			stage = EShLangVertex;
			break;
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			stage = EShLangFragment;
			break;
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			stage = EShLangGeometry;
			break;
		case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
			stage = EShLangRayGen;
			spirv_target = glslang::EShTargetSpv_1_4;
			break;
		case VK_SHADER_STAGE_MISS_BIT_KHR:
			stage = EShLangMiss;
			spirv_target = glslang::EShTargetSpv_1_4;
			break;
		case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
			stage = EShLangClosestHit;
			spirv_target = glslang::EShTargetSpv_1_4;
			break;
		case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
			stage = EShLangAnyHit;
			spirv_target = glslang::EShTargetSpv_1_4;
			break;
		case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
			stage = EShLangIntersect;
			spirv_target = glslang::EShTargetSpv_1_4;
			break;
	}
	glslang::TShader shader(stage);

	glslang::EshTargetClientVersion vulkan_version;
	switch (FzbMainComponent::apiVersion) {
		case VK_API_VERSION_1_0:
			vulkan_version = glslang::EShTargetVulkan_1_0;
			break;
		case VK_API_VERSION_1_1:
			vulkan_version = glslang::EShTargetVulkan_1_1;
			break;
		case VK_API_VERSION_1_2:
			vulkan_version = glslang::EShTargetVulkan_1_2;
			break;
		case VK_API_VERSION_1_3:
			vulkan_version = glslang::EShTargetVulkan_1_3;
			break;
	}

	const char* shaderStrings[] = { source.c_str() };
	shader.setStrings(shaderStrings, 1);
	shader.setEnvClient(glslang::EShClient::EShClientVulkan, vulkan_version);
	shader.setEnvTarget(glslang::EShTargetSpv, spirv_target);
	shader.setStringsWithLengths(source_ptr, &length, 1);
	shader.setPreamble(preamble.c_str());
	shader.getIntermediate()->setSource(glslang::EShSourceGlsl);

	//shader.setAutoMapBindings(true);
	shader.getIntermediate()->setSourceFile(shaderPath.c_str());
#ifdef DEBUG_MODE
	shader.getIntermediate()->addSourceText(source, size);
	shader.getIntermediate()->setSourceFile((RESOURCE_PATH + shader_path).c_str());
#endif
	HBE_Includer includer(shaderPath);
	EShMessages message = static_cast<EShMessages>(EShMessages::EShMsgVulkanRules | EShMessages::EShMsgSpvRules);

	if (!shader.parse(&DefaultTBuiltInResource,
		450,
		ECoreProfile,
		false,
		false,
		message,
		includer)) {
		//Log::warning(shader.getInfoDebugLog());
		std::cerr << "Shader compilation failed: " << shader.getInfoLog() << std::endl;
	}
}

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

	return mainContent;
}
*/
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
	}
};
class FileIncluder : public glslang::TShader::Includer {
private:
	std::string basePath;
	std::set<IncludeResult*> includeResults;

public:
	FileIncluder(const std::string& basePath) : basePath(basePath) {}

	virtual ~FileIncluder() {
		for (auto r : includeResults) {
			delete[] r->headerData;
			delete r;
		}
	}

	// 处理系统包含（<>语法）
	IncludeResult* includeSystem(const char* file_path, const char* includer_name, size_t inclusion_depth) override {
		std::string* source = new std::string();
		source->erase(std::find(source->begin(), source->end(), '\0'), source->end());
		getSource(file_path, *source);
		IncludeResult* result = new IncludeResult(file_path, source->c_str(), source->size() - 2, source);
		includeResults.emplace(result);
		return result;
	}

	// 处理本地包含（""语法）
	IncludeResult* includeLocal(const char* file_path, const char* includer_name, size_t inclusion_depth) override {
		std::string* source = new std::string();
		std::string includeShaderPath = basePath + "/" + file_path;
		getSource(includeShaderPath, *source);
		source->erase(std::find(source->begin(), source->end(), '\0'), source->end());
		IncludeResult* result = new IncludeResult(includeShaderPath, source->c_str(), source->size(), source);
		includeResults.emplace(result);
		return result;
	}

	// 释放包含结果
	void releaseInclude(IncludeResult* result) override {
		if (result != NULL) {
			includeResults.erase(result);
			delete (std::string*)result->userData;
			delete result;
		}
	}
};
std::vector<uint32_t> compileGLSL(
	const std::string& filePath,
	VkShaderStageFlagBits stage,
	std::string shaderVersion,
	const std::map<std::string, bool>& macros)
{
	int shaderVersion_I = std::stoi(shaderVersion);

	// 读取主文件
	std::ifstream mainFile(filePath);
	if (!mainFile.is_open()) throw std::runtime_error("Failed to open shader file: " + filePath);
	std::string source( (std::istreambuf_iterator<char>(mainFile)), std::istreambuf_iterator<char>());

	//添加版本和宏
	std::string preamble = "#version " + shaderVersion + "\n";
	preamble += "#extension GL_GOOGLE_include_directive : require\n";
	for (const auto& macro : macros) {
		if (macro.second) preamble += "#define " + macro.first + "\n";
	}
	source = preamble + source;
	const char* sourcePtr = source.c_str();
	int shaderLength = source.length();
	
	// 设置着色器阶段
	EShLanguage lang = EShLangVertex;
	auto spirv_target = glslang::EShTargetSpv_1_3;
	std::string stageString = "顶点着色器";
	switch (stage) {
	case VK_SHADER_STAGE_COMPUTE_BIT:
		lang = EShLangCompute;
		stageString = "计算着色器";
		break;
	case VK_SHADER_STAGE_VERTEX_BIT:
		lang = EShLangVertex;
		stageString = "顶点着色器";
		break;
	case VK_SHADER_STAGE_FRAGMENT_BIT:
		lang = EShLangFragment;
		stageString = "片元着色器";
		break;
	case VK_SHADER_STAGE_GEOMETRY_BIT:
		lang = EShLangGeometry;
		stageString = "几何着色器";
		break;
	case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
		lang = EShLangRayGen;
		spirv_target = glslang::EShTargetSpv_1_4;
		stageString = "RayGen着色器";
		break;
	case VK_SHADER_STAGE_MISS_BIT_KHR:
		lang = EShLangMiss;
		spirv_target = glslang::EShTargetSpv_1_4;
		stageString = "RayMiss着色器";
		break;
	case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
		lang = EShLangClosestHit;
		spirv_target = glslang::EShTargetSpv_1_4;
		stageString = "ClosesHit着色器";
		break;
	case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
		lang = EShLangAnyHit;
		spirv_target = glslang::EShTargetSpv_1_4;
		stageString = "AnyHit着色器";
		break;
	case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
		lang = EShLangIntersect;
		spirv_target = glslang::EShTargetSpv_1_4;
		stageString = "Intersect着色器";
		break;
	}
	glslang::EshTargetClientVersion vulkan_version = glslang::EShTargetVulkan_1_0;
	switch (FzbRenderer::globalData.apiVersion) {
	case VK_API_VERSION_1_0:
		vulkan_version = glslang::EShTargetVulkan_1_0;
		break;
	case VK_API_VERSION_1_1:
		vulkan_version = glslang::EShTargetVulkan_1_1;
		break;
	case VK_API_VERSION_1_2:
		vulkan_version = glslang::EShTargetVulkan_1_2;
		break;
	case VK_API_VERSION_1_3:
		vulkan_version = glslang::EShTargetVulkan_1_3;
		break;
	}

	// 创建着色器对象
	glslang::TShader shader(lang);
	shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, shaderVersion_I);
	shader.setEnvClient(glslang::EShClient::EShClientVulkan, vulkan_version);
	shader.setEnvTarget(glslang::EShTargetSpv, spirv_target);
	//shader.setPreamble(preamble.c_str());
	shader.setStringsWithLengths(&sourcePtr, &shaderLength, 1);
	//shader.setStrings(&sourcePtr, 1);

	// 设置编译选项
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);	//EShMsgSpvRules表示生成spv，EShMsgVulkanRules为使用vulkan规范
	TBuiltInResource resources = DefaultTBuiltInResource;
	FileIncluder includer(std::filesystem::path(filePath).parent_path().string());
	if (!shader.parse(&resources, shaderVersion_I, ENoProfile, false, false, messages, includer)) {
		throw std::runtime_error(stageString + " GLSL compilation failed:\n" + std::string(shader.getInfoLog()));
	}

	// 链接程序
	glslang::TProgram program;
	program.addShader(&shader);
	if (!program.link(messages)) {
		throw std::runtime_error("GLSL linking failed:\n" + std::string(program.getInfoLog()));
	}

	// 生成 SPIR-V
	std::vector<uint32_t> spirv;
	glslang::SpvOptions options{};
#ifdef NDEBUG
	options.generateDebugInfo = true;
	options.stripDebugInfo = false;
	options.disableOptimizer = true;
#else
	options.generateDebugInfo = false;
	options.stripDebugInfo = true;
	options.disableOptimizer = false;
#endif
	glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &options);

	return spirv;
}

//--------------------------------------------shader-----------------------------------
FzbShader::FzbShader() {}
FzbShader::FzbShader(std::string path, FzbShaderExtensionsSetting extensionsSetting, bool useStaticCompile) {
	this->path = path;
	pugi::xml_document doc;
	if (!doc.load_file((path + "/shaderInfo.xml").c_str())) {
		throw std::runtime_error("pugixml打开文件失败");
	}

	pugi::xml_node shaderInfos = doc.document_element();	//获取根节点，即<ShaderInfo>
	if (pugi::xml_node properties = shaderInfos.child("Properties")) {	//只有usability=true，才会加入map，其实可以用vector存储，但是不好使用string检索。
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
	//initVertexFormat();

	if (pugi::xml_node shaders = shaderInfos.child("Shaders")) {
		if (pugi::xml_node shaderVersion = shaders.child("shaderVersion")) {
			this->shaderVersion = shaderVersion.attribute("value").value();
		}
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
				if (primitiveTopologyValue == "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST") {
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
				else if (rasterizerExtensionsValue == "conservativeRasterization" && extensionsSetting.conservativeRasterization) {
					float OverestimationSize = 0.5f;
					if (rasterizerExtensionsNode.attribute("OverestimationSize"))
						OverestimationSize = std::stof(rasterizerExtensionsNode.attribute("OverestimationSize").value());
					pipelineCreateInfo.conservativeRasterizationState = getRasterizationConservativeState(OverestimationSize, pipelineCreateInfo.rasterizerExtensions);
					pipelineCreateInfo.rasterizerExtensions = &pipelineCreateInfo.conservativeRasterizationState;
				}
				else {
					//throw std::runtime_error("新的光栅化扩展，快来写");
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
		if (pugi::xml_node dynamicStateInfoNode = pipelineInfoNode.child("dynamicStateInfo")) {	//目前动态只关于视口，等于sceneXML中给出的分辨率
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
			if (pugi::xml_node viewGetNode = viewportStateInfoNode.child("program")) {
				//程序内编写，不需要从xml中读取
			}
			else {
				if (pugi::xml_node resolutionNode = viewportStateInfoNode.child("resolution")) {
					this->resolution.width = std::stof(resolutionNode.attribute("width").value());
					this->resolution.height = std::stof(resolutionNode.attribute("height").value());
				}
				for (pugi::xml_node viewportInfoNode : viewportStateInfoNode.children("viewport")) {
					VkViewport viewport;
					viewport.x = std::stof(viewportInfoNode.attribute("x").value());
					viewport.y = std::stof(viewportInfoNode.attribute("y").value());
					viewport.width = std::stof(viewportInfoNode.attribute("width").value());
					viewport.height = std::stof(viewportInfoNode.attribute("height").value());
					if (pugi::xml_attribute minDepth = viewportInfoNode.attribute("minDepth")) viewport.minDepth = std::stof(minDepth.value());
					else viewport.minDepth = 0.0f;
					if (pugi::xml_attribute maxDepth = viewportInfoNode.attribute("maxDepth")) viewport.maxDepth = std::stof(maxDepth.value());
					else viewport.maxDepth = 1.0f;
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
					else if (viewportExtensionValue == "swizzle" && extensionsSetting.swizzle) {
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
						/*
						pipelineCreateInfo.swizzles[0] = {		//前面
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
						};
						pipelineCreateInfo.swizzles[1] = {		//左面
							VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
						};
						pipelineCreateInfo.swizzles[2] = {		//下面
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
							VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
						};
						*/
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

	if (useStaticCompile) {
		this->useStaticCompile = true;
		for (auto& shaderPair : this->shaders) {
			std::string spvPath = path + "/spv/" + shaderPair.second.substr(0, shaderPair.second.find_first_of(".")) + ".spv";
			this->shaderSpvs.insert({ shaderPair.first, readSPIRVFile(spvPath) });
		}
	}
}
void FzbShader::clean() {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].clean();
	}
}

VkExtent2D FzbShader::getResolution() {
	return this->resolution;
}
void FzbShader::setViewStateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, void* viewportExtensios) {
	this->pipelineCreateInfo.viewports = viewports;
	this->pipelineCreateInfo.scissors = scissors;
	this->pipelineCreateInfo.viewportExtensions = viewportExtensios;
}

void FzbShader::createShaderVariant(FzbMaterial* material, FzbVertexFormat vertexFormat) {
	for (auto& materialPair : material->properties.textureProperties) {
		if (!this->properties.textureProperties.count(materialPair.first)) {
			std::string error = "材质" + material->id + "传入shader没有的资源: " + materialPair.first;
			throw std::runtime_error(error);
		}
	}
	for (auto& materialPair : material->properties.numberProperties) {
		if (!this->properties.numberProperties.count(materialPair.first)) {
			std::string error = "材质" + material->id + "传入shader没有的资源: " + materialPair.first;
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
void FzbShader::createMeshBatch(std::vector<FzbMesh>& sceneMeshSet) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].createMeshBatch(sceneMeshSet);
	}
}
void FzbShader::createDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage>& sceneImages) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].createDescriptor(sceneDescriptorPool, sceneImages);
	}
}
void FzbShader::createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
	for (int i = 0; i < shaderVariants.size(); i++) {
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts_temp = descriptorSetLayouts;
		if (shaderVariants[i].descriptorSetLayout) descriptorSetLayouts_temp.push_back(shaderVariants[i].descriptorSetLayout);
		shaderVariants[i].createPipeline(renderPass, subPassIndex, descriptorSetLayouts_temp);
	}
}
void FzbShader::render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets) {
	if (pipelineCreateInfo.dynamicView) {
		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(resolution.width);
		viewport.height = static_cast<float>(resolution.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = resolution;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
	}
	
	for (int i = 0; i < shaderVariants.size(); i++) {
		shaderVariants[i].render(commandBuffer, componentDescriptorSets);
	}
}
bool FzbShader::operator==(const FzbShader& other) const {
	return path == other.path;
}

//--------------------------------------------编译shader变体-----------------------------------
void FzbShaderVariant::changeVertexFormatAndMacros(FzbVertexFormat vertexFormat) {
	this->vertexFormat.mergeUpward(vertexFormat);	//比如说一定要有texCoords，但是由于material和shader变体没有纹理，所以可以不开启宏

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
	this->publicShader = publicShader;
	this->macros = publicShader->macros;
	this->materials.push_back(material);
	this->vertexFormat = FzbVertexFormat();

	//如果shader开启了某个资源类型，但是material没有传入相应的资源，则关闭shader的相应的资源类（宏）
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
	//number有点不同，如果material没有传入，而shader开启了，则使用默认值
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
	//创造描述符
	uint32_t textureNum = this->properties.textureProperties.size();
	uint32_t numberNum = this->properties.numberProperties.size() > 0 ? 1 : 0;	//所有数值属性用一个uniformBuffer即可
	if (textureNum + numberNum == 0) {
		return;
	}
	std::vector<VkDescriptorType> type(textureNum + numberNum);
	std::vector<VkShaderStageFlags> stage(textureNum + numberNum);
	for (int i = 0; i < textureNum; i++) {
		type[i] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		stage[i] = VK_SHADER_STAGE_ALL;	//shaderStage设置为all和特地shader不会影响数据的读取速度，只是在编译时候检测范围不同而已（影响编译速度而已）。
	}
	if (numberNum) {
		type[textureNum] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		stage[textureNum] = VK_SHADER_STAGE_ALL;
	}
	this->descriptorSetLayout = fzbCreateDescriptLayout(type.size(), type, stage);
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
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	if (pipeline) {
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, pipeline, nullptr);
	}
	vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	meshBatch.clean();
}
void FzbShaderVariant::createMeshBatch(std::vector<FzbMesh>& sceneMeshSet) {
	/*
	如果所有的mesh的material是相同的或者给定的materials只有一个material，则meshBatch的materials只需要存储一个material即可，并且其useSameMaterial=true
	*/
	this->meshBatch = FzbMeshBatch();
	std::unordered_set<FzbMaterial*> materialSet(this->materials.begin(), this->materials.end());
	std::map<FzbMaterial*, FzbMaterial*> meshMaterials;
	std::vector<FzbMaterial*> meshBatchMaterials;
	for (size_t i = 0; i < sceneMeshSet.size(); i++) {
		FzbMesh& mesh = sceneMeshSet[i];
		FzbMaterial* material = mesh.material;
		if (materialSet.find(material) != materialSet.end()) {
			this->meshBatch.meshes.push_back(&mesh);
			meshBatchMaterials.push_back(material);
			if (!meshMaterials.count(material)) {	//将所有不重复的material放入其中，如果最后只有一个，那么说明所有mesh共用一个material
				meshMaterials.insert({ material, material });
			}
		}
	}
	if (meshMaterials.size() == 1) {	//说明所有mesh共用一个material，则可以复用
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
		std::vector<uint32_t> shaderSpvCode;
		if (this->publicShader->useStaticCompile) shaderSpvCode = this->publicShader->shaderSpvs[shader.first];
		else shaderSpvCode = compileGLSL(shaderPath, shaderStage, this->publicShader->shaderVersion, this->macros);

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = shaderSpvCode.size() * sizeof(uint32_t);
		createInfo.pCode = shaderSpvCode.data();

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(FzbRenderer::globalData.logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
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
void FzbShaderVariant::createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

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

	pipelineLayout = fzbCreatePipelineLayout(&descriptorSetLayouts);

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
void FzbShaderVariant::render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets) {
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	//0：组件的uniform,1: material、texture、transform，2：不同meshBatch的materialIndex
	for (int i = 0; i < componentDescriptorSets.size(); i++) {
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, i, 1, &componentDescriptorSets[i], 0, nullptr);
	}
	if (meshBatch.meshes.size() > 0) {
		meshBatch.render(commandBuffer, pipelineLayout, componentDescriptorSets.size());
	}
}
bool FzbShaderVariant::operator==(const FzbShaderVariant& other) const {
	return publicShader == other.publicShader && macros == other.macros;	//宏不同就是不同的shader变体，属于不同的SRP Batch，其实还应该加上管线设置的，之后再说吧
}