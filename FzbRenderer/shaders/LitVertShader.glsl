#version 460

struct Material {
	vec4 ka;
	vec4 kd;
	vec4 ks;
	vec4 ke;
};

struct MeshMaterialUniformObject {
	int transformIndex;
	int materialIndex;	//当前draw的mesh的材质在材质buffer中的索引
	int albedoTextureIndex;	//反射率纹理索引
	int normalTextureIndex; //发现纹理索引
};

layout(location = 0) in vec3 pos_in;
//#ifdef useNormal
//layout(location = 1) in vec3 normal_in;
//#endif
//#ifdef useTexture
//layout(location = 2) in vec2 texCoord_in;
//#endif
//#ifdef useTBN
//layout(location = 3) in vec3 tangent_in;
//#endif

layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

layout(set = 1, binding = 0) buffer readonly transformMatrixBuffer{
	mat4 models[];
};

layout(set = 1, binding = 1) buffer readonly materialBuffer {
	Material materials[];
};

layout(set = 2, binding = 0) buffer readonly meshMaterialIndexBuffer {
	MeshMaterialUniformObject materialIndexs[];
};

void main() {

	int meshMaterialIndex = materialIndexs[gl_DrawID].transformIndex;
	mat4 model = meshMaterialIndex != -1 ? models[meshMaterialIndex] : mat4(1.0f);
	gl_Position = cubo.proj * cubo.view * model * vec4(pos_in, 1.0f);
	//gl_Position = cubo.proj * cubo.view * cubo.model * vec4(pos_in, 1.0f);
}
