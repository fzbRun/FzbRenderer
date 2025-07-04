#version 460

layout(location = 0) in vec3 vertexWorldPos;
#ifdef useVertexNormal
layout(location = 1) in vec3 vertexNormal;
#endif
#ifdef useVertexTexCoords
layout(location = 2) in vec2 vertexTexCoords;
#endif

#ifdef useAlbedoMap
#define albedoMapIndex 0
#endif
#ifdef useNormalMap
#define normalMapIndex albedoMapIndex + 1
#endif

#if defined(useAlbedoMap) && defined(useNormalMap)
#define textureNum 2
#elif defined(useAlbedoMap) || defined(useNormalMap)
#define textureNum 1
#else
#define textureNum 0
#endif

#if defined(useTextureProperty) || defined(useNumberProperty)
#define meshDescriptorSetIndex 2
#else
#define meshDescriptorSetIndex 1
#endif

//主组件描述符，相机信息
layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;
//主组件描述符，光照信息
//layout(set = 0, binding = 1) uniform lightUniformBufferObject{
//	mat4 model;
//	mat4 view;
//	mat4 proj;
//	vec4 lightPos_strength;
//#ifdef useAreaLight
//	vec4 normal;
//	vec4 size;
//#endif
//} lubo;

#ifdef useAlbedoMap
layout(set = 1, binding = albedoMapIndex) uniform sampler2D albedoMap;
#endif
#ifdef useNormalMap
layout(set = 1, binding = normalMapIndex) uniform sampler2D normalMap;
#endif
#ifdef useNumberProperty
layout(set = 1, binding = textureNum) uniform MaterialBuffer{
#ifdef useAlbedo
	vec4 albedo;
#endif
};
#endif

layout(set = meshDescriptorSetIndex, binding = 0) uniform MeshBuffer{
	mat4 transformMatrix;
};

#include "../../common/getAttribute.glsl"

layout(location = 0) out vec4 fragColor;

void main() {
	vec3 cameraPos = cubo.cameraPos.xyz;
	vec3 lightPos = vec3(0.0f, 10.0f, 0.0f);	// lubo.lightPos_strength.xyz;
	float lightStrength = 2.0f;	// lubo.lightPos_strength.w;

	vec3 i = normalize(lightPos - vertexWorldPos);
	vec3 o = normalize(cameraPos - vertexWorldPos);
	vec3 h = normalize(i + o);

	vec3 normal = getNormal();
	vec3 albedo = getAlbedo().rgb;

	float ambient = 0.1f;
	float diffuse = dot(i, normal);
	vec3 specular = vec3(pow(dot(h, normal), 32));

	fragColor = vec4(((ambient + diffuse) * albedo + specular), 1.0f);
}