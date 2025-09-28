layout(location = 0) in vec3 vertexWorldPos;
#ifdef useVertexNormal
layout(location = 1) in vec3 vertexNormal;
#endif
#ifdef useVertexTexCoords
layout(location = 2) in vec2 vertexTexCoords;
#endif

#ifdef useNormalMap
#define normalMapIndex 0
#ifdef useAlbedoMap
#define albedoMapIndex 1
#endif
#else
#ifdef useAlbedoMap
#define albedoMapIndex 0
#endif
#endif

#if defined(useNormalMap) && defined(useAlbedoMap)
#define textureNum 2
#elif defined(useNormalMap) || defined(useAlbedoMap)
#define textureNum 1
#else
#define textureNum 0
#endif

//#if defined(useTextureProperty) || defined(useNumberProperty)
//#define meshDescriptorSetIndex 2
//#else
//#define meshDescriptorSetIndex 1
//#endif

//������������������Ϣ
layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

struct LightDate {
	mat4 view;
	mat4 proj;
	vec4 pos;
	vec4 strength;
#ifdef useAreaLight
	vec4 normal;
	vec4 size;
#endif
};

//�������������������Ϣ
layout(set = 0, binding = 1) uniform lightsUniformBufferObject{
	LightDate lightData[16];
	uint lightNum;
} lubo;

#ifdef useNormalMap
layout(set = 1, binding = normalMapIndex) uniform sampler2D normalMap;
#endif
#ifdef useAlbedoMap
layout(set = 1, binding = albedoMapIndex) uniform sampler2D albedoMap;
#endif
#ifdef useNumberProperty
layout(set = 1, binding = textureNum) uniform MaterialBuffer{
#ifdef useAlbedo
	vec4 albedo;
#endif
#ifdef useEmissive
	vec4 emissive;
#endif
};
#endif

#include "../../../../../shaders/common/getAttribute.glsl"
#include "../../../../../shaders/common/getIllumination.glsl"

layout(location = 0) out vec4 fragColor;

void main() {
	vec3 cameraPos = cubo.cameraPos.xyz;
	vec3 o = normalize(cameraPos - vertexWorldPos);
	vec3 normal = getNormal();
	vec3 vertexAlbedo = getAlbedo().rgb;

	fragColor = getIllumination(o, normal, vertexAlbedo) * 0.01f;
}