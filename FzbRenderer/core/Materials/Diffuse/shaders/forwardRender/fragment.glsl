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

//#if defined(useTextureProperty) || defined(useNumberProperty)
//#define meshDescriptorSetIndex 2
//#else
//#define meshDescriptorSetIndex 1
//#endif

//主组件描述符，相机信息
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

//主组件描述符，光照信息
layout(set = 0, binding = 1) uniform lightsUniformBufferObject{
	LightDate lightData[16];
	uint lightNum;
} lubo;

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

#include "../../../../../shaders/common/getAttribute.glsl"
#include "../../../../../shaders/common/getIllumination.glsl"

layout(location = 0) out vec4 fragColor;

void main() {
	vec3 cameraPos = cubo.cameraPos.xyz;
	vec3 o = normalize(cameraPos - vertexWorldPos);
	vec3 normal = getNormal();
	vec3 vertexAlbedo = getAlbedo().rgb;

	fragColor = getIllumination(o, normal, vertexAlbedo);
}