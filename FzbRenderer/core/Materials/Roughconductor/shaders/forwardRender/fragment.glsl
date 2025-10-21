layout(location = 0) in vec3 vertexWorldPos;
#ifdef VERTEX_NORMAL
layout(location = VERTEX_NORMAL_CHANNEL) in vec3 vertexNormal;
#endif
#ifdef VERTEX_TEXCOORDS
layout(location = VertexTexCoordsChannel) in vec2 vertexTexCoords;
#endif
#ifdef VERTEX_TANGENT
layout(location = VertexTangentChannel) in vec3 vertexTangent;
#endif

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
};
//主组件描述符，光照信息
layout(set = 0, binding = 1) uniform lightsUniformBufferObject{
	LightDate lightData[16];
	uint lightNum;
} lubo;

#ifdef NORMALMAP
layout(set = 1, binding = NORMALMAP_CHANNEL) uniform sampler2D normalMap;
#endif
#ifdef ALBEDOMAP
layout(set = 1, binding = ALBEDOMAP_CHANNEL) uniform sampler2D albedoMap;
#endif

layout(set = 1, binding = NUMBERPROPERTIES_CHANNEL) uniform MaterialBuffer{
	vec4 albedo;
	vec4 bsdfPara;
	vec4 emissive;
};

#include "../../../shaders/getAttribute.glsl"
#include "../../../shaders/getIllumination.glsl"

layout(location = 0) out vec4 fragColor;

void main() {
	vec3 cameraPos = cubo.cameraPos.xyz;
	vec3 o = normalize(cameraPos - vertexWorldPos);
	vec3 normal = getNormal();
	vec3 vertexAlbedo = getAlbedo().rgb;

	fragColor = getIllumination(o, normal, vertexAlbedo) * 0.1f + vec4(emissive.xyz, 0.0f);
}