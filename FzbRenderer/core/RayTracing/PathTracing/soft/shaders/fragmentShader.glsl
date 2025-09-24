#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require

layout(set = 0, binding = 0) uniform PathTracingSetting{
	uint screenWidth;
	uint screenHeight;
};
layout(set = 0, binding = 1, std430) readonly buffer PathTracingBuffer {
	vec4 result[];
};

struct FzbAABB {
	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;
};
struct FzbBvhNode {
	uint leftNodeIndex;
	uint rightNodeIndex;
	uint triangleNum;
	FzbAABB AABB;
	//uint depth;
};
layout(set = 0, binding = 2, std430) readonly buffer FzbBVHNodes {
	FzbBvhNode fzbBVHNodes[];
};

struct FzbBvhNodeTriangleInfo {
	uint8_t materialIndex;
	uint8_t vertexFormat;	//3位，第一为表示是否使用法线，第二位表示是否使用uv，第三位表示是否使用tangent
	uint indices0;
	uint indices1;
	uint indices2;
	//FzbAABB AABB;
};
layout(set = 0, binding = 3, scalar) readonly buffer FzbTriangleInfos {
	FzbBvhNodeTriangleInfo fzbTriangles[];
};

layout(location = 0) out vec4 fragColor;

void main() {
	uint rightTriangleIndex = fzbBVHNodes[0].rightNodeIndex;
	if (rightTriangleIndex > 10000000) discard;

	vec2 texel = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
	uint texelX = uint(texel.x * screenWidth);
	uint texelY = uint(texel.y * screenHeight);
	uint resultIndex = texelX + texelY * screenWidth;
	fragColor = vec4(result[resultIndex].xyz, 1.0f);
	//fragColor = vec4(texel, 0.0f, 1.0f);
}