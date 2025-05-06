#version 450

layout(lines) in;
layout(line_strip, max_vertices = 2) out;

layout(location = 0) in uint voxelIndex[];
layout(location = 0) out vec3 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz是随机数，而w是帧数
	vec4 swapChainExtent;
} cubo;

layout(set = 0, binding = 1) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

struct FzbSVONode {
	uint nodeIndex;	//当前节点在八叉树中的索引
	uint voxelNum;	//该节点所包含的叶子节点数
	uint subsequentIndex;
	uint hasSubNode;
	vec4 nodePos_Size;
};

layout(set = 1, binding = 0, std430) readonly buffer FzbSVONodes{
	FzbSVONode fzbSVONodes[];
};


void main() {

	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;

	FzbSVONode node = fzbSVONodes[voxelIndex[0]];
	if (node.voxelNum > 0) {
		gl_Position = cubo.proj * cubo.view * cubo.model * vec4(p0 * node.nodePos_Size.w + node.nodePos_Size.xyz, 1.0f);
		EmitVertex();

		gl_Position = cubo.proj * cubo.view * cubo.model * vec4(p1 * node.nodePos_Size.w + node.nodePos_Size.xyz, 1.0f);
		EmitVertex();

		EndPrimitive();
	}

}