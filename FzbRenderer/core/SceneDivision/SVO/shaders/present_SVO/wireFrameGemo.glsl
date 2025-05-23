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
} cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

struct FzbSVONode {
	uint shuffleKey;	//��ǰ�ڵ��ڰ˲����е�����
	uint voxelNum;	//�ýڵ���������Ҷ�ӽڵ���
	uint label;
	uint hasSubNode;
};
layout(set = 2, binding = 0, std430) readonly buffer FzbSVONodes{
	FzbSVONode fzbSVONodes[];
};

vec4 getWorldPos(vec3 p, uint shuffleKey) {

	uint svoDepth = (shuffleKey >> 28) & 0xF;	//�õ���ǰ�ڵ��������˲����ĵڼ���
	float svoVoxelNum = vubo.voxelSize_Num.y;
	float levelSize = svoVoxelNum / pow(2, svoDepth);	//�õ��ò�һ���˷�Բ�м������أ����0��128����һ��64
	float size = vubo.voxelSize_Num.x * levelSize;
	vec3 worldPos = p * size + vubo.voxelStartPos.xyz;

	while (svoDepth > 0) {
		worldPos.x += (shuffleKey & 1) * size;
		worldPos.y += ((shuffleKey >> 1) & 1) * size;
		worldPos.z += ((shuffleKey >> 2) & 1) * size;

		svoDepth -= 1;
		shuffleKey = shuffleKey >> 3;
		size *= 2;
	}
	return vec4(worldPos, 1.0f);
}

void main() {

	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;

	FzbSVONode node = fzbSVONodes[voxelIndex[0]];
	if (node.voxelNum > 0) {
		gl_Position = cubo.proj * cubo.view * cubo.model * getWorldPos(p0, node.shuffleKey);
		EmitVertex();

		gl_Position = cubo.proj * cubo.view * cubo.model * getWorldPos(p1, node.shuffleKey);
		EmitVertex();

		EndPrimitive();
	}

}