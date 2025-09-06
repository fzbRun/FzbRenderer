layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(set = 0, binding = 0) uniform cameralUniformBufferObject{
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

layout(set = 1, binding = 0) uniform BVHUniformBuffer {
	uint nodeIndex;
	uint bvhTreeDepth;
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
	FzbAABB AABB;
};
layout(set = 1, binding = 1, std430) readonly buffer FzbSVONodes {
	FzbBvhNode fzbBVHNodes[];
};

layout(location = 0) out vec3 worldPos;

/*
˼��һ�£��ڻ���ʱ���Ҹ��ݽ�������������ε�����������л��֣�Ȼ����ݻ��ֵõ��Ľڵ��ټ����ӽڵ��AABB
��ô��Ҳ����˵���ֵܽڵ��������Ҳ�п����ڵ�ǰ�ڵ��AABB�У�����Ҳ���ͨ���������Ƿ��ڽڵ�AABB���жϸ��������Ƿ����ڸýڵ�
��õķ�ʽ�Ǵ洢�ڵ�ı����ֵ㣬�������ԡ�����Ҫ��¼�������ĸ��������֡�wow�����ø㰡��һ��Ҳ�����š�
���Ȱ�AABB���ԡ�
*/
void main() {
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;
	vec3 bcPos = (p0 + p1 + p2) * 0.333f;

	FzbBvhNode node = fzbBVHNodes[nodeIndex];
	FzbAABB nodeAABB = node.AABB;
	if (bcPos.x < nodeAABB.leftX || bcPos.x > nodeAABB.rightX || bcPos.y < nodeAABB.leftY || bcPos.y > nodeAABB.rightY || bcPos.z < nodeAABB.leftZ || bcPos.z > nodeAABB.rightZ) {
		return;
	}

	worldPos = p0;
	gl_Position = cubo.proj * cubo.view * gl_in[0].gl_Position;
	EmitVertex();
	worldPos = p1;
	gl_Position = cubo.proj * cubo.view * gl_in[1].gl_Position;
	EmitVertex();
	worldPos = p2;
	gl_Position = cubo.proj * cubo.view * gl_in[2].gl_Position;
	EmitVertex();
	EndPrimitive();
}