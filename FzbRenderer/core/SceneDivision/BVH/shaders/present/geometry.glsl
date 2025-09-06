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
思考一下，在划分时，我根据结点中所有三角形的重心坐标进行划分，然后根据划分得到的节点再计算子节点的AABB
那么，也就是说在兄弟节点的三角形也有可能在当前节点的AABB中，因此我不能通过三角形是否处于节点AABB来判断该三角形是否属于该节点
最好的方式是存储节点的被划分点，可以试试。还需要记录到底用哪个轴来划分。wow，懒得搞啊，一点也不优雅。
我先按AABB试试。
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