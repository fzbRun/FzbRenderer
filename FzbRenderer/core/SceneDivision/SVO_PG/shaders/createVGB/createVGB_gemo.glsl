layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out flat vec3 normal;

layout(set = 0, binding = 0) uniform voxelBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

void main() {
	//计算面法线
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	normal = normalize(cross(e1, e2));
	vec3 absNormal = abs(normal);

	int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

	worldPos = p0;
	gl_Position = vubo.VP[viewIndex] * gl_in[0].gl_Position;
	//gl_Position = cubo.proj * cubo.view * gl_in[0].gl_Position;
	EmitVertex();

	worldPos = p1;
	gl_Position = vubo.VP[viewIndex] * gl_in[1].gl_Position;
	//gl_Position = cubo.proj * cubo.view * gl_in[1].gl_Position;
	EmitVertex();

	worldPos = p2;
	gl_Position = vubo.VP[viewIndex] * gl_in[2].gl_Position;
	//gl_Position = cubo.proj * cubo.view * gl_in[2].gl_Position;
	EmitVertex();

	EndPrimitive();
}