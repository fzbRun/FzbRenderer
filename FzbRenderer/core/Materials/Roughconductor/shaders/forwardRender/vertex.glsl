layout(location = 0) in vec3 pos_in;
layout(location = 0) out vec3 vertexWorldPos;	//从顶点传出去的属性，就按vertex+属性来命名
#ifdef VERTEX_NORMAL
layout(location = VERTEX_NORMAL_CHANNEL) in vec3 normal_in;
layout(location = VERTEX_NORMAL_CHANNEL) out vec3 vertexNormal;
#endif
#ifdef VERTEX_TEXCOORDS
layout(location = VERTEX_TEXCOORDS_CHANNEL) in vec2 texCoords_in;
layout(location = VERTEX_TEXCOORDS_CHANNEL) out vec2 vertexTexCoords;
#endif
#ifdef VERTEX_TANGENT
layout(location = VERTEX_TANGENT_CHANNEL) in vec3 tangent_in;
layout(location = VERTEX_TANGENT_CHANNEL) out vec3 vertexTangent;
#endif

//主组件描述符，相机信息
layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

#ifdef DYNAMICMESH 
layout(set = 2, binding = 0) uniform MeshBuffer{
	mat4 transformMatrix;
};
#endif

void main() {
#ifdef DYNAMICMESH
	vertexWorldPos = (transformMatrix * vec4(pos_in, 1.0f)).xyz;
	gl_Position = cubo.proj * cubo.view * vec4(vertexWorldPos, 1.0f);
#ifdef VERTEX_NORMAL
	vertexNormal = (transpose(inverse(transformMatrix)) * vec4(normal_in, 1.0f)).xyz;
#endif
#ifdef VERTEX_TANGENT
	vertexTangent = (transformMatrix * vec4(tangent_in, 1.0f)).xyz;
#endif
#else
	vertexWorldPos = pos_in;
	gl_Position = cubo.proj * cubo.view * vec4(vertexWorldPos, 1.0f);
#ifdef VERTEX_NORMAL
	vertexNormal = normal_in;
#endif
#ifdef VERTEX_TANGENT
	vextexTangent = tangent_in;
#endif
#endif
#ifdef VERTEX_TEXCOORDS
	vertexTexCoords = texCoords_in;
#endif
}
