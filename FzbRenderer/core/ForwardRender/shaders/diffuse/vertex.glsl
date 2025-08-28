layout(location = 0) in vec3 pos_in;
#ifdef useVertexNormal
layout(location = 1) in vec3 normal_in;
#endif
#ifdef useVertexTexCoords
layout(location = 2) in vec2 texCoords_in;
#endif
#ifdef useVertexTangent
layout(location = 3) in vec3 tangent_in;
#endif

#if defined(useAlbedoMap) && defined(useNormalMap)
#define textureNum 2
#elif defined(useAlbedoMap) || defined(useNormalMap)
#define textureNum 1
#else
#define textureNum 0
#endif

#if defined(useTextureProperty) || defined(useNumberProperty)
#define meshDescriptorSetIndex 2
#else
#define meshDescriptorSetIndex 1
#endif

//主组件描述符，相机信息
layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

//meshBatch描述符，变换矩阵和材质信息
#ifdef useNumberProperty
layout(set = 1, binding = textureNum) uniform MaterialBuffer {
#ifdef useAlbedo
	vec4 albedo;
#endif
};
#endif

layout(set = meshDescriptorSetIndex, binding = 0) uniform MeshBuffer{
	mat4 transformMatrix;
};

layout(location = 0) out vec3 vertexWorldPos;	//从顶点传出去的属性，就按vertex+属性来命名
#ifdef useVertexNormal
layout(location = 1) out vec3 vertexNormal;
#endif
#ifdef useVertexTexCoords
layout(location = 2) out vec2 vextexTexCoords;
#endif

void main() {
	//mat4 model = transformMatrixs[gl_DrawID];
	mat4 model = transformMatrix;
	vertexWorldPos = (model * vec4(pos_in, 1.0f)).xyz;
	gl_Position = cubo.proj * cubo.view * vec4(vertexWorldPos, 1.0f);
#ifdef useVertexNormal
	vertexNormal = (transpose(inverse(model)) * vec4(normal_in, 1.0f)).xyz;
#endif
#ifdef useVertexTexCoords
	vextexTexCoords = texCoords_in;
#endif
}
