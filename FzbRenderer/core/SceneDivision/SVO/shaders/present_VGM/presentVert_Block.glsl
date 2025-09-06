layout(location = 0) in vec3 pos_in;
layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vbo;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out ivec3 voxelIndex;

void main(){
	//gl_Position = cubo.proj * cubo.view * cubo.model * vec4(pos_in, 1.0f);
	//worldPos = (cubo.model * vec4(pos_in, 1.0f)).xyz;

	//int voxelNum = int(vbo.voxelSize_Num.y * vbo.voxelSize_Num.y * vbo.voxelSize_Num.y);	//现在画面中如果相机转到后面会发现正面会覆盖后面，这是因为没有深度测试，导致后渲染的片元会覆盖先渲染的片元，而先后顺序是我们通过gl_InstanceIndex确定的。
	//int instanceIndex = voxelNum - gl_InstanceIndex;										//因此如果我们voxelNum - gl_InstanceIndex反过来就会得到相反的结果，背面会覆盖正面
	int instanceIndex = gl_InstanceIndex;
	int voxelNum = int(vbo.voxelSize_Num.w);
	int voxelNumSquared = voxelNum * voxelNum;

	int Z = instanceIndex / voxelNumSquared;
	instanceIndex -= Z * voxelNumSquared;
	int Y = instanceIndex / voxelNum;
	int X = instanceIndex - Y * voxelNum;
	vec3 offset = vec3(X,Y,Z);
	
	vec3 pos = offset * vbo.voxelSize_Num.xyz + pos_in;	// +vbo.voxelStartPos.xyz; pos_in就是从startPos开始的
	gl_Position = cubo.proj * cubo.view * vec4(pos, 1.0f);
	worldPos = pos;
	voxelIndex = ivec3(offset);
}

//layout(location = 0) out vec2 texCoord;
//
//void main(){
//	texCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//    gl_Position = vec4(texCoord * 2.0f + -1.0f, 0.0f, 1.0f);
//}