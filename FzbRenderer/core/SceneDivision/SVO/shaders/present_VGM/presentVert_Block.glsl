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

	//int voxelNum = int(vbo.voxelSize_Num.y * vbo.voxelSize_Num.y * vbo.voxelSize_Num.y);	//���ڻ�����������ת������ᷢ������Ḳ�Ǻ��棬������Ϊû����Ȳ��ԣ����º���Ⱦ��ƬԪ�Ḳ������Ⱦ��ƬԪ�����Ⱥ�˳��������ͨ��gl_InstanceIndexȷ���ġ�
	//int instanceIndex = voxelNum - gl_InstanceIndex;										//����������voxelNum - gl_InstanceIndex�������ͻ�õ��෴�Ľ��������Ḳ������
	int instanceIndex = gl_InstanceIndex;
	int voxelNum = int(vbo.voxelSize_Num.w);
	int voxelNumSquared = voxelNum * voxelNum;

	int Z = instanceIndex / voxelNumSquared;
	instanceIndex -= Z * voxelNumSquared;
	int Y = instanceIndex / voxelNum;
	int X = instanceIndex - Y * voxelNum;
	vec3 offset = vec3(X,Y,Z);
	
	vec3 pos = offset * vbo.voxelSize_Num.xyz + pos_in;	// +vbo.voxelStartPos.xyz; pos_in���Ǵ�startPos��ʼ��
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