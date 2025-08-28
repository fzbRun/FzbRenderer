layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec4 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

layout(set = 1, binding = 1, r32ui) uniform coherent uimage3D voxelMap;

void main() {

	ivec3 voxelIndex = ivec3((worldPos.xyz - vubo.voxelStartPos.xyz) / vubo.voxelSize_Num.xyz);
	uint voxelValueU = imageLoad(voxelMap, voxelIndex).r;
	vec4 voxelValue = unpackUnorm4x8(voxelValueU);
	if (voxelValue.w == 0) {
		discard;
	}
	FragColor = vec4(voxelValue.rgb, 1.0f);
}