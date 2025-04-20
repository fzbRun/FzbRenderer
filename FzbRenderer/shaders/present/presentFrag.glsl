#version 450

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec2 texCoord;

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

layout(set = 1, binding = 0, r32ui) uniform coherent uimage3D voxelImage;
layout(input_attachment_index = 0, set = 2, binding = 0) uniform subpassInput inputColor;
layout(set = 3, binding = 0, r32ui) uniform coherent uimage2D depthImage;

//const float depthDigit = 65536.0f;	//4294967295 = 2的32次方-1

void main() {

	//FragColor = subpassLoad(inputColor);
	//return;

	float depth = float(imageLoad(depthImage, ivec2(texCoord * cubo.swapChainExtent.xy)).r) / cubo.swapChainExtent.z;
	if (depth >= 1.0f) {
		discard;
	}
	vec4 ndcPos = vec4(texCoord * 2.0f - 1.0f, depth, 1.0f);
	vec4 worldPos = inverse(cubo.proj * cubo.view) * ndcPos;
	worldPos.xyz /= worldPos.w;

	//FragColor = vec4(worldPos.rgb, 1.0f);
	//return;

	ivec3 voxelIndex = ivec3((worldPos.xyz - vubo.voxelStartPos.xyz) / vubo.voxelSize_Num.x);
	vec4 voxelValue = unpackUnorm4x8(imageLoad(voxelImage, voxelIndex).r);
	if (voxelValue.w == 0) {
		discard;
	}
	FragColor = vec4(voxelValue.rgb, 1.0f);

}