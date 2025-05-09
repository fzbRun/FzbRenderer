#version 450

//const float depthDigit = 65536.0f;	//4294967295 = 2µÄ32´Î·½-1

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
}cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vbo;

layout(set = 1, binding = 1, r32ui) uniform uimage3D voxelMap;
//layout(set = 3, binding = 0, r32ui) uniform uimage2D depthMap;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in flat ivec3 voxelIndex;

layout(location = 0) out vec4 FragColor;

/*
ivec3 getTexelVoxelIndex(vec2 texCoords){
	
	ivec2 texCoordsU = ivec2(texCoords * cubo.swapChainExtent.xy);
	uint texelDepthU = imageLoad(depthMap, texCoordsU).r;
	float texelDepth = float(texelDepthU) / cubo.swapChainExtent.z;
	texelDepth = texelDepth * 2.0f - 1.0f;

	if(texelDepth == 1.0f){
		discard;
	}

	vec4 texelWorldPos = inverse(cubo.proj * cubo.view) * vec4(vec3(texCoords, texelDepth), 1.0f);
	texelWorldPos /= texelWorldPos.w;

	ivec3 texelVoxelIndex = ivec3((texelWorldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	return texelVoxelIndex;

}

ivec3 getTexelVoxelIndex(vec3 worldPos){
	
	vec4 clipPos = cubo.proj * cubo.view * cubo.model * vec4(worldPos, 1.0f);
	vec4 ndcPos = clipPos / clipPos.w;
	vec2 texCoords = ndcPos.xy * 0.5f + 0.5f;
	ivec2 texCoordsU = ivec2(texCoords * cubo.swapChainExtent.xy);
	uint texelDepthU = imageLoad(depthMap, texCoordsU).r;
	float texelDepth = float(texelDepthU) / cubo.swapChainExtent.z;
	texelDepth = texelDepth * 2.0f - 1.0f;

	vec4 texelWorldPos = inverse(cubo.proj * cubo.view) * vec4(vec3(ndcPos.xy, texelDepth), 1.0f);
	texelWorldPos /= texelWorldPos.w;
	ivec3 texelVoxelIndex = ivec3((texelWorldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	return texelVoxelIndex;

}
*/

void main(){
	
	//vec4 viewPos = cubo.view * vec4(worldPos, 1.0f);
	//vec4 clipPos = cubo.proj * viewPos;
	//vec4 ndcPos = clipPos / clipPos.w;
	//vec2 texCoord = ndcPos.xy * 0.5f + 0.5f;
	//float screenDepth = float(imageLoad(depthMap, ivec2(texCoord * cubo.swapChainExtent.xy)).r) / cubo.swapChainExtent.z;

	//vec4 screenViewPos = inverse(cubo.proj) * vec4(ndcPos.xy, screenDepth, 1.0f);
	//screenViewPos = screenViewPos / screenViewPos.w;

	//if (screenDepth < ndcPos.z) {
	//	discard;
	//}

	uint voxelValueU = imageLoad(voxelMap, ivec3(voxelIndex)).r;
	vec4 voxelValue = unpackUnorm4x8(voxelValueU);
	if(voxelValue.w == 0){
		discard;
	}

	//if (worldPos.z > 0) {
	//	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	//}
	//else {
	//	FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	//}
	FragColor = vec4(voxelValue.rgb, 1.0f);

}