layout(location = 0) in vec3 worldPos;

layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) uniform voxelBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

layout(set = 0, binding = 1, r32ui) uniform coherent volatile uimage3D voxelMap;

//const float depthDigit = 65536.0f;	//4294967295 = 2的32次方-1

vec4 convRGBA8ToVec4(uint val) {
	return vec4(float((val & 0x000000FF)), float((val & 0x0000FF00) >> 8U), float((val & 0x00FF0000) >> 16U), float((val & 0xFF000000) >> 24U));
}

uint convVec4ToRGBA8(vec4 val) {
	return (uint(val.w) & 0x000000FF) << 24U | (uint(val.z) & 0x000000FF) << 16U | (uint(val.y) & 0x000000FF) << 8U | (uint(val.x) & 0x000000FF);
}

void imageAtomicRGBA8Avg(ivec3 coords, vec4 val) {

	//val.rgb *= 255.0f; // Optimise following calculations
	//uint newVal = convVec4ToRGBA8(val);
	uint newVal = packUnorm4x8(val);
	uint prevStoredVal = 0;
	uint curStoredVal;

	//主要思路就是，判断image中的值与刚才拿到的image中的值是否相等，相等说明没有其他线程修改，那么可以将新值放入；
	//反之，说明在while循环过程中有其他线程修改，则重新计算新值
	while ((curStoredVal = imageAtomicCompSwap(voxelMap, coords, prevStoredVal, newVal)) != prevStoredVal) {
		prevStoredVal = curStoredVal;
		//vec4 rval = convRGBA8ToVec4(curStoredVal);
		vec4 rval = unpackUnorm4x8(curStoredVal);
		rval.xyz = rval.xyz * rval.w;	// Denormalize
		vec4 curValF = rval + val;		// Add new value
		curValF.xyz /= curValF.w;		 // Renormalize
		//newVal = convVec4ToRGBA8(curValF);
		newVal = packUnorm4x8(curValF);
	}
}

//void imageAtomicDepthTest(ivec2 texCoords, float depth) {
//
//	uint udepth = uint(depth * cubo.swapChainExtent.z);
//	//uint prevStoredVal = uint(cubo.swapChainExtent.z);
//	//uint curStoredVal = imageAtomicCompSwap(depthMap, texCoords, prevStoredVal, udepth);
//
//	//while (curStoredVal != prevStoredVal && udepth < curStoredVal) {
//	//	prevStoredVal = curStoredVal;
//	//	curStoredVal = imageAtomicCompSwap(depthMap, texCoords, prevStoredVal, udepth);
//	//}
//	imageAtomicMin(depthMap, texCoords, udepth);
//
//}

void main() {

	//FragColor = vec4(abs(normalize(worldPos)), 1.0f);
	//return;

	//vec4 ndcPos = cubo.proj * cubo.view * vec4(worldPos, 1.0f);
	//ndcPos /= ndcPos.w;
	//if (ndcPos.z < 0.0f || ndcPos.z > 1.0f) {
	//	discard;
	//}

	//float depth = ndcPos.z;
	//ivec2 texCoords = ivec2((ndcPos.xy * 0.5f + 0.5f) * cubo.swapChainExtent.xy);
	//imageAtomicDepthTest(texCoords, depth);

	ivec3 voxelIndex = ivec3((worldPos - vubo.voxelStartPos.xyz) / vubo.voxelSize_Num.xyz);
	imageAtomicRGBA8Avg(voxelIndex, vec4(normalize(abs(worldPos)), 1.0f));

}