#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 worldPos;

layout(set = 0, binding = 0) uniform voxelBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vubo;

struct FzbAABBUint {
	uint leftX;
	uint rightX;
	uint leftY;
	uint rightY;
	uint leftZ;
	uint rightZ;
};
struct FzbSVOVoxelData_PG {
	uint hasData;
	vec3 irradiance;
	FzbAABBUint AABB;
};
layout(set = 0, binding = 1, scalar) coherent volatile buffer VGB {
	FzbSVOVoxelData_PG vgb[];
};

/*
void atomicFloatMin(inout uint target, float value) {
	uint newVal = floatBitsToUint(value);
	uint prevVal;
	do {
		prevVal = target;
		if (uintBitsToFloat(prevVal) <= value) break;
	} while (atomicCompSwap(target, prevVal, newVal) != prevVal);
}
void atomicFloatMax(inout uint target, float value) {
	uint newVal = floatBitsToUint(value);
	uint prevVal;
	do {
		prevVal = target;
		if (uintBitsToFloat(prevVal) >= value) break;
	} while (atomicCompSwap(target, prevVal, newVal) != prevVal);
}

#define ATOMIC_FLOAT_MIN(target, value) \
	do { \
		float atomic_val = (value); \
		uint atomic_newVal = floatBitsToUint(atomic_val); \
		uint atomic_prevVal; \
		do { \
			atomic_prevVal = (target); \
			if (uintBitsToFloat(atomic_prevVal) <= atomic_val) break; \
		} while (atomicCompSwap((target), atomic_prevVal, atomic_newVal) != atomic_prevVal); \
	} while (false)

#define ATOMIC_FLOAT_MAX(target, value) \
	do { \
		float atomic_val = (value); \
		uint atomic_newVal = floatBitsToUint(atomic_val); \
		uint atomic_prevVal; \
		do { \
			atomic_prevVal = (target); \
			if (uintBitsToFloat(atomic_prevVal) >= atomic_val) break; \
		} while (atomicCompSwap((target), atomic_prevVal, atomic_newVal) != atomic_prevVal); \
	} while (false)
*/

void main() {
	ivec3 voxelIndex = ivec3((worldPos - vubo.voxelStartPos.xyz) / vubo.voxelSize_Num.xyz);
	uint voxelIndexU = voxelIndex.z * uint(vubo.voxelSize_Num.w * vubo.voxelSize_Num.w) + voxelIndex.y * uint(vubo.voxelSize_Num.w) + voxelIndex.x;

	FzbAABBUint AABB = vgb[voxelIndexU].AABB;
	atomicExchange(vgb[voxelIndexU].hasData, 1);
	float leftX = uintBitsToFloat(AABB.leftX);
	float leftY = uintBitsToFloat(AABB.leftY);
	float leftZ = uintBitsToFloat(AABB.leftZ);
	float rightX = uintBitsToFloat(AABB.rightX);
	float rightY = uintBitsToFloat(AABB.rightY);
	float rightZ = uintBitsToFloat(AABB.rightZ);

	if (worldPos.x < leftX) {
		uint preVal = AABB.leftX;
		uint newVal = floatBitsToUint(worldPos.x);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.leftX, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) <= worldPos.x) break;
			preVal = curVal;
		}
	}
	if (worldPos.y < leftY) {
		uint preVal = AABB.leftY;
		uint newVal = floatBitsToUint(worldPos.y);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.leftY, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) <= worldPos.y) break;
			preVal = curVal;
		}
	}
	if (worldPos.z < leftZ) {
		uint preVal = AABB.leftZ;
		uint newVal = floatBitsToUint(worldPos.z);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.leftZ, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) <= worldPos.z) break;
			preVal = curVal;
		}
	}
	if (worldPos.x > rightX) {
		uint preVal = AABB.rightX;
		uint newVal = floatBitsToUint(worldPos.x);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.rightX, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) >= worldPos.x) break;
			preVal = curVal;
		}
	}
	if (worldPos.y > rightY) {
		uint preVal = AABB.rightY;
		uint newVal = floatBitsToUint(worldPos.y);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.rightY, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) >= worldPos.y) break;
			preVal = curVal;
		}
	}
	if (worldPos.z > rightZ) {
		uint preVal = AABB.rightZ;
		uint newVal = floatBitsToUint(worldPos.z);
		uint curVal;
		while ((curVal = atomicCompSwap(vgb[voxelIndexU].AABB.rightZ, preVal, newVal)) != preVal) {
			if (uintBitsToFloat(curVal) >= worldPos.z) break;
			preVal = curVal;
		}
	}
	//ATOMIC_FLOAT_MIN(vgb[voxelIndexU].AABB.leftX, worldPos.x);
	//ATOMIC_FLOAT_MIN(vgb[voxelIndexU].AABB.leftY, worldPos.y);
	//ATOMIC_FLOAT_MIN(vgb[voxelIndexU].AABB.leftZ, worldPos.z);
	//ATOMIC_FLOAT_MAX(vgb[voxelIndexU].AABB.rightX, worldPos.x);
	//ATOMIC_FLOAT_MAX(vgb[voxelIndexU].AABB.rightY, worldPos.y);
	//ATOMIC_FLOAT_MAX(vgb[voxelIndexU].AABB.rightZ, worldPos.z);
}
