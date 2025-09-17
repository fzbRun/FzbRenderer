#include "./FzbCollisionDetection.cuh"

__device__ bool AABBCollisionDetection(FzbAABB AABB, FzbRay ray) {
	//判断光线是不是在场景内部发出的，如果是还不能直接抛弃别的场景
	//虽然会导致每次都与自身发射点的AABB再检测一次hitMesh
	if (ray.startPos.x > AABB.leftX && ray.startPos.x < AABB.rightX &&
		ray.startPos.y > AABB.leftY && ray.startPos.y < AABB.rightY &&
		ray.startPos.z > AABB.leftZ && ray.startPos.z < AABB.rightZ) {
		return true;
	}

	float maxInTime = -FLT_MAX;
	float minOutTime = FLT_MAX;

	if (ray.direction.x != 0) {	//直射与面都不考虑
		float leftX = (AABB.leftX - ray.startPos.x) / ray.direction.x;
		float rightX = (AABB.rightX - ray.startPos.x) / ray.direction.x;
		maxInTime = max(min(leftX, rightX), maxInTime);
		minOutTime = min(max(leftX, rightX), minOutTime);
	}
	else if (ray.startPos.x < AABB.leftX || ray.startPos.x > AABB.rightX) return false;

	if (ray.direction.y != 0) {
		float leftY = (AABB.leftY - ray.startPos.y) / ray.direction.y;
		float rightY = (AABB.rightY - ray.startPos.y) / ray.direction.y;
		maxInTime = max(min(leftY, rightY), maxInTime);
		minOutTime = min(max(leftY, rightY), minOutTime);
	}
	else if (ray.startPos.y < AABB.leftY || ray.startPos.y > AABB.rightY) return false;

	if (ray.direction.z != 0) {
		float leftZ = (AABB.leftZ - ray.startPos.z) / ray.direction.z;
		float rightZ = (AABB.rightZ - ray.startPos.z) / ray.direction.z;
		maxInTime = max(min(leftZ, rightZ), maxInTime);
		minOutTime = min(max(leftZ, rightZ), minOutTime);
	}
	else if (ray.startPos.z < AABB.leftZ || ray.startPos.z > AABB.rightZ) return false;

	if (minOutTime < maxInTime) return false;
	if (maxInTime > ray.depth) return false;	//深度测试
	return true;
}
__device__ bool meshCollisionDetection(FzbBvhNodeTriangleInfo& triangle, float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute) {
	//判断三角形是否被撞击到
	getTriangleVertexAttribute(vertices, materialTextures, triangle, triangleAttribute);
	//目前默认不折射
	if (glm::dot(triangleAttribute.normal, -ray.direction)) return false;	//打到了背面，不算撞到
	glm::vec3 E1 = triangleAttribute.pos1 - triangleAttribute.pos0;
	glm::vec3 E2 = triangleAttribute.pos2 - triangleAttribute.pos0;
	glm::vec3 S = ray.startPos - triangleAttribute.pos0;
	glm::vec3 S1 = glm::cross(ray.direction, E2);
	glm::vec3 S2 = cross(S, E1);
	glm::vec3 tbb = 1 / glm::dot(S1, E1) * glm::vec3(glm::dot(S2, E2), glm::dot(S1, S), glm::dot(S2, ray.direction));
	if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {	//打到了
		if (tbb.x > ray.depth) return false;	//深度测试没通过
		ray.depth = tbb.x;
	}
	else return false;
	return true;
}
__device__ bool sceneCollisionDetection(FzbBvhNode* __restrict__ bvhNodeArray, FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute, bool notOnlyDetection) {
	uint32_t nodeIndices[BVH_MAX_DEPTH] = { 0 };
	int queueTail = 0;
	FzbBvhNode node;
	FzbBvhNodeTriangleInfo hitTriangle;
	bool result = false;
	while (queueTail > -1) {
		uint32_t nodeIndex = nodeIndices[queueTail];
		node = bvhNodeArray[nodeIndex];
		if (AABBCollisionDetection(node.AABB, ray)) {
			if (node.leftNodeIndex == 0) {	//如果撞到叶节点了，则进行mesh碰撞检测
				FzbBvhNodeTriangleInfo triangle = bvhTriangleInfoArray[node.rightNodeIndex];
				if (meshCollisionDetection(triangle, vertices, materialTextures, ray, triangleAttribute)) {
					hitTriangle = triangle;
					result = true;
				}
				--queueTail;
			}
			else {	//不是叶节点则继续检测
				nodeIndices[queueTail] = node.rightNodeIndex;
				nodeIndices[++queueTail] = node.leftNodeIndex;
			}
		}
		else --queueTail;
	}
	if (result && notOnlyDetection) getTriangleMaterialAttribute(materialTextures, hitTriangle.materialIndex, triangleAttribute);
	return result;
}