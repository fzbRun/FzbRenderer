/*
这个glsl用来获取各种属性
因为有很多宏的原因，获取一个属性的方式不同，如normal
1. 可以是直接通过顶点属性获取
2. 可以是通过normalMap获取
3. 可以是通过面法线获取
因此，我们将获取属性的方式封装起来，隐藏内部信息
*/
vec3 getNormal() {
#if defined(useFaceNormal)
	vec3 tangent = normalize(ddx(vertexWorldPos));
	vec3 bitangent = normalize(ddy(vertexWorldPos));
	return normalize(cross(tangent, bitangent));
#elif defined(useVertexNormal)
	#if defined(useNormalMap)
	vec3 normal = (texture(normalMap, vertexTexCoords).xyz) * 2.0f - 1.0f;
		#if defined(useVertexTangent)
	vec3 vertexBitangent = normalize(cross(vertexNormal, vertexTangent));
		#else
	vec3 tangent = normalize(ddx(vertexWorldPos));
	vec3 vertexBitangent = normalize(cross(vertexNormal, tangent));
	vec3 vertexTangent = normalize(cross(vertexBitangent, vertexNormal));
		#endif
	mat3 TBN = mat3(vertexTangent, vertexBitangent, vertexNormal);
	return normalize(TBN * normal);
	#else
	return vertexNormal;
	#endif
#elif defined(useNormalMap)
	vec3 normal_map = (texture(normalMap, vertexTexCoords).xyz) * 2.0f - 1.0f;
	vec3 tangent = normalize(ddx(vertexWorldPos));
	vec3 bitangent = normalize(ddy(vertexWorldPos));
	vec3 normal = normalize(cross(tangent, bitangent));
	mat3 TBN = mat3(tangent, bitangent, normal);
	return normalize(TBN * normal_map);
#else
	return vec3(0.0f);
#endif
}

vec4 getAlbedo() {
	vec4 vertexAlbedo = vec4(1.0f);
#ifdef useAlbedo
	vertexAlbedo = albedo;
#endif

#ifdef useAlbedoMap
	vertexAlbedo *= texture(albedoMap, vertexTexCoords);
#endif
	return vertexAlbedo;
}

