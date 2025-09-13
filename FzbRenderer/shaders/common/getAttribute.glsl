/*
���glsl������ȡ��������
��Ϊ�кܶ���ԭ�򣬻�ȡһ�����Եķ�ʽ��ͬ����normal
1. ������ֱ��ͨ���������Ի�ȡ
2. ������ͨ��normalMap��ȡ
3. ������ͨ���淨�߻�ȡ
��ˣ����ǽ���ȡ���Եķ�ʽ��װ�����������ڲ���Ϣ
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

