/*
���glsls������ȡ���ֹ��պ���Ӱ
�򵥵�����Դ���۹⡢ƽ�й⡢���Դ������
���ӵģ������Ǹ���ȫ�ֹ����㷨������ͨ��������glslд�����ǿ��������glsl���ҵ����ӣ�
*/

//Ŀǰֻ�ȴ�����Դ����
vec3 blindFong(vec3 i, vec3 h, vec3 normal, vec3 albedo, vec3 lightStrength) {
	float ambient = 0.1f;
	float diffuse = max(dot(i, normal), 0.0f);
	vec3 specular = vec3(pow(max(dot(h, normal), 0.0f), 32));

	return ((ambient + diffuse) * albedo + specular) * lightStrength;
}

vec4 getIllumination(vec3 o, vec3 normal, vec3 albedo) {
	vec3 radiance = vec3(0.0f);
	uint lightNum = lubo.lightNum;
	for (int index = 0; index < lightNum; index++) {
		LightDate light = lubo.lightData[index];
		vec3 i = normalize(light.pos.xyz - vertexWorldPos);
		vec3 h = normalize(i + o);

		radiance += blindFong(i, h, normal, albedo, light.strength.xyz);
	}

	return vec4(radiance, 1.0f);
}

