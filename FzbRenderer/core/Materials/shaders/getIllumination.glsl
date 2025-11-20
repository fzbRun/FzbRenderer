/*
这个glsls用来获取各种光照和阴影
简单的如点光源、聚光、平行光、面光源的照明
复杂的：可以是各种全局光照算法（可能通过单独的glsl写，但是可以在这个glsl中找到链接）
*/

//目前只先处理点光源好了
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

	radiance = normalize(radiance);
	return vec4(radiance, 1.0f);
}

