#version 450

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec4 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
} cubo;

void main() {

	FragColor = vec4(normalize(worldPos.xyz), 1.0f);

}