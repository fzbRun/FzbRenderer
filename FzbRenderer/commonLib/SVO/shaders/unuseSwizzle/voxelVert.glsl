#version 450

layout(location = 0) in vec3 pos_in;
//layout(location = 0) out vec3 worldPos;

layout(set = 0, binding = 0) uniform voxelUniformBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
} vubo;

void main() {
	gl_Position = vubo.model * vec4(pos_in, 1.0f);
	//worldPos = (cubo.model * vec4(pos_in, 1.0f)).xyz;
}