#extension GL_ARB_shader_viewport_layer_array : require
#extension GL_NV_viewport_array2 : require

layout(location = 0) in vec3 pos_in;

layout(location = 0) out vec3 worldPos;

layout(set = 0, binding = 0) uniform voxelUniformBufferObject{
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
} vubo;

layout(set = 1, binding = 0) uniform MeshBuffer{
	mat4 transformMatrix;
};

void main(){
	gl_ViewportMask[0] = 0x07;
	gl_Position = vubo.VP[0] * transformMatrix * vec4(pos_in, 1.0f);
	worldPos = (transformMatrix * vec4(pos_in, 1.0f)).xyz;
}