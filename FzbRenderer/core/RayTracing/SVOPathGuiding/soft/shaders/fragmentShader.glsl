#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require

layout(set = 0, binding = 0) uniform SVOPathGuidingSettingUniformObject{
	uint screenWidth;
	uint screenHeight;
};
layout(set = 0, binding = 1, std430) readonly buffer PathTracingBuffer {
	vec4 result[];
};

layout(location = 0) out vec4 fragColor;

void main() {
	vec2 texel = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
	uint texelX = uint(texel.x * screenWidth);
	uint texelY = uint(texel.y * screenHeight);
	uint resultIndex = texelX + texelY * screenWidth;
	vec3 result = result[resultIndex].xyz;
	//if (length(result) > 1.0f) result = normalize(result);
	fragColor = vec4(result, 1.0f);
}