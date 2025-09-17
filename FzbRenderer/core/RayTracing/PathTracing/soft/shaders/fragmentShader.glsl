layout(set = 0, binding = 0) uniform PathTracingSetting{
	uint screenWidth;
	uint screenHeight;
};
layout(set = 0, binding = 0, std430) readonly buffer PathTracingBuffer {
	vec4 result[];
};

layout(location = 0) out vec4 fragColor;

void main() {
	vec2 texel = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
	uint texelX = uint(texel.x * screenWidth);
	uint texelY = uint(texel.y * screenHeight);
	uint resultIndex = texelX + texelY * screenWidth;
	//fragColor = vec4(result[resultIndex].xyz, 1.0f);
	fragColor = vec4(texel, 0.0f, 1.0f);
}