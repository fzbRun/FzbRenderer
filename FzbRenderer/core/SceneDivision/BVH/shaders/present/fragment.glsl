layout(location = 0) in vec3 worldPos;
layout(location = 0) out vec4 fragColor;

void main() {
	fragColor = vec4(worldPos, 1.0f);
}