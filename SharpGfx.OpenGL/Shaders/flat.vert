#version 410

in vec3 positionIn;

uniform mat4 model;
uniform mat4 cameraView;
uniform mat4 projection;

out vec3 fragPos;

void main(void)
{
	vec4 fragPos4 = vec4(positionIn, 1.0) * model;
	fragPos = vec3(fragPos4);
	gl_Position = fragPos4 * cameraView * projection;
}