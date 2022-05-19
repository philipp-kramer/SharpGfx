#version 450

in vec3 positionIn;
in vec2 texCoordIn;

uniform mat4 model;
uniform mat4 cameraView;
uniform mat4 projection;

out vec2 texCoord;

void main(void)
{
	const mat4 invView = inverse(cameraView);
	gl_Position = vec4(positionIn, 1.0) * invView * model * cameraView * projection;
    texCoord = texCoordIn;
}