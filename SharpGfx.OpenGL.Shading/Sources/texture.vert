#version 410

in vec3 positionIn;
in vec2 texCoordIn;

uniform mat4 model;
uniform mat4 cameraView;
uniform mat4 projection;

out vec2 texCoord;

void main(void)
{
	gl_Position = vec4(positionIn, 1.0) * model * cameraView * projection;
    texCoord = texCoordIn;
}