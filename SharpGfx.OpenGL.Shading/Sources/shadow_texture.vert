#version 410

in vec3 positionIn;
in vec2 texCoordIn;

uniform mat4 model;
uniform mat4 cameraView;
uniform mat4 projection;
uniform mat4 lightViewProjection;

out vec3 fragPos;
out vec4 fragPosLightSpace;
out vec2 texCoord;

void main(void)
{
	vec4 fragPos4 = vec4(positionIn, 1.0) * model;
	fragPos = vec3(fragPos4);
	fragPosLightSpace = fragPos4 * lightViewProjection;
    texCoord = texCoordIn;
	gl_Position = fragPos4 * cameraView * projection;
}