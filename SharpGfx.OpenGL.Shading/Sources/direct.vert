#version 410

in vec3 positionIn;

void main(void)
{
	gl_Position = vec4(positionIn, 1.0);
}