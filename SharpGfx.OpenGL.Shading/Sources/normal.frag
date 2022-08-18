#version 410

#define COLOR_CORRECTION vec3(0, 0.2, 0.1)

in vec3 fragPos;
in vec3 normal;

out vec4 fragColor;

void main()
{	
	fragColor = vec4(0.5 + 0.5 * (normalize(normal) - COLOR_CORRECTION), 1.0);
}
