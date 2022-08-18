#version 410

in vec2 texCoord;

uniform sampler2D texUnit;

out vec4 fragColor;

void main(void)
{
    fragColor = texture(texUnit, texCoord);
}