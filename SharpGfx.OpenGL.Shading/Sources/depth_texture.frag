#version 410

in vec2 texCoord;

uniform sampler2D texUnit;

out vec4 fragColor;

float linearizedDepth(vec2 uv)
{
  float n = 0.1;   // camera z near
  float f = 100.0; // camera z far
  float z = texture2D(texUnit, uv).x;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main(void)
{
  float c = linearizedDepth(texCoord.xy);
  fragColor = vec4(c, c, c, 1.0);
}