#version 410

in vec2 texCoord;
in vec4 fragPosLightSpace;

uniform sampler2D texUnit;
uniform sampler2D shadowUnit;
uniform vec3 ambient;

out vec4 fragColor;

float ShadowCalculation()
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowUnit, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    return currentDepth > closestDepth  ? 0.0 : 1.0;
}

void main()
{
    vec4 texColor = texture(texUnit, texCoord);
	fragColor = vec4((ambient + (1 - ambient) * ShadowCalculation()), 1.0) * texColor;
}
