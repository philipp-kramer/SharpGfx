#version 410

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
};

struct Material {
    vec3 diffuse;
};

in vec3 fragPos;
in vec3 normal;
in vec2 texCoord;

uniform Light light;
uniform Material material;
uniform sampler2D texUnit;

out vec4 fragColor;

void main()
{
	vec3 normDir = normalize(normal);
	vec3 lightDir = normalize(light.position - fragPos); 
	float cosTheta = max(dot(normDir, lightDir), 0.0);

	vec3 diffuse = (cosTheta * light.diffuse + light.ambient)  * material.diffuse;
	fragColor = texture(texUnit, texCoord) * vec4(diffuse, 1.0);
}
