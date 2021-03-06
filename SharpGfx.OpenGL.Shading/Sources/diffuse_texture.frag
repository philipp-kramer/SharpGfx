#version 450

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
	const vec3 ambient = light.ambient * material.diffuse;
	
	const vec3 normDir = normalize(normal);
	const vec3 lightDir = normalize(light.position - fragPos); 
	const float cosTheta = max(dot(normDir, lightDir), 0.0);
	const vec3 diffuse = cosTheta * light.diffuse * material.diffuse;

	fragColor = texture(texUnit, texCoord) * vec4(ambient + diffuse, 1.0);
}
