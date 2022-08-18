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

uniform Light light;
uniform Material material;

out vec4 fragColor;

void main()
{
	vec3 ambient = light.ambient * material.diffuse;
	
	vec3 normDir = normalize(normal);
	vec3 lightDir = normalize(light.position - fragPos); 
	float cosTheta = max(dot(normDir, lightDir), 0.0);
	vec3 diffuse = cosTheta * light.diffuse * material.diffuse;

	fragColor = vec4(ambient + diffuse, 1.0);
}
