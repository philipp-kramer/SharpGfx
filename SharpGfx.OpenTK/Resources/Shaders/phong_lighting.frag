#version 450

struct Light {
    vec3 position;
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Material {
    vec3 diffuse;
    vec3 specular;

    float shininess;
};

in vec3 fragPos;
in vec3 normal;

uniform Light light;
uniform Material material;
uniform vec3 cameraPosition;

out vec4 fragColor;

void main()
{
	const vec3 ambient = light.ambient * material.diffuse;

	const vec3 normDir = normalize(normal);
	const vec3 lightDir = normalize(light.position - fragPos); 
	const float cosTheta = max(dot(normDir, lightDir), 0.0);
	const vec3 diffuse = cosTheta * light.diffuse * material.diffuse;

    const vec3 cameraDir = normalize(cameraPosition - fragPos);
    const vec3 reflectDir = reflect(-lightDir, normDir);
    const float spec = pow(max(dot(cameraDir, reflectDir), 0.0), material.shininess);
    const vec3 specular = spec * light.specular * material.specular;

	fragColor = vec4(ambient + diffuse + specular, 1.0);
}
