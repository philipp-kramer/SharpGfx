#version 410

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
	vec3 ambient = light.ambient * material.diffuse;

	vec3 normDir = normalize(normal);
	vec3 lightDir = normalize(light.position - fragPos); 
	float cosTheta = max(dot(normDir, lightDir), 0.0);
	vec3 diffuse = cosTheta * light.diffuse * material.diffuse;

    vec3 cameraDir = normalize(cameraPosition - fragPos);
    vec3 reflectDir = reflect(-lightDir, normDir);
    float intensity = pow(max(dot(cameraDir, reflectDir), 0.0), material.shininess);
    vec3 specular = intensity * light.specular * material.specular;

	fragColor = vec4(ambient + diffuse + specular, 1.0);
}
