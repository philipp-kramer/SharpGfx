#version 410

in vec3 fragPos;
in vec3 normal;

uniform vec3 ambient;
uniform vec3 lightPositions[16];
uniform vec3 lightColors[16];
uniform int lightCount;

uniform vec4 material;
uniform vec3 materialSpecular;
uniform float materialShininess;
uniform vec3 cameraPosition;

out vec4 fragColor;

void main()
{
	vec3 normDir = normalize(normal);
    vec3 cameraDir = normalize(cameraPosition - fragPos);

	vec3 color = ambient * material.xyz;

    for (int i = 0; i < lightCount; i++) {
		vec3 lightDir = normalize(lightPositions[i] - fragPos); 

		float cosTheta = max(dot(normDir, lightDir), 0.0);
		vec3 diffuse = cosTheta * lightColors[i] * material.xyz;
		color = 1 - (1 - color)*(1 - diffuse);

		vec3 reflectDir = reflect(-lightDir, normDir);
		float intensity = pow(max(dot(cameraDir, reflectDir), 0.0), materialShininess);
		vec3 specular = intensity * lightColors[i] * materialSpecular;
		color = 1 - (1 - color)*(1 - specular);
	}

	fragColor = vec4(color, material.w);
}
