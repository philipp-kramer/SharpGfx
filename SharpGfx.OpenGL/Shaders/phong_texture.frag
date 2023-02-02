#version 410

in vec3 fragPos;
in vec3 normal;
in vec2 texCoord;

uniform vec3 ambient;
uniform vec3 lightPositions[16];
uniform vec3 lightColors[16];
uniform int lightCount;

uniform vec3 materialSpecular;
uniform float materialShininess;
uniform vec3 cameraPosition;
uniform sampler2D texUnit;

out vec4 fragColor;

void main()
{
	vec3 normDir = normalize(normal);
    vec3 cameraDir = normalize(cameraPosition - fragPos);

	vec3 color = ambient;

    for (int i = 0; i < lightCount; i++) {
		vec3 lightDir = normalize(lightPositions[i] - fragPos); 

		float cosTheta = max(dot(normDir, lightDir), 0.0);
		vec3 diffuse = cosTheta * lightColors[i];
		color = 1 - (1 - color)*(1 - diffuse);

		vec3 reflectDir = reflect(-lightDir, normDir);
		float intensity = pow(max(dot(cameraDir, reflectDir), 0.0), materialShininess);
		vec3 specular = intensity * lightColors[i] * materialSpecular;
		color = 1 - (1 - color)*(1 - specular);
	}

	fragColor = texture(texUnit, texCoord) * vec4(color, 1);
}
