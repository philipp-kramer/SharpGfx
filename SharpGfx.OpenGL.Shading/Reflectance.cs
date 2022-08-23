using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public readonly struct Reflectance
    {
        public readonly Color3 Diffuse;
        public readonly Color3 Specular;
        public readonly float Shininess;

        public Reflectance(Color3 diffuse, Color3 specular, float shininess)
        {
            Diffuse = diffuse;
            Specular = specular;
            Shininess = shininess;
        }
    }
}