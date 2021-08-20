using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public readonly struct Light
    {
        public readonly Point3 Position;

        public readonly Color3 Ambient;
        public readonly Color3 Diffuse;
        public readonly Color3 Specular;

        public Light(Point3 position, Color3 ambient, Color3 diffuse, Color3 specular)
        {
            Position = position;
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
        }
    };
}
