using SharpGfx.Primitives;

namespace SharpGfx;

public readonly struct Reflectance<T>
{
    public readonly float Alpha;
    public readonly T Diffuse;
    public readonly Color3 Specular;
    public readonly float Shininess;

    public Reflectance(float alpha, T diffuse, Color3 specular, float shininess)
    {
        Alpha = alpha;
        Diffuse = diffuse;
        Specular = specular;
        Shininess = shininess;
    }
}