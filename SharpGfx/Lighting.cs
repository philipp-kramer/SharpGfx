using SharpGfx.Primitives;

namespace SharpGfx;

public class Lighting
{
    public Color3 Ambient { get; }
    public PointLight[] Lights { get; }

    public Lighting(Color3 ambient, params PointLight[] lights)
    {
        Ambient = ambient;
        Lights = lights;
    }
}