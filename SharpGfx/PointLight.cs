using SharpGfx.Primitives;

namespace SharpGfx;

public struct PointLight
{
    public Point3 Position { get; set; }
    public Color3 Color { get; set; }

    public PointLight(Point3 position, Color3 color)
    {
        Position = position;
        Color = color;
    }
}