namespace SharpGfx.Primitives;

public readonly struct Color3
{
    public readonly IVector3 Vector;

    public float R => Vector.X;
    public float G => Vector.Y;
    public float B => Vector.Z;

    internal Color3(IVector3 vector)
    {
        Vector = vector;
    }

    public Color4 GetColor4(float alpha)
    {
        return new Color4(Vector.Extend(Vector.Space, alpha));
    }

    public static Color3 operator *(Color3 l, float r)
    {
        return new Color3(l.Vector * r);
    }

    public static Color3 Combine(float wa, Color3 a, Color3 b)
    {
        return new Color3(a.Vector + wa * (b.Vector - a.Vector));
    }
}