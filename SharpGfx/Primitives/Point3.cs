namespace SharpGfx.Primitives;

public readonly struct Point3
{
    public readonly IVector3 Vector;
        
    public float X => Vector.X;
    public float Y => Vector.Y;
    public float Z => Vector.Z;

    public Point3(IVector3 vector)
    {
        Vector = vector;
    }

    public static Point3 operator +(Point3 l, IVector3 r)
    {
        return new Point3(l.Vector + r);
    }

    public static Point3 operator -(Point3 l, IVector3 r)
    {
        return new Point3(l.Vector - r);
    }

    public static IVector3 operator -(Point3 l, Point3 r)
    {
        return l.Vector - r.Vector;
    }

    public static Point3 Combine(float wa, Point3 a, Point3 b)
    {
        return a + wa * (b - a);
    }

    public static Point3 Center(Point3 a, Point3 b, Point3 c)
    {
        return a + 1f/3 * (b - a) + 1f/3 * (c - a);
    }

    public override string ToString()
    {
        return Vector.ToString() ?? string.Empty;
    }
}

internal static class Point3Extensions
{
    public static Point3 GetPoint3(this float[] values, Space space, long offset)
    {
        long index = 3 * offset;
        return new Point3(
            space.Vector3(
                values[index + 0],
                values[index + 1],
                values[index + 2]));
    }
}