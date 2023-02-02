using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host;

public readonly struct HostVector3 : IVector3
{
    private readonly Space _space;
    private readonly float[] _values;

    public float X => _values[0];
    public float Y => _values[1];
    public float Z => _values[2];

    public HostVector3(Space space, float x, float y, float z)
    {
        _space = space;
        _values = new[] { x, y, z };
    }

    Space IPrimitive.Space => _space;
    public float this[int index] => index switch { 0 => X, 1 => Y, 2 => Z, _ => throw new ArgumentOutOfRangeException(nameof(index)) };
    public float Length => MathF.Sqrt(Dot(this));
    public IVector2 Xy => new HostVector2(_space, X, Y);
    public IVector2 Xz => new HostVector2(_space, X, Z);
    public IVector2 Yz => new HostVector2(_space, Y, Z);

    public Array Values => _values;

    public IVector4 Extend(Space space, float w)
    {
        return new HostVector4(space, X, Y, Z, w);
    }

    IVector3 IVector3.Neg()
    {
        return new HostVector3(_space, -X, -Y, -Z);
    }

    IVector3 IVector3.Add(IVector3 r)
    {
        return new HostVector3(r.Space, X + r.X, Y + r.Y, Z + r.Z);
    }

    IVector3 IVector3.Sub(IVector3 r)
    {
        return new HostVector3(r.Space, X - r.X, Y - r.Y, Z - r.Z);
    }

    IVector3 IVector3.Mul(float scalar)
    {
        return new HostVector3(_space, scalar * X, scalar * Y, scalar * Z);
    }

    IVector3 IVector3.Mul(IVector3 r)
    {
        return new HostVector3(r.Space, X * r.X, Y * r.Y, Z * r.Z);
    }

    public IVector3 Cross(IVector3 r)
    {
        return new HostVector3(
            r.Space,
            Y * r.Z - Z * r.Y,
            Z * r.X - X * r.Z,
            X * r.Y - Y * r.X);
    }

    public float Dot(IVector3 r)
    {
        return X * r.X + Y * r.Y + Z * r.Z;
    }

    public IVector3 Normalized()
    {
        float invLength = 1 / Length;
        return new HostVector3(_space, X * invLength, Y * invLength, Z * invLength);
    }
}