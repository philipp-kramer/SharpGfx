using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host;

public readonly struct HostVector4 : IVector4
{
    private readonly Space _space;
    private readonly float[] _values;

    public float X => _values[0];
    public float Y => _values[1];
    public float Z => _values[2];

    public float W => _values[3];

    public HostVector4(Space space, float x, float y, float z, float w)
    {
        _space = space;
        _values = new[] { x, y, z, w };
    }

    Space IPrimitive.Space => _space;

    public float this[int index] => index switch { 0 => X, 1 => Y, 2 => Z, 3 => W, _ => throw new ArgumentOutOfRangeException(nameof(index)) };
    public float Length => MathF.Sqrt(Dot(this));
    public IVector3 Xyz => new HostVector3(_space, X, Y, Z);
    public Array Values => _values;

    IVector4 IVector4.Neg()
    {
        return new HostVector4(_space, X, Y, Z, W);
    }

    IVector4 IVector4.Add(IVector4 r)
    {
        return new HostVector4(r.Space, X + r.X, Y + r.Y, Z + r.Z, W + r.W);
    }

    IVector4 IVector4.Sub(IVector4 r)
    {
        return new HostVector4(r.Space, X - r.X, Y - r.Y, Z - r.Z, W - r.W);
    }

    IVector4 IVector4.Mul(float scalar)
    {
        return new HostVector4(_space, scalar * X, scalar * Y, scalar * Z, scalar * W);
    }

    IVector4 IVector4.Mul(IVector4 r)
    {
        return new HostVector4(r.Space, X * r.X, Y * r.Y, Z * r.Z, W * r.W);
    }

    IVector4 IVector4.Mul(Matrix4 r)
    {
        return new HostVector4(
            _space,
            X * r[0, 0] + Y * r[1, 0] + Z * r[2, 0] + W * r[3, 0],
            X * r[0, 1] + Y * r[1, 1] + Z * r[2, 1] + W * r[3, 1],
            X * r[0, 2] + Y * r[1, 2] + Z * r[2, 2] + W * r[3, 2],
            X * r[0, 3] + Y * r[1, 3] + Z * r[2, 3] + W * r[3, 3]);
    }

    public float Dot(IVector4 r)
    {
        return X * r.X + Y * r.Y + Z * r.Z + W * r.W;
    }

    public IVector4 Normalized()
    {
        float invLength = 1 / Length;
        return new HostVector4(_space, X * invLength, Y * invLength, Z * invLength, W*invLength);
    }
}