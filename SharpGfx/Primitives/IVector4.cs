using System;

namespace SharpGfx.Primitives;

public interface IVector4 : IPrimitive
{
    public float X { get; }
    public float Y { get; }
    public float Z { get; }
    public float W { get; }
    public float this[int index] { get; }
    public float Length { get; }
    public Array Values { get; }
    public IVector3 Xyz { get; }

    protected IVector4 Neg();
    protected IVector4 Add(IVector4 r);
    protected IVector4 Sub(IVector4 r);
    protected IVector4 Mul(float scalar);
    protected IVector4 Mul(IVector4 r);
    protected IVector4 Mul(Matrix4 r);
    public float Dot(IVector4 r);
    public IVector4 Normalized();

    public static IVector4 operator -(IVector4 v)
    {
        return v.Neg();
    }

    public static IVector4 operator +(IVector4 l, IVector4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Add(r);
    }

    public static IVector4 operator -(IVector4 l, IVector4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Sub(r);
    }

    public static IVector4 operator *(float scalar, IVector4 v)
    {
        return v.Mul(scalar);
    }

    public static IVector4 operator *(IVector4 v, float scalar)
    {
        return v.Mul(scalar);
    }

    public static IVector4 operator *(IVector4 l, IVector4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Mul(r);
    }

    public static IVector4 operator *(IVector4 l, Matrix4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Mul(r);
    }
}