using System;

namespace SharpGfx.Primitives;

public interface Matrix4 : IPrimitive
{
    public float[,] Elements { get; }

    public float this[int row, int col] { get; }

    public Matrix4 ToSpace(Space space);
    public IVector4 Mul(IVector4 r);
    public Matrix4 Mul(Matrix4 r);
    public Matrix4 Transposed();

    public static Matrix4 operator *(Matrix4 l, Matrix4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Mul(r);
    }

    public static IVector4 operator *(Matrix4 l, IVector4 r)
    {
        if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
        return l.Mul(r);
    }
}