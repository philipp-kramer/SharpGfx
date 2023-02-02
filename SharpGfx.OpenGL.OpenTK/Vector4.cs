using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.OpenTK;

internal class Vector4 : IVector4
{
    public readonly global::OpenTK.Mathematics.Vector4 Value;
    private readonly Space _space;

    public Vector4(Space space, global::OpenTK.Mathematics.Vector4 value)
    {
        _space = space;
        Value = value;
    }

    Space IPrimitive.Space => _space;
    public float X => Value.X;
    public float Y => Value.Y;
    public float Z => Value.Z;
    public float W => Value.W;
    public float this[int index] => Value[index];
    public float Length => Value.Length;
    public Array Values => new[] { Value.X, Value.Y, Value.Z, Value.W };
    public IVector3 Xyz => new Vector3(_space, Value.Xyz);

    IVector4 IVector4.Neg()
    {
        return new Vector4(_space, -Value);
    }

    IVector4 IVector4.Add(IVector4 r)
    {
        var ovr = (Vector4)r;
        return new Vector4(ovr._space, Value + ovr.Value);
    }

    IVector4 IVector4.Sub(IVector4 r)
    {
        var ovr = (Vector4)r;
        return new Vector4(ovr._space, Value - ovr.Value);
    }

    IVector4 IVector4.Mul(float scalar)
    {
        return new Vector4(_space, Value * scalar);
    }

    IVector4 IVector4.Mul(IVector4 r)
    {
        var ovr = (Vector4)r;
        return new Vector4(ovr._space, Value * ovr.Value);
    }

    IVector4 IVector4.Mul(Primitives.Matrix4 r)
    {
        var omr = (Matrix4)r;
        return new Vector4(omr.Space, Value * omr.Value);
    }

    public float Dot(IVector4 r)
    {
        var ovr = (Vector4)r;
        return global::OpenTK.Mathematics.Vector4.Dot(Value, ovr.Value);
    }

    public IVector4 Normalized()
    {
        return new Vector4(_space, Value.Normalized());
    }
}