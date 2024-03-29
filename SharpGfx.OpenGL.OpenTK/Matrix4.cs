﻿using System.Runtime.CompilerServices;
using SharpGfx.Primitives;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenGL.OpenTK;

internal readonly struct Matrix4 : Primitives.Matrix4
{
    internal readonly global::OpenTK.Mathematics.Matrix4 Value;
    internal readonly Space Space;

    public Matrix4(Space space, global::OpenTK.Mathematics.Matrix4 value)
    {
        Space = space;
        Value = value;
    }

    Space IPrimitive.Space => Space;
    public float[,] Elements => new[,]
    {
        { Value.M11, Value.M12, Value.M13, Value.M14 },
        { Value.M21, Value.M22, Value.M23, Value.M24 },
        { Value.M31, Value.M32, Value.M33, Value.M34 },
        { Value.M41, Value.M42, Value.M43, Value.M44 }
    };

    public float this[int row, int col] => Value[row, col];

    public Primitives.Matrix4 ToSpace(Space space)
    {
        return new Matrix4(space, Value);
    }

    public IVector4 Mul(IVector4 r)
    {
        return new Vector4(Space, Value * ((Vector4) r).Value);
    }

    public Primitives.Matrix4 Mul(Primitives.Matrix4 r)
    {
        var omr = (Matrix4) r;
        return new Matrix4(omr.Space, Value * omr.Value);
    }

    public Primitives.Matrix4 Transposed()
    {
        return new Matrix4(
            Space,
            new global::OpenTK.Mathematics.Matrix4(
                Value.Column0, 
                Value.Column1, 
                Value.Column2, 
                Value.Column3));
    }

    public override string ToString()
    {
        var e = Elements;
        var a = new[]
        {
            e[0, 0], e[0, 1], e[0, 2], e[0, 3],
            e[1, 0], e[1, 1], e[1, 2], e[1, 3],
            e[2, 0], e[2, 1], e[2, 2], e[2, 3],
            e[3, 0], e[3, 1], e[3, 2], e[3, 3]
        };
        return $"[{string.Join(' ', a)}]";
    }
}