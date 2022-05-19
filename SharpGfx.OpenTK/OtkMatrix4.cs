using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkMatrix4 : Matrix4
    {
        internal readonly global::OpenTK.Mathematics.Matrix4 Value;
        internal readonly Space _space;

        public OtkMatrix4(Space space, global::OpenTK.Mathematics.Matrix4 value)
        {
            _space = space;
            Value = value;
        }

        Space IPrimitive.Space => _space;
        public float this[int row, int col] => Value[row, col];

        Array Matrix4.Values => new[]
        {
            Value.M11, Value.M12, Value.M13, Value.M14,
            Value.M21, Value.M22, Value.M23, Value.M24,
            Value.M31, Value.M32, Value.M33, Value.M34,
            Value.M41, Value.M42, Value.M43, Value.M44,
        };

        public Matrix4 ToSpace(Space space)
        {
            return new OtkMatrix4(space, Value);
        }

        public Vector4 Mul(Vector4 r)
        {
            return new OtkVector4(_space, Value * ((OtkVector4) r).Value);
        }

        public Matrix4 Mul(Matrix4 r)
        {
            var omr = (OtkMatrix4) r;
            return new OtkMatrix4(omr._space, Value * omr.Value);
        }
    }
}