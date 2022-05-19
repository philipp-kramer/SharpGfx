using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal class OtkVector4 : Vector4
    {
        public readonly global::OpenTK.Mathematics.Vector4 Value;
        private readonly Space _space;

        public OtkVector4(Space space, global::OpenTK.Mathematics.Vector4 value)
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
        public IVector3 Xyz => new OtkVector3(_space, Value.Xyz);

        Vector4 Vector4.Neg()
        {
            return new OtkVector4(_space, -Value);
        }

        Vector4 Vector4.Add(Vector4 r)
        {
            var ovr = (OtkVector4)r;
            return new OtkVector4(ovr._space, Value + ovr.Value);
        }

        Vector4 Vector4.Sub(Vector4 r)
        {
            var ovr = (OtkVector4)r;
            return new OtkVector4(ovr._space, Value - ovr.Value);
        }

        Vector4 Vector4.Mul(float scalar)
        {
            return new OtkVector4(_space, Value * scalar);
        }

        Vector4 Vector4.Mul(Vector4 r)
        {
            var ovr = (OtkVector4)r;
            return new OtkVector4(ovr._space, Value * ovr.Value);
        }

        Vector4 Vector4.Mul(Matrix4 r)
        {
            var omr = (OtkMatrix4)r;
            return new OtkVector4(omr._space, Value * omr.Value);
        }

        public float Dot(Vector4 r)
        {
            var ovr = (OtkVector4)r;
            return global::OpenTK.Mathematics.Vector4.Dot(Value, ovr.Value);
        }

        public Vector4 Normalized()
        {
            return new OtkVector4(_space, Value.Normalized());
        }
    }
}
