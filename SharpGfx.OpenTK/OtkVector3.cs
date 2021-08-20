using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkVector3 : Vector3
    {
        public readonly global::OpenTK.Vector3 Value;
        private readonly Space _space;

        public OtkVector3(Space space, global::OpenTK.Vector3 value)
        {
            _space = space;
            Value = value;
        }

        Space Primitive.Space => _space;
        public float this[int index] => Value[index];
        public Array Values => new[] { Value.X, Value.Y, Value.Z };
        public float X => Value.X;
        public float Y => Value.Y;
        public float Z => Value.Z;
        public float Length => Value.Length;
        public Vector2 Xy => new OtkVector2(_space, Value.Xy);
        public Vector2 Xz => new OtkVector2(_space, Value.Xz);
        public Vector2 Yz => new OtkVector2(_space, Value.Yz);

        public Vector4 Extend(float w)
        {
            return new OtkVector4(_space, new global::OpenTK.Vector4(Value, w));
        }

        Vector3 Vector3.Neg()
        {
            return new OtkVector3(_space, -Value);
        }

        Vector3 Vector3.Add(Vector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value + ovr.Value);
        }

        Vector3 Vector3.Sub(Vector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value - ovr.Value);
        }

        Vector3 Vector3.Mul(float scalar)
        {
            return new OtkVector3(_space, Value * scalar);
        }

        Vector3 Vector3.Mul(Vector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value * ovr.Value);
        }

        public Vector3 Cross(Vector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, global::OpenTK.Vector3.Cross(Value, ovr.Value));
        }

        public float Dot(Vector3 r)
        {
            return global::OpenTK.Vector3.Dot(Value, ((OtkVector3)r).Value);
        }

        public Vector3 Normalized()
        {
            return new OtkVector3(_space, Value.Normalized());
        }
    }
}
