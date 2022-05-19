using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkVector3 : IVector3
    {
        public readonly global::OpenTK.Mathematics.Vector3 Value;
        private readonly Space _space;

        public OtkVector3(Space space, global::OpenTK.Mathematics.Vector3 value)
        {
            _space = space;
            Value = value;
        }

        Space IPrimitive.Space => _space;
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
            return new OtkVector4(_space, new global::OpenTK.Mathematics.Vector4(Value, w));
        }

        IVector3 IVector3.Neg()
        {
            return new OtkVector3(_space, -Value);
        }

        IVector3 IVector3.Add(IVector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value + ovr.Value);
        }

        IVector3 IVector3.Sub(IVector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value - ovr.Value);
        }

        IVector3 IVector3.Mul(float scalar)
        {
            return new OtkVector3(_space, Value * scalar);
        }

        IVector3 IVector3.Mul(IVector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, Value * ovr.Value);
        }

        public IVector3 Cross(IVector3 r)
        {
            var ovr = (OtkVector3)r;
            return new OtkVector3(ovr._space, global::OpenTK.Mathematics.Vector3.Cross(Value, ovr.Value));
        }

        public float Dot(IVector3 r)
        {
            return global::OpenTK.Mathematics.Vector3.Dot(Value, ((OtkVector3)r).Value);
        }

        public IVector3 Normalized()
        {
            return new OtkVector3(_space, Value.Normalized());
        }
    }
}
