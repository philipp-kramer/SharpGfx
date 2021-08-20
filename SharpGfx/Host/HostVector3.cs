using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostVector3 : Vector3
    {
        private readonly Space _space;
        public float X { get; }
        public float Y { get; }
        public float Z { get; }

        public HostVector3(Space space, float x, float y, float z)
        {
            _space = space;
            X = x;
            Y = y;
            Z = z;
        }

        Space Primitive.Space => _space;
        public float this[int index] => index switch { 0 => X, 1 => Y, 2 => Z, _ => throw new ArgumentOutOfRangeException(nameof(index)) };
        public float Length => MathF.Sqrt(Dot(this));
        public Vector2 Xy => new HostVector2(_space, X, Y);
        public Vector2 Xz => new HostVector2(_space, X, Z);
        public Vector2 Yz => new HostVector2(_space, Y, Z);

        public Array Values => new[] { X, Y, Z };

        public Vector4 Extend(float w)
        {
            return new HostVector4(_space, X, Y, Z, w);
        }

        Vector3 Vector3.Neg()
        {
            return new HostVector3(_space, -X, -Y, -Z);
        }

        Vector3 Vector3.Add(Vector3 r)
        {
            return new HostVector3(r.Space, X + r.X, Y + r.Y, Z + r.Z);
        }

        Vector3 Vector3.Sub(Vector3 r)
        {
            return new HostVector3(r.Space, X - r.X, Y - r.Y, Z - r.Z);
        }

        Vector3 Vector3.Mul(float scalar)
        {
            return new HostVector3(_space, scalar * X, scalar * Y, scalar * Z);
        }

        Vector3 Vector3.Mul(Vector3 r)
        {
            return new HostVector3(r.Space, X * r.X, Y * r.Y, Z * r.Z);
        }

        public Vector3 Cross(Vector3 r)
        {
            return new HostVector3(
                r.Space,
                Y * r.Z - Z * r.Y,
                Z * r.X - X * r.Z,
                X * r.Y - Y * r.X);
        }

        public float Dot(Vector3 r)
        {
            return X * r.X + Y * r.Y + Z * r.Z;
        }

        public Vector3 Normalized()
        {
            float invLength = 1 / Length;
            return new HostVector3(_space, X * invLength, Y * invLength, Z * invLength);
        }
    }
}
