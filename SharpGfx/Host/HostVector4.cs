using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostVector4 : Vector4
    {
        private readonly Space _space;

        public float X { get; }

        public float Y { get; }

        public float Z { get; }

        public float W { get; }

        public HostVector4(Space space, float x, float y, float z, float w)
        {
            _space = space;
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        Space IPrimitive.Space => _space;

        public float this[int index] => index switch { 0 => X, 1 => Y, 2 => Z, 3 => W, _ => throw new ArgumentOutOfRangeException(nameof(index)) };
        public float Length => MathF.Sqrt(Dot(this));
        public IVector3 Xyz => new HostVector3(_space, X, Y, Z);
        public Array Values => new[] { X, Y, Z, W };

        Vector4 Vector4.Neg()
        {
            return new HostVector4(_space, X, Y, Z, W);
        }

        Vector4 Vector4.Add(Vector4 r)
        {
            return new HostVector4(r.Space, X + r.X, Y + r.Y, Z + r.Z, W + r.W);
        }

        Vector4 Vector4.Sub(Vector4 r)
        {
            return new HostVector4(r.Space, X - r.X, Y - r.Y, Z - r.Z, W - r.W);
        }

        Vector4 Vector4.Mul(float scalar)
        {
            return new HostVector4(_space, scalar * X, scalar * Y, scalar * Z, scalar * W);
        }

        Vector4 Vector4.Mul(Vector4 r)
        {
            return new HostVector4(r.Space, X * r.X, Y * r.Y, Z * r.Z, W * r.W);
        }

        Vector4 Vector4.Mul(Matrix4 r)
        {
            return new HostVector4(
                _space,
                X * r[0, 0] + Y * r[1, 0] + Z * r[2, 0] + W * r[3, 0],
                X * r[0, 1] + Y * r[1, 1] + Z * r[2, 1] + W * r[3, 1],
                X * r[0, 2] + Y * r[1, 2] + Z * r[2, 2] + W * r[3, 2],
                X * r[0, 3] + Y * r[1, 3] + Z * r[2, 3] + W * r[3, 3]);
        }

        public float Dot(Vector4 r)
        {
            return X * r.X + Y * r.Y + Z * r.Z + W * r.W;
        }

        public Vector4 Normalized()
        {
            float invLength = 1 / Length;
            return new HostVector4(_space, X * invLength, Y * invLength, Z * invLength, W*invLength);
        }
    }
}