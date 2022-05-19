using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostVector2 : Vector2
    {
        private readonly Space _space;
        public float X { get; }
        public float Y { get; }
        public float Length => MathF.Sqrt(Dot(this));

        public HostVector2(Space space, float x, float y)
        {
            _space = space;
            X = x;
            Y = y;
        }

        Space IPrimitive.Space => _space;

        Vector2 Vector2.Add(Vector2 r)
        {
            return new HostVector2(r.Space, X + r.X, Y + r.Y);
        }

        Vector2 Vector2.Sub(Vector2 r)
        {
            return new HostVector2(r.Space, X - r.X, Y - r.Y);
        }

        Vector2 Vector2.Mul(float scalar)
        {
            return new HostVector2(_space, scalar * X, scalar * Y);
        }

        Vector2 Vector2.Mul(Vector2 r)
        {
            return new HostVector2(r.Space, X * r.X, Y * r.Y);
        }

        public float Dot(Vector2 r)
        {
            return X * r.X + Y * r.Y;
        }

        public Vector2 Normalized()
        {
            float invLength = 1 / MathF.Sqrt(Dot(this));
            return new HostVector2(_space, X * invLength, Y * invLength);
        }
    }
}