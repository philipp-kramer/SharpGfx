using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    public readonly struct HostVector2 : IVector2
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

        IVector2 IVector2.Add(IVector2 r)
        {
            return new HostVector2(r.Space, X + r.X, Y + r.Y);
        }

        IVector2 IVector2.Sub(IVector2 r)
        {
            return new HostVector2(r.Space, X - r.X, Y - r.Y);
        }

        IVector2 IVector2.Mul(float scalar)
        {
            return new HostVector2(_space, scalar * X, scalar * Y);
        }

        IVector2 IVector2.Mul(IVector2 r)
        {
            return new HostVector2(r.Space, X * r.X, Y * r.Y);
        }

        public float Dot(IVector2 r)
        {
            return X * r.X + Y * r.Y;
        }

        public IVector2 Normalized()
        {
            float invLength = 1 / MathF.Sqrt(Dot(this));
            return new HostVector2(_space, X * invLength, Y * invLength);
        }
    }
}