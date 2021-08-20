using System;

namespace SharpGfx.Primitives
{
    public interface Vector2 : Primitive
    {
        public float X { get; }
        public float Y { get; }
        public float Length { get; }

        protected Vector2 Add(Vector2 r);
        protected Vector2 Sub(Vector2 r);
        protected Vector2 Mul(float scalar);
        protected Vector2 Mul(Vector2 r);
        public float Dot(Vector2 r);
        public Vector2 Normalized();

        public static Vector2 operator +(Vector2 l, Vector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Add(r);
        }

        public static Vector2 operator -(Vector2 l, Vector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Sub(r);
        }

        public static Vector2 operator *(float scalar, Vector2 v)
        {
            return v.Mul(scalar);
        }

        public static Vector2 operator *(Vector2 v, float scalar)
        {
            return v.Mul(scalar);
        }

        public static Vector2 operator *(Vector2 l, Vector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }
        public static float Dot(Vector2 l, Vector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Dot(r);
        }
    }
}