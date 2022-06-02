using System;

namespace SharpGfx.Primitives
{
    public interface IVector2 : IPrimitive
    {
        public float X { get; }
        public float Y { get; }
        public float Length { get; }

        protected IVector2 Add(IVector2 r);
        protected IVector2 Sub(IVector2 r);
        protected IVector2 Mul(float scalar);
        protected IVector2 Mul(IVector2 r);
        public float Dot(IVector2 r);
        public IVector2 Normalized();

        public static IVector2 operator +(IVector2 l, IVector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Add(r);
        }

        public static IVector2 operator -(IVector2 l, IVector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Sub(r);
        }

        public static IVector2 operator *(float scalar, IVector2 v)
        {
            return v.Mul(scalar);
        }

        public static IVector2 operator *(IVector2 v, float scalar)
        {
            return v.Mul(scalar);
        }

        public static IVector2 operator *(IVector2 l, IVector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }
        public static float Dot(IVector2 l, IVector2 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Dot(r);
        }
    }
}