using System;

namespace SharpGfx.Primitives
{
    public interface IVector3 : IPrimitive
    {
        public float X { get; }
        public float Y { get; }
        public float Z { get; }

        public float this[int index] { get; }
        public Array Values { get; }
        public float Length { get; }
        public IVector2 Xy { get; }
        public IVector2 Xz { get; }
        public IVector2 Yz { get; }

        public Vector4 Extend(float w);
        protected IVector3 Neg();
        protected IVector3 Add(IVector3 r);
        protected IVector3 Sub(IVector3 r);
        protected IVector3 Mul(float scalar);
        protected IVector3 Mul(IVector3 r);
        public IVector3 Cross(IVector3 r);
        public float Dot(IVector3 r);
        public IVector3 Normalized();

        public static IVector3 operator +(IVector3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Add(r);
        }

        public static IVector3 operator -(IVector3 v)
        {
            return v.Neg();
        }

        public static IVector3 operator -(IVector3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Sub(r);
        }

        public static IVector3 operator *(float scalar, IVector3 v)
        {
            return v.Mul(scalar);
        }

        public static IVector3 operator *(IVector3 v, float scalar)
        {
            return v.Mul(scalar);
        }

        public static IVector3 operator *(IVector3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

        public static IVector3 Cross(IVector3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Cross(r);
        }

        public static float Dot(IVector3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Dot(r);
        }
    }
}
