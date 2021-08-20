using System.Collections.Generic;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Space
    {
        public Domain Domain { get; }

        protected Space(Domain domain)
        {
            Domain = domain;
        }

        public abstract Vector2 Zero2 { get; }
        public abstract Vector2 Unit2X { get; }
        public abstract Vector2 Unit2Y { get; }

        public abstract Vector3 Zero3 { get; }
        public abstract Vector3 Unit3X { get; }
        public abstract Vector3 Unit3Y { get; }
        public abstract Vector3 Unit3Z { get; }

        public abstract Vector4 Zero4 { get; }
        public abstract Matrix4 Identity4 { get; }

        public abstract Vector2 Vector2(float x, float y);

        public abstract Vector3 Vector3(float x, float y, float z);

        public abstract Vector4 Vector4(float x, float y, float z, float w);

        public abstract Matrix2 Matrix2(
            float a00, float a01,
            float a10, float a11);
        public abstract Matrix2 Rotation2(float angle);

        public abstract Matrix3 Matrix3(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22);

        public abstract Matrix3 RotationZ3(float angle);

        public abstract Matrix4 Matrix4(
            float a00, float a01, float a02, float a03,
            float a10, float a11, float a12, float a13,
            float a20, float a21, float a22, float a23,
            float a30, float a31, float a32, float a33);
        public abstract Matrix4 Scale4(float s);
        public abstract Matrix4 Scale4(float x, float y, float z);
        public abstract Matrix4 Scale4(Vector3 s);
        public abstract Matrix4 Translation4(Vector3 p);
        public abstract Matrix4 RotationX4(float angle);
        public abstract Matrix4 RotationY4(float angle);
        public abstract Matrix4 RotationZ4(float angle);
    }

    public static class SpaceExtensions
    {
        private static readonly Dictionary<Space, Point2> _origin2 = new Dictionary<Space, Point2>();
        private static readonly Dictionary<Space, Point3> _origin3 = new Dictionary<Space, Point3>();

        public static Point2 Origin2(this Space space)
        {
            if (!_origin2.TryGetValue(space, out var origin))
            {
                origin = new Point2(space.Zero2);
                _origin2.Add(space, origin);
            }
            return origin;
        }

        public static Point3 Origin3(this Space space)
        {
            if (!_origin3.TryGetValue(space, out var origin))
            {
                origin = new Point3(space.Zero3);
                _origin3.Add(space, origin);
            }
            return origin;
        }

        public static Point2 Point2(this Space space, float x, float y)
        {
            return new Point2(space.Vector2(x, y));
        }

        public static Point3 Point3(this Space space, float x, float y, float z)
        {
            return new Point3(space.Vector3(x, y, z));
        }
    }
}
