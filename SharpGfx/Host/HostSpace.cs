using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    public sealed class HostSpace : Space
    {
        public override Vector2 Zero2 { get; }
        public override Vector2 Unit2X { get; }
        public override Vector2 Unit2Y { get; }

        public override Vector3 Zero3 { get; }
        public override Vector3 Unit3X { get; }
        public override Vector3 Unit3Y { get; }
        public override Vector3 Unit3Z { get; }

        public override Vector4 Zero4 { get; }

        public override Matrix4 Identity4 { get; }

        public HostSpace(Domain domain)
            : base(domain)
        {
            Zero2 = new HostVector2(this, 0, 0);
            Unit2X = new HostVector2(this, 1, 0);
            Unit2Y = new HostVector2(this, 0, 1);
            Zero3 = new HostVector3(this, 0, 0, 0);
            Unit3X = new HostVector3(this, 1, 0, 0);
            Unit3Y = new HostVector3(this, 0, 1, 0);
            Unit3Z = new HostVector3(this, 0, 0, 1);
            Zero4 = new HostVector4(this, 0, 0, 0, 0);
            Identity4 = new HostMatrix4(
                this,
                new float[,]
                {
                    { 1, 0, 0, 0 },
                    { 0, 1, 0, 0 },
                    { 0, 0, 1, 0 },
                    { 0, 0, 0, 1 },
                });
        }

        public override Vector2 Vector2(float x, float y)
        {
            return new HostVector2(this, x, y);

        }

        public override Vector3 Vector3(float x, float y, float z)
        {
            return new HostVector3(this, x, y, z);
        }

        public override Vector4 Vector4(float x, float y, float z, float w)
        {
            return new HostVector4(this, x, y, z, w);
        }

        public override Matrix2 Matrix2(float a00, float a01, float a10, float a11)
        {
            return new HostMatrix2(
                this, 
                new[,]
                {
                    { a00, a01 }, 
                    {a10, a11 }
                });
        }

        public override Matrix2 Rotation2(float angle)
        {
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            return new HostMatrix2(
                this, 
                new[,]
                {
                    { cos, sin },
                    {-sin, cos }
                });
        }

        public override Matrix3 Matrix3(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22)
        {
            return new HostMatrix3(
                this, 
                new[,]
                {
                    { a00, a01, a02 },
                    { a10, a11, a12 },
                    { a20, a21, a22 },
                });
        }

        public override Matrix4 Matrix4(
            float a00, float a01, float a02, float a03,
            float a10, float a11, float a12, float a13,
            float a20, float a21, float a22, float a23,
            float a30, float a31, float a32, float a33)
        {
            return new HostMatrix4(
                this, 
                new[,]
                {
                    { a00, a01, a02, a03 },
                    { a10, a11, a12, a13 },
                    { a20, a21, a22, a23 },
                    { a30, a31, a32, a33 },
                });
        }

        public override Matrix4 Scale4(float s)
        {
            return Scale4(s, s, s);
        }

        public override Matrix4 Scale4(float x, float y, float z)
        {
            return new HostMatrix4(
                this, 
                new[,]
                {
                    { x, 0, 0, 0 },
                    { 0, y, 0, 0 },
                    { 0, 0, z, 0 },
                    { 0, 0, 0, 1 },
                });
        }

        public override Matrix4 Scale4(Vector3 s)
        {
            return Scale4(s.X, s.Y, s.Z);
        }

        public override Matrix4 Translation4(Vector3 p)
        {
            return new HostMatrix4(
                this, 
                new[,]
                {
                    {   1,   0,   0, 0 },
                    {   0,   1,   0, 0 },
                    {   0,   0,   1, 0 },
                    { p.X, p.Y, p.Z, 1 },
                });
        }

        public override Matrix4 RotationX4(float angle)
        {
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            return new HostMatrix4(
                this, 
                new[,]
                {
                    { 1,   0,   0, 0 },
                    { 0, cos, sin, 0 },
                    { 0,-sin, cos, 0 },
                    { 0,   0,   0, 1 }
                });
        }

        public override Matrix4 RotationY4(float angle)
        {
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            return new HostMatrix4(
                this, 
                new[,]
                {
                    { cos, 0,-sin, 0 },
                    {   0, 1,   0, 0 },
                    { sin, 0, cos, 0 },
                    {   0, 0,   0, 1 }
                });
        }

        public override Matrix3 RotationZ3(float angle)
        {
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            return new HostMatrix3(
                this,
                new[,]
                {
                    { cos, sin, 0},
                    {-sin, cos, 0},
                    {   0,   0, 1}
                });
        }

        public override Matrix4 RotationZ4(float angle)
        {
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            return new HostMatrix4(
                this, 
                new[,]
                {
                    { cos, sin, 0, 0},
                    {-sin, cos, 0, 0},
                    {  0,   0, 1, 0},
                    {  0,   0, 0, 1}
                });
        }
    }
}
