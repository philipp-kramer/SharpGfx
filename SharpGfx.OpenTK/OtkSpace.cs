using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkSpace : Space
    {
        public override IVector2 Zero2 { get; }
        public override IVector2 Unit2X { get; }
        public override IVector2 Unit2Y { get; }

        public override IVector3 Zero3 { get; }
        public override IVector3 Unit3X { get; }
        public override IVector3 Unit3Y { get; }
        public override IVector3 Unit3Z { get; }

        public override Vector4 Zero4 { get; }

        public override Matrix4 Identity4 { get; }

        public OtkSpace(Domain domain)
            : base(domain)
        {
            Zero2 = new OtkVector2(this, global::OpenTK.Mathematics.Vector2.Zero);
            Unit2X = new OtkVector2(this, global::OpenTK.Mathematics.Vector2.UnitX);
            Unit2Y = new OtkVector2(this, global::OpenTK.Mathematics.Vector2.UnitY);
            Zero3 = new OtkVector3(this, global::OpenTK.Mathematics.Vector3.Zero);
            Unit3X = new OtkVector3(this, global::OpenTK.Mathematics.Vector3.UnitX);
            Unit3Y = new OtkVector3(this, global::OpenTK.Mathematics.Vector3.UnitY);
            Unit3Z = new OtkVector3(this, global::OpenTK.Mathematics.Vector3.UnitZ);
            Zero4 = new OtkVector4(this, global::OpenTK.Mathematics.Vector4.Zero);
            Identity4 = new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.Identity);
        }

        public override IVector2 Vector2(float x, float y)
        {
            return new OtkVector2(this, new global::OpenTK.Mathematics.Vector2(x, y));
        }

        public override IVector3 Vector3(float x, float y, float z)
        {
            return new OtkVector3(this, new global::OpenTK.Mathematics.Vector3(x, y, z));
        }

        public override Vector4 Vector4(float x, float y, float z, float w)
        {
            return new OtkVector4(this, new global::OpenTK.Mathematics.Vector4(x, y, z, w));
        }

        public override Matrix2 Matrix2(float a00, float a01, float a10, float a11)
        {
            return new OtkMatrix2(this, new global::OpenTK.Mathematics.Matrix2(a00, a01, a10, a11));
        }

        public override Matrix2 Rotation2(float angle)
        {
            return new OtkMatrix2(this, global::OpenTK.Mathematics.Matrix2.CreateRotation(angle));
        }

        public override Matrix3 Matrix3(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22)
        {
            return new OtkMatrix3(
                this, 
                new global::OpenTK.Mathematics.Matrix3(
                    a00, a01, a02,
                    a10, a11, a12,
                    a20, a21, a22));
        }

        public override Matrix3 RotationZ3(float angle)
        {
            return new OtkMatrix3(this, global::OpenTK.Mathematics.Matrix3.CreateRotationX(angle));
        }

        public override Matrix4 Matrix4(
            float a00, float a01, float a02, float a03,
            float a10, float a11, float a12, float a13,
            float a20, float a21, float a22, float a23,
            float a30, float a31, float a32, float a33)
        {
            return new OtkMatrix4(
                this, 
                new global::OpenTK.Mathematics.Matrix4(
                    a00, a01, a02, a03,
                    a10, a11, a12, a13,
                    a20, a21, a22, a23,
                    a30, a31, a32, a33));
        }

        public override Matrix4 Scale4(float s)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateScale(s));
        }

        public override Matrix4 Scale4(float x, float y, float z)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateScale(x, y, z));
        }

        public override Matrix4 Scale4(IVector3 s)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateScale(((OtkVector3)s).Value));
        }

        public override Matrix4 Translation4(IVector3 p)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateTranslation(((OtkVector3)p).Value));
        }

        public override Matrix4 RotationX4(float angle)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateRotationX(angle));
        }

        public override Matrix4 RotationY4(float angle)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateRotationY(angle));
        }

        public override Matrix4 RotationZ4(float angle)
        {
            return new OtkMatrix4(this, global::OpenTK.Mathematics.Matrix4.CreateRotationZ(angle));
        }
    }
}
