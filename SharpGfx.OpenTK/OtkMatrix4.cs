using System.Runtime.CompilerServices;
using SharpGfx.Primitives;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenTK
{
    internal readonly struct OtkMatrix4 : Matrix4
    {
        internal readonly global::OpenTK.Mathematics.Matrix4 Value;
        internal readonly Space Space;

        public OtkMatrix4(Space space, global::OpenTK.Mathematics.Matrix4 value)
        {
            Space = space;
            Value = value;
        }

        Space IPrimitive.Space => Space;
        public float[,] Elements => new[,]
        {
            { Value.M11, Value.M12, Value.M13, Value.M14 },
            { Value.M21, Value.M22, Value.M23, Value.M24 },
            { Value.M31, Value.M32, Value.M33, Value.M34 },
            { Value.M41, Value.M42, Value.M43, Value.M44 }
        };

        public float this[int row, int col] => Value[row, col];

        public Matrix4 ToSpace(Space space)
        {
            return new OtkMatrix4(space, Value);
        }

        public Vector4 Mul(Vector4 r)
        {
            return new OtkVector4(Space, Value * ((OtkVector4) r).Value);
        }

        public Matrix4 Mul(Matrix4 r)
        {
            var omr = (OtkMatrix4) r;
            return new OtkMatrix4(omr.Space, Value * omr.Value);
        }

        public Matrix4 Transposed()
        {
            return new OtkMatrix4(
                Space,
                new global::OpenTK.Mathematics.Matrix4(
                    Value.Column0, 
                    Value.Column1, 
                    Value.Column2, 
                    Value.Column3));
        }
    }
}