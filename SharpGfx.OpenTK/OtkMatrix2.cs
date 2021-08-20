using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkMatrix2 : Matrix2
    {
        private readonly global::OpenTK.Matrix2 _value;
        private readonly Space _space;

        public OtkMatrix2(Space space, global::OpenTK.Matrix2 value)
        {
            _space = space;
            _value = value;
        }

        public float this[int row, int col] => _value[row, col];

        Array Matrix2.Values => new []
        {
            _value.M11, _value.M12,
            _value.M21, _value.M22,
        };

        Space Primitive.Space => _space;
    }
}