using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostMatrix2 : Matrix2
    {
        private readonly float[,] _elements;
        private readonly Space _space;

        public HostMatrix2(Space space, float[,] elements)
        {
            _space = space;
            _elements = elements;
        }

        Space Primitive.Space => _space;
        public float this[int row, int col] => _elements[row, col];

        Array Matrix2.Values => _elements;
    }
}