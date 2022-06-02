using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    public readonly struct HostMatrix2 : Matrix2
    {
        private readonly Space _space;

        public HostMatrix2(Space space, float[,] elements)
        {
            _space = space;
            Elements = elements;
        }

        Space IPrimitive.Space => _space;

        public float[,] Elements { get; }
        public float this[int row, int col] => Elements[row, col];
    }
}