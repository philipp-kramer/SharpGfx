using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostMatrix3 : Matrix3
    {
        private readonly float[,] _elements;
        private readonly Space _space;

        public HostMatrix3(Space space, float[,] elements)
        {
            _space = space;
            _elements = elements;
        }

        Space IPrimitive.Space => _space;
        public float this[int row, int col] => _elements[row, col];

        public IVector3 Mul(IVector3 r)
        {
            return new HostVector3(
                r.Space,
                _elements[0, 0] * r.X + _elements[0, 1] * r.Y + _elements[0, 2] * r.Z,
                _elements[1, 0] * r.X + _elements[1, 1] * r.Y + _elements[1, 2] * r.Z,
                _elements[2, 0] * r.X + _elements[2, 1] * r.Y + _elements[2, 2] * r.Z);
        }
    }
}
