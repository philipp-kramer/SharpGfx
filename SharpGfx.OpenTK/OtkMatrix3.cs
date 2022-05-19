using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkMatrix3 : Matrix3
    {
        private readonly global::OpenTK.Mathematics.Matrix3 _value;
        private readonly Space _space;

        public OtkMatrix3(Space space, global::OpenTK.Mathematics.Matrix3 value)
        {
            _space = space;
            _value = value;
        }

        Space IPrimitive.Space => _space;
    
        public float this[int row, int col] => _value[row, col];

        public IVector3 Mul(IVector3 r)
        {
            return new OtkVector3(_space, _value * ((OtkVector3)r).Value);
        }
    }
}
