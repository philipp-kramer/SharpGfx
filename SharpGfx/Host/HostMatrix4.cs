using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host
{
    internal readonly struct HostMatrix4 : Matrix4
    {
        private readonly float[,] _elements;
        private readonly Space _space;

        public HostMatrix4(Space space, float[,] elements)
        {
            _space = space;
            _elements = elements;
        }

        Space IPrimitive.Space => _space;
        public float this[int row, int col] => _elements[row, col];

        Array Matrix4.Values => _elements;

        public Matrix4 ToSpace(Space space)
        {
            return new HostMatrix4(space, _elements);
        }

        public Vector4 Mul(Vector4 r)
        {
            return new HostVector4(
                r.Space,
                _elements[0, 0] * r.X + _elements[0, 1] * r.Y + _elements[0, 2] * r.Z + _elements[0, 3] * r.W,
                _elements[1, 0] * r.X + _elements[1, 1] * r.Y + _elements[1, 2] * r.Z + _elements[1, 3] * r.W,
                _elements[2, 0] * r.X + _elements[2, 1] * r.Y + _elements[2, 2] * r.Z + _elements[2, 3] * r.W,
                _elements[3, 0] * r.X + _elements[3, 1] * r.Y + _elements[3, 2] * r.Z + _elements[3, 3] * r.W);
        }

        public Matrix4 Mul(Matrix4 r)
        {
            float[,] result = new float[4, 4];

            for (int row = 0; row < 4; row++)
            {
                for (int col = 0; col < 4; col++)
                {
                    float sum = 0;
                    for (int k = 0; k < 4; k++)
                    {
                        sum += _elements[row, k] * r[k, col];
                    }
                    result[row, col] = sum;
                }
            }

            return new HostMatrix4(r.Space, result);
        }
    }
}