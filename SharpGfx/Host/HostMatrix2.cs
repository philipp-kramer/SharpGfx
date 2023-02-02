using SharpGfx.Primitives;

namespace SharpGfx.Host;

public readonly struct HostMatrix2 : Matrix2
{
    private readonly Space _space;

    public HostMatrix2(Space space, float[,] elements)
    {
        _space = space;
        Elements = elements;
    }

    public HostMatrix2(Space space, IVector2 row0, IVector2 row1)
    {
        _space = space;
        Elements = new [,]
        {
            { row0.X, row0.Y },
            { row1.X, row1.Y }
        };
    }

    Space IPrimitive.Space => _space;

    public float[,] Elements { get; }
    public float this[int row, int col] => Elements[row, col];

    public Matrix2 Inverted()
    {
        float det = Elements[0, 0] * Elements[1, 1] - Elements[0, 1] * Elements[1, 0];
        return det * new HostMatrix2(_space, new[,]
        {
            { Elements[1, 1], -Elements[0, 1]},
            { -Elements[1, 0], Elements[0, 0]},
        });
    }

    public static Matrix2 operator *(float scalar, HostMatrix2 m)
    {
        return m;
    }
}