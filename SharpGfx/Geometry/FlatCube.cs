using System.Linq;

namespace SharpGfx.Geometry;

public static class FlatCube
{
    public static float[] Vertices { get; } =
    {
        // front
        -1,-1, 1,
        1,-1, 1,
        1, 1, 1,

        1, 1, 1,
        -1, 1, 1,
        -1,-1, 1,

        // right
        1,-1, 1,
        1,-1,-1,
        1, 1,-1,

        1, 1,-1,
        1, 1, 1,
        1,-1, 1,

        // back
        1,-1,-1,
        -1,-1,-1,
        -1, 1,-1,

        -1, 1,-1,
        1, 1,-1,
        1,-1,-1,

        // left
        -1,-1,-1,
        -1,-1, 1,
        -1, 1, 1,

        -1, 1, 1,
        -1, 1,-1,
        -1,-1,-1,

        // top
        -1, 1, 1,
        1, 1, 1,
        1, 1,-1,

        1, 1,-1,
        -1, 1,-1,
        -1, 1, 1,

        // bottom
        1,-1, 1,
        -1,-1, 1,
        -1,-1,-1,

        -1,-1,-1,
        1,-1,-1,
        1,-1, 1,
    };

    public static float[] Texture { get; } = Enumerable
        .Repeat(new float[] { 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0 }, 6)
        .SelectMany(side => side)
        .ToArray();
}