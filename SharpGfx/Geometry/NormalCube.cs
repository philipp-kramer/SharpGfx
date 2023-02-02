using System.Linq;

namespace SharpGfx.Geometry;

public static class NormalCube
{
    public static float[] Vertices { get; } =
    {
        // front
        -1,-1, 1,
        1,-1, 1,
        1, 1, 1,
        -1, 1, 1,
        // right
        1,-1, 1,
        1,-1,-1,
        1, 1,-1,
        1, 1, 1,
        // back
        1,-1,-1,
        -1,-1,-1,
        -1, 1,-1,
        1, 1,-1,
        // left
        -1,-1,-1,
        -1,-1, 1,
        -1, 1, 1,
        -1, 1,-1,
        // top
        -1, 1, 1,
        1, 1, 1,
        1, 1,-1,
        -1, 1,-1,
        // bottom
        1,-1, 1,
        -1,-1, 1,
        -1,-1,-1,
        1,-1,-1,
    };

    public static float[] Normals { get; } =
    {
        // front
        0, 0, 1, 
        0, 0, 1,
        0, 0, 1,
        0, 0, 1, 
        // right
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0, 
        // back
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        // left
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
        // top
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        // bottom
        0,-1, 0,
        0,-1, 0,
        0,-1, 0,
        0,-1, 0,
    };

    public static float[] Tangent { get; } =
    {
        // front
        1, 0, 0, 
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        // right
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        // back
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
        // left
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        // top
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        // bottom
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
        -1, 0, 0,
    };

    public static float[] BiTangent { get; } =
    {
        // front
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        // right
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        // back
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        // left
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        // top
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        // bottom
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
        0, 0,-1,
    };

    public static float[] Texture { get; } = Enumerable
        .Repeat(new float[] { 0, 0, 1, 0, 1, 1, 0, 1 }, 6)
        .SelectMany(side => side)
        .ToArray();

    public static ushort[] Triangles { get; } =
    {
        // front
        0, 1, 2,
        2, 3, 0,
        // right
        4, 5, 6,
        6, 7, 4,
        // back
        8, 9, 10,
        10, 11, 8,
        // left
        12, 13, 14,
        14, 15, 12,
        // top
        16, 17, 18,
        18, 19, 16,
        // bottom
        20, 21, 22,
        22, 23, 20,
    };
}