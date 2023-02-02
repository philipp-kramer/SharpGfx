namespace SharpGfx.Geometry;

public static class UniformCube
{
    public static float[] Vertices { get; } =
    {
        // front
        -1,-1, 1,
        1,-1, 1,
        1, 1, 1,
        -1, 1, 1,
        // back
        1,-1,-1,
        -1,-1,-1,
        -1, 1,-1,
        1, 1,-1,
    };

    public static ushort[] Triangles { get; } =
    {
        // front
        0, 1, 2,
        2, 3, 0,
        // right
        1, 4, 7,
        7, 2, 1,
        // back
        4, 5, 6,
        6, 7, 4,
        // left
        5, 0, 3,
        3, 6, 5,
        // top
        3, 2, 7,
        7, 6, 3,
        // bottom
        1, 0, 5,
        5, 4, 1
    };
}