using System;

namespace SharpGfx.Geometry
{
    public static class Cube
    {
        public static float[] Vertices { get; } =
        {
            // front
            -1,-1, 1,
             1,-1, 1,
             1, 1, 1,
            -1, 1, 1,
            // back
            -1,-1,-1,
             1,-1,-1,
             1, 1,-1,
            -1, 1,-1,
        };

        // TODO: fix orientation
        public static float[] VerticesWithNormals { get; } =
        {
            // front
            -1,-1, 1, 0, 0, 1, 
             1,-1, 1, 0, 0, 1,
             1, 1, 1, 0, 0, 1,
            -1, 1, 1, 0, 0, 1, 
            // right
             1,-1,-1, 1, 0, 0,
             1, 1,-1, 1, 0, 0,
             1, 1, 1, 1, 0, 0,
             1,-1, 1, 1, 0, 0, 
            // back
            -1,-1,-1, 0, 0,-1,
             1,-1,-1, 0, 0,-1,
             1, 1,-1, 0, 0,-1,
            -1, 1,-1, 0, 0,-1,
            // left
            -1,-1,-1,-1, 0, 0,
            -1, 1,-1,-1, 0, 0,
            -1, 1, 1,-1, 0, 0,
            -1,-1, 1,-1, 0, 0,
            // bottom
            -1,-1,-1, 0,-1, 0,
             1,-1,-1, 0,-1, 0,
             1,-1, 1, 0,-1, 0,
            -1,-1, 1, 0,-1, 0,
            // top
            -1, 1,-1, 0, 1, 0,
             1, 1,-1, 0, 1, 0,
             1, 1, 1, 0, 1, 0,
            -1, 1, 1, 0, 1, 0,
        };

        // TODO: fix orientation
        public static float[] VerticesWithTexture { get; } =
        {
            // front
            -1,-1, 1, 0, 0,
             1,-1, 1, 1, 0,
             1, 1, 1, 1, 1,
            -1, 1, 1, 0, 1, 
            // right
            1,-1,-1, 0, 0,
            1, 1,-1, 1, 0,
            1, 1, 1, 1, 1,
            1,-1, 1, 0, 1, 
            // back
            -1,-1,-1, 0, 0,
             1,-1,-1, 1, 0,
             1, 1,-1, 1, 1,
            -1, 1,-1, 0, 1,
            // left
            -1,-1,-1, 0, 0,
            -1, 1,-1, 1, 0,
            -1, 1, 1, 1, 1,
            -1,-1, 1, 0, 1,
            // bottom
            -1,-1,-1, 0, 0,
             1,-1,-1, 1, 0,
             1,-1, 1, 1, 1,
            -1,-1, 1, 0, 1,
            // top
            -1, 1,-1, 0, 0,
             1, 1,-1, 1, 0,
             1, 1, 1, 1, 1,
            -1, 1, 1, 0, 1,
        };

        public static ushort[] Triangles { get; } =
        {
            // front
            0, 1, 2,
            2, 3, 0,
            // right
            1, 5, 6,
            6, 2, 1,
            // back
            7, 6, 5,
            5, 4, 7,
            // left
            4, 0, 3,
            3, 7, 4,
            // bottom
            4, 5, 1,
            1, 0, 4,
            // top
            3, 2, 6,
            6, 7, 3
        };

        public static ushort[] SeparateTriangles { get; } =
        {
            // front
            0, 1, 2,
            2, 3, 0,
            // right
            4, 5, 6,
            6, 7, 4,
            // back
            11, 10, 9,
            9, 8, 11,
            // left
            13, 12, 14,
            15, 14, 12,
            // bottom
            16, 17, 18,
            18, 19, 16,
            // top
            20, 21, 22,
            22, 23, 20,
        };

        public static void Debug(int components)
        {
            var vertices = VerticesWithTexture;
            var triangles = Triangles;
            for (int t = 0; t < triangles.Length / 3; t++)
            {
                for (int v = 0; v < 3; v++)
                {
                    var index = triangles[t * 3 + v];

                    for (int c = 0; c < 3; c++)
                    {
                        Console.Write(vertices[index * components + c]);
                        Console.Write(" ");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
    }
}