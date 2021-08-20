using System;

namespace SharpGfx.Geometry
{
    public static class Cube
    {
        public static float[] GetVertices()
        {
            return new[] {
                // front
                -1f,-1f, 1f,
                 1f,-1f, 1f,
                 1f, 1f, 1f,
                -1f, 1f, 1f,
                // back
                -1f,-1f,-1f,
                 1f,-1f,-1f,
                 1f, 1f,-1f,
                -1f, 1f,-1f,
            };
        }

        public static float[] GetVerticesWithTexture()
        {
            return new[] {
                // front
                -1f,-1f, 1f,-1f,-1f,
                 1f,-1f, 1f, 1f,-1f,
                 1f, 1f, 1f, 1f, 1f,
                -1f, 1f, 1f,-1f, 1f,  
                // back
                -1f,-1f,-1f,-1f,-1f,
                 1f,-1f,-1f, 1f,-1f,
                 1f, 1f,-1f, 1f, 1f,
                -1f, 1f,-1f,-1f, 1f,
            };
        }

        public static ushort[] GetIndices()
        {
            return new ushort[] {
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
        }

        public static float[] GetTriangleVertices()
        {
            return new[]{
                -1f, -1f, -1f,
                1f, -1f, -1f, 
                1f,  1f, -1f, 
                1f,  1f, -1f, 
                -1f,  1f, -1f,
                -1f, -1f, -1f,

                -1f, -1f,  1f,
                1f, -1f,  1f, 
                1f,  1f,  1f, 
                1f,  1f,  1f, 
                -1f,  1f,  1f,
                -1f, -1f,  1f,

                -1f,  1f,  1f,
                -1f,  1f, -1f,
                -1f, -1f, -1f,
                -1f, -1f, -1f,
                -1f, -1f,  1f,
                -1f,  1f,  1f,

                1f,  1f,  1f, 
                1f,  1f, -1f, 
                1f, -1f, -1f, 
                1f, -1f, -1f, 
                1f, -1f,  1f, 
                1f,  1f,  1f, 

                -1f, -1f, -1f,
                1f, -1f, -1f, 
                1f, -1f,  1f, 
                1f, -1f,  1f, 
                -1f, -1f,  1f,
                -1f, -1f, -1f,

                -1f,  1f, -1f,
                1f,  1f, -1f, 
                1f,  1f,  1f, 
                1f,  1f,  1f, 
                -1f,  1f,  1f,
                -1f,  1f, -1f,
            };
        }

        public static float[] GetTriangleTexture()
        {
            return new []{
                0.0f, 0.0f,
                1.0f, 0.0f,
                1.0f, 1.0f,
                1.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,

                0.0f, 0.0f,
                1.0f, 0.0f,
                1.0f, 1.0f,
                1.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,

                1.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,

                1.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,

                0.0f, 1.0f,
                1.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 0.0f,
                0.0f, 0.0f,
                0.0f, 1.0f,

                0.0f, 1.0f,
                1.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 0.0f,
                0.0f, 0.0f,
                0.0f, 1.0f
            };
        }

        public static void Debug(int components)
        {
            var vertices = GetVerticesWithTexture();
            var indices = GetIndices();
            for (int t = 0; t < indices.Length / 3; t++)
            {
                for (int v = 0; v < 3; v++)
                {
                    var index = indices[t * 3 + v];

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