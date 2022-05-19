using System;
using SharpGfx.Host;

namespace SharpGfx.Geometry
{
    public static class Tetrahedron
    {
        public static ushort[] Triangles = {
            // the base triangle
            0, 1, 2,
            // the sides
            0, 1, 3,
            1, 2, 3,
            2, 0, 3
        };
        
        public static float[] GetVertices(float r)
        {
            var angle = 2f / 3 * MathF.PI;
            var rCos = r * MathF.Cos(angle);
            var rSin = r * MathF.Sin(angle);
            return new[]
            {
                r, 0, 0,
                rCos, 0, rSin,
                rCos, 0,-rSin,
                0, 1, 0
            };
        }

        public static (float[], uint[]) CreateFromPointVectors(float[] points, float[] directions, float scale)
        {
            var space = new HostSpace(Domain.World);
            var arrow = GetVertices(0.05f);
            var pointCount = points.Length / 3;
            var vertices = new float[pointCount * arrow.Length];
            var triangles = new uint[pointCount * Triangles.Length];
            int v = 0;
            int i = 0;
            uint indexOffset = 0;
            for (int p = 0; p < points.Length; p += 3)
            {
                for (int a = 0; a < arrow.Length; a += 3)
                {
                    float x = directions[p];
                    float y = directions[p + 1];
                    float z = directions[p + 2];
                    float x2 = x * x;
                    float y2 = y * y;
                    float z2 = z * z;
                    float xy = x*y;
                    float xz = x*z;
                    float yz = y*z;
                    float x2z2 = x2 + z2;
                    var rotation = space.Matrix4(
                        -y-z, x+z, x-y, 0,
                         x, y, z, 0,
                        xy - y2 - xz - z2, xy - yz - x2z2, x2z2 + xz + y2, 0,
                        0, 0, 0, 1);

                    var translation = space.Vector4(
                        points[p],
                        points[p + 1],
                        points[p + 2],
                        1);

                    var vertex = space.Vector4(
                        arrow[a],
                        arrow[a + 1],
                        arrow[a + 2],
                        1);

                    vertex *= scale;
                    vertex *= rotation;
                    vertex += translation;

                    vertices[v++] = vertex.X;
                    vertices[v++] = vertex.Y;
                    vertices[v++] = vertex.Z;
                }

                foreach (var index in Triangles)
                {
                    triangles[i++] = index + indexOffset;
                }

                indexOffset += 4;
            }

            return (vertices, triangles);
        }
    }
}
