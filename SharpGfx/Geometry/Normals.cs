using System;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class Normals
    {
        public static Vector3 Reflect(this Vector3 incident, Vector3 normal)
        {
            if (Math.Abs(normal.Length - 1f) > 10 * float.Epsilon) throw new InvalidOperationException("vector must be normalized");

            return incident - 2 * Vector3.Dot(incident, normal) * normal;
        }

        public static float[] FromVertices(Space space, float[] vertices, Point3 center)
        {
            int i = 0;
            var normals = new float[vertices.Length];
            while (i < vertices.Length)
            {
                var a = vertices.GetPoint3(space, i + 0);
                var b = vertices.GetPoint3(space, i + 3);
                var c = vertices.GetPoint3(space, i + 6);
                var n = GetN(center, a, b, c);
                for (int j = i; j < i + 9; j++)
                {
                    normals[j] = n[j % 3];
                }

                i += 9;
            }

            return normals;
        }

        public static float[] FromIndices(Space space, float[] vertices, ushort[] indices, Point3 center)
        {
            int i = 0;
            var normals = new float[vertices.Length];
            while (i < indices.Length)
            {
                var a = GetPoint3(space, vertices, indices, i);
                var b = GetPoint3(space, vertices, indices, i + 1);
                var c = GetPoint3(space, vertices, indices, i + 2);
                var n = GetN(center, a, b, c);
                SetVector(normals, indices, n, i);
                SetVector(normals, indices, n, i + 1);
                SetVector(normals, indices, n, i + 2);
                i += 3;
            }

            return normals;
        }

        private static Vector3 GetN(Point3 center, Point3 a, Point3 b, Point3 c)
        {
            var n = Vector3.Cross(b - a, c - a);
            var outside = a - center;
            if (Vector3.Dot(outside, n) < 0)
            {
                n = -n;
            }

            return n.Normalized();
        }

        private static Point3 GetPoint3(Space space, float[] vertices, ushort[] indices, int index)
        {
            return vertices.GetPoint3(space, indices[index] * 3);
        }

        private static void SetVector(float[] vertices, ushort[] indices, Vector3 value, int index)
        {
            var vi = indices[index] * 3;
            vertices[vi + 0] += 1f / 3f * value[0];
            vertices[vi + 1] += 1f / 3f * value[1];
            vertices[vi + 2] += 1f / 3f * value[2];
        }
    }
}
