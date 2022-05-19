using System;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class Normals
    {
        public static IVector3 Reflect(this IVector3 incident, IVector3 normal)
        {
            if (Math.Abs(normal.Length - 1f) > 10 * float.Epsilon) throw new InvalidOperationException("vector must be normalized");

            return incident - 2 * IVector3.Dot(incident, normal) * normal;
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
                var n = GetN(a, b, c);
                for (int j = i; j < i + 9; j++)
                {
                    normals[j] = n[j % 3];
                }

                i += 9;
            }

            return normals;
        }

        public static float[] FromTriangles<TIndex>(
            Space space, 
            float[] vertices, 
            TIndex[] triangles)
            where TIndex : struct
        {
            var normals = new float[vertices.Length];

            for (int i = 0; i < triangles.Length; i += 3)
            {
                var vertex0 = triangles[i];
                var vertex1 = triangles[i + 1];
                var vertex2 = triangles[i + 2];
                var a = vertices.GetPoint3(space, 3 * Convert.ToInt64(vertex0));
                var b = vertices.GetPoint3(space, 3 * Convert.ToInt64(vertex1));
                var c = vertices.GetPoint3(space, 3 * Convert.ToInt64(vertex2));
                var n = GetN(a, b, c);
                AddVector(vertex0, normals, n);
                AddVector(vertex1, normals, n);
                AddVector(vertex2, normals, n);
            }

            for (int i = 0; i < normals.Length; i += 3)
            {
                float x = normals[i];
                float y = normals[i + 1];
                float z = normals[i + 2];
                float l = MathF.Sqrt(x * x + y * y + z * z);
                normals[i] /= l;
                normals[i + 1] /= l;
                normals[i + 2] /= l;
            }

            return normals;
        }

        private static IVector3 GetN(Point3 a, Point3 b, Point3 c)
        {
            return IVector3
                .Cross(b - a, c - a)
                .Normalized();
        }

        private static void AddVector<TIndex>(TIndex vertex, float[] vertices, IVector3 value)
            where TIndex : struct
        {
            var vi = 3 * Convert.ToInt32(vertex);
            vertices[vi + 0] += value[0];
            vertices[vi + 1] += value[1];
            vertices[vi + 2] += value[2];
        }
    }
}
