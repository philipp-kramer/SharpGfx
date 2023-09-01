using System;
using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry;

public static class Normals
{
    internal readonly ref struct HostMatrix23
    {
        public float[,] Elements { get; }

        public HostMatrix23(float[,] elements)
        {
            Elements = elements;
        }

        public HostMatrix23(IVector3 row0, IVector3 row1)
        {
            Elements = new float[2, 3];
            Elements[0, 0] = row0.X;
            Elements[0, 1] = row0.Y;
            Elements[0, 2] = row0.Z;
            Elements[1, 0] = row1.X;
            Elements[1, 1] = row1.Y;
            Elements[1, 2] = row1.Z;
        }

        public float this[int row, int col] => Elements[row, col];
    }

    public static IVector3 Reflect(this IVector3 incident, IVector3 normal)
    {
        if (Math.Abs(normal.Length - 1f) > 10 * float.Epsilon) throw new InvalidOperationException("vector must be normalized");

        return incident - 2 * IVector3.Dot(incident, normal) * normal;
    }

    public static float[] FromVertices(Space space, float[] xyz)
    {
        var normals = new float[xyz.Length];
        int count = xyz.Length / 3;
        for (int i = 0; i < count; i += 3)
        {
            var a = xyz.GetPoint3(space, i + 0);
            var b = xyz.GetPoint3(space, i + 1);
            var c = xyz.GetPoint3(space, i + 2);
            var n = GetN(a, b, c);
            for (int j = i; j < i + 3; j++)
            {
                int index = 3 * j;
                normals[index + 0] = n[0];
                normals[index + 1] = n[1];
                normals[index + 2] = n[2];
            }
        }

        return normals;
    }

    public static (float[], float[]) TangentsFromVertices(Space space, float[] xyz, float[] uv)
    {
        var tangents = new float[xyz.Length];
        var biTangents = new float[xyz.Length];
        int count = xyz.Length / 3;
        for (int i = 0; i < count; i += 3)
        {
            var t = uv.GetPoint2(space, i);
            var duv = new HostMatrix2(
                space, 
                uv.GetPoint2(space, i + 1) - t, 
                uv.GetPoint2(space, i + 2) - t);

            var p = xyz.GetPoint3(space, i);
            var dp = new HostMatrix23(
                xyz.GetPoint3(space, i + 1) - p,
                xyz.GetPoint3(space, i + 2) - p);

            var inverted = (HostMatrix2) duv.Inverted();
            var tb = Mul(inverted, dp);
            var il0 = 1 / GetRowLength(tb, 0);
            var il1 = 1 / GetRowLength(tb, 1);

            for (int j = i; j < i + 3; j++)
            {
                int index = 3 * j;
                tangents[index + 0] = il0 * tb[0, 0];
                tangents[index + 1] = il0 * tb[0, 1];
                tangents[index + 2] = il0 * tb[0, 2];
                biTangents[index + 0] = il1 * tb[1, 0];
                biTangents[index + 1] = il1 * tb[1, 1];
                biTangents[index + 2] = il1 * tb[1, 2];
            }
        }

        return (tangents, biTangents);
    }

    private static float GetRowLength(HostMatrix23 tb, int row)
    {
        float tb00 = tb[row, 0];
        float tb01 = tb[row, 1];
        float tb02 = tb[row, 2];
        return MathF.Sqrt(tb00 * tb00 + tb01 * tb01 + tb02 * tb02);
    }

    private static HostMatrix23 Mul(HostMatrix2 l, HostMatrix23 r)
    {
        float[,] result = new float[2, 3];

        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                float sum = 0;
                for (int k = 0; k < 2; k++)
                {
                    sum += l[row, k] * r[k, col];
                }
                result[row, col] = sum;
            }
        }

        return new HostMatrix23(result);
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
            var vertex0 = triangles[i + 0];
            var vertex1 = triangles[i + 1];
            var vertex2 = triangles[i + 2];
            var a = vertices.GetPoint3(space, Convert.ToInt64(vertex0));
            var b = vertices.GetPoint3(space, Convert.ToInt64(vertex1));
            var c = vertices.GetPoint3(space, Convert.ToInt64(vertex2));
            var n = GetN(a, b, c);
            AddVector(vertex0, normals, n);
            AddVector(vertex1, normals, n);
            AddVector(vertex2, normals, n);
        }

        for (int i = 0; i < normals.Length / 3; i++)
        {
            int index = 3 * i;
            float x = normals[index];
            float y = normals[index + 1];
            float z = normals[index + 2];
            float l = MathF.Sqrt(x * x + y * y + z * z);
            normals[index + 0] /= l;
            normals[index + 1] /= l;
            normals[index + 2] /= l;
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