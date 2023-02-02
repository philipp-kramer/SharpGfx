using System;

namespace SharpGfx.Geometry;

public static class Patch
{
    public static float[] GetVertices(int nx, int nz, Func<float, float, float> y)
    {
        var vertices = new float[(nx + 1) * (nz + 1) * 3];
        float d = 2f / (nx + nz);
        float cz = -nz * d / 2f;
        float cx0 = -nx * d / 2f;
        int i = 0;
        for (int z = 0; z <= nz; z++)
        {
            float cx = cx0;

            for (int x = 0; x <= nx; x++)
            {
                vertices[i++] = cx;
                vertices[i++] = y(cx, cz);
                vertices[i++] = cz;
                cx += d;
            }

            cz += d;
        }

        return vertices;
    }

    public static float[] GetTexture(int nx, int nz)
    {
        var vertices = new float[(nx + 1) * (nz + 1) * 2];
        float d = 2f / (nx + nz);
        float cz = 0;
        int i = 0;
        for (int z = 0; z <= nz; z++)
        {
            float cx = 0;
            for (int x = 0; x <= nx; x++)
            {
                vertices[i++] = cx;
                vertices[i++] = cz;
                cx += d;
            }
            cz += d;
        }

        return vertices;
    }

    public static ushort[] GetTriangles(int nx, int nz)
    {
        var triangles = new ushort[nx * nz * 2 * 3];
        int t = 0;

        for (int z = 0; z < nz; z++)
        {
            for (int x = 0; x < nx; x++)
            {
                int i = z * (nx + 1) + x;
                triangles[t++] = (ushort) i;
                triangles[t++] = (ushort)(i + nx + 1);
                triangles[t++] = (ushort)(i + 1);

                triangles[t++] = (ushort)(i + nx + 2);
                triangles[t++] = (ushort)(i + 1);
                triangles[t++] = (ushort)(i + nx + 1);
            }
        }

        return triangles;
    }
}