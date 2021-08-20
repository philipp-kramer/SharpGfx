using System;

namespace SharpGfx.Geometry
{
    public static class Plane
    {
        public static float[] GetVertices(int nx, int nz, Func<float, float, float> f)
        {
            var vertices = new float[(nx + 1) * (nz + 1) * 3];
            float d = 2f / (nx + nz);
            float cx = -nx * d / 2f;
            int i = 0;
            for (int x = 0; x <= nx; x++)
            {
                float cz = -nz * d / 2f;
                for (int z = 0; z <= nz; z++)
                {
                    vertices[i++] = cx;
                    vertices[i++] = f(cx, cz);
                    vertices[i++] = cz;
                    cz += d;
                }

                cx += d;
            }

            return vertices;
        }

        public static float[] GetTexture(int nx, int nz, float scale)
        {
            var vertices = new float[(nx + 1) * (nz + 1) * 2];
            float d = 2f / (scale * (nx + nz));
            float cx = 0;
            int i = 0;
            for (int x = 0; x <= nx; x++)
            {
                float cz = 0;
                for (int z = 0; z <= nz; z++)
                {
                    vertices[i++] = cx;
                    vertices[i++] = cz;
                    cz += d;
                }
                cx += d;
            }

            return vertices;
        }

        public static ushort[] GetIndices(int nx, int nz, int diag)
        {
            ushort[] indices = new ushort[nx * nz * 2 * 3];
            int ix = 0;
            int i = 0;
            int deltaz = nz + 1;

            for (int x = 0; x < nx; x++)
            {
                for (int z = 0; z < nz; z++)
                {
                    indices[i++] = (ushort) (ix + z);
                    indices[i++] = (ushort) (ix + z + 1);
                    indices[i++] = (ushort) (ix + z + deltaz + diag);

                    indices[i++] = (ushort)(ix + z + 1 - diag);
                    indices[i++] = (ushort)(ix + z + deltaz + 1);
                    indices[i++] = (ushort)(ix + z + deltaz);
                }

                ix += deltaz;
            }
            return indices;
        }
    }
}
