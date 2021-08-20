using System;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class Cylinder
    {
        public static float[] GetVertices(Space space, int corners, int layers, Func<int, float> radius, Func<float, Point3> center, out float[] normals)
        {
            float phi = 2 * MathF.PI / corners;
            int coordsPerLayer = 3 * (corners + 1);
            var vertices = new float[(layers + 1) * coordsPerLayer];
            normals = new float[(layers + 1) * coordsPerLayer];

            for (int layer = 0; layer <= layers; layer++)
            {
                float angle = 0;
                var up = center(layer + 0.1f) - center(layer - 0.1f);
                up = up.Normalized();
                var d1 = space.Vector3(-up[1], up[0], up[2]);
                var d2 = Vector3.Cross(up, d1);
                d2 = d2.Normalized();
                var c = center(layer);
                for (int corner = 0; corner < coordsPerLayer; corner += 3)
                {
                    float cos = radius(layer) * MathF.Cos(angle);
                    float sin = radius(layer) * MathF.Sin(angle);
                    var vertex = c + (cos * d1 + sin * d2);
                    int refCorner = layer * coordsPerLayer + corner;
                    vertices.Set(refCorner, vertex);
                    normals.Set(refCorner, vertex - c);
                    angle += phi;
                }
            }

            return vertices;
        }

        public static ushort[] GetIndices(int corners, int layers)
        {
            var indices = new ushort[layers * 6 * corners];
            int virtCorners = corners + 1;
            int i = 0;
            for (int layer = 0; layer < layers; layer++)
            {
                for (int corner = 0; corner < corners; corner++)
                {
                    int refCorner = layer * virtCorners + corner;
                    indices[i++] = (ushort) refCorner;
                    indices[i++] = (ushort) (refCorner + 1);
                    indices[i++] = (ushort) (refCorner + virtCorners);
                    indices[i++] = (ushort) (refCorner + 1);
                    indices[i++] = (ushort) (refCorner + virtCorners + 1);
                    indices[i++] = (ushort) (refCorner + virtCorners);
                }
            }

            return indices;
        }

        public static float[] GetTextureVertices(int corners, int layers)
        {
            int coordsPerLayer = 2 * (corners + 1);
            var vertices = new float[(layers + 1) * coordsPerLayer];
            float tdl = 1f / layers;
            float tdc = 1f / corners;

            float tl = 0;
            for (int layer = 0; layer <= layers; layer++)
            {
                float tc = 0;
                for (int corner = 0; corner < coordsPerLayer; corner += 2)
                {
                    int refCorner = layer * coordsPerLayer + corner;
                    vertices[refCorner + 0] = tc;
                    vertices[refCorner + 1] = 1 - tl;
                    tc += tdc;
                }
                tl += tdl;
            }

            return vertices;
        }


    }
}
