using System;
using System.Collections.Generic;

namespace SharpGfx.Geometry
{
    // TODO: consider using equilateral triangles (rings with up and down teeth)
    public class Sphere
    {
        public static float[] GetVertices(int longitudeCount, int latitudeCount)
        {
            var vertices = new float[(longitudeCount + 1) * (latitudeCount + 1) * 3];

            float lngStep = 2 * MathF.PI / longitudeCount;
            float latStep = MathF.PI / latitudeCount;

            int i = 0;
            for (int lat = 0; lat <= latitudeCount; lat++)
            {
                float latitude = lat * latStep - MathF.PI / 2;
                float xz = MathF.Cos(latitude);
                float y = MathF.Sin(latitude); // vertex position

                // add (sectorCount+1) vertices per stack
                // the first and last vertices have same position and normal, but different tex coords
                float longitude = 0;
                for (int lng = 0; lng <= longitudeCount; lng++)
                {
                    vertices[i++] = xz * MathF.Cos(longitude); // x
                    vertices[i++] = y;
                    vertices[i++] = xz * MathF.Sin(longitude); // z
                    longitude += lngStep;
                }
            }

            return vertices;
        }

        public static ushort[] GetTriangles(int longitudeCount, int latitudeCount)
        {
            ushort[] triangles = new ushort[longitudeCount * latitudeCount * 2 * 3];
            int upperCapOffset = (latitudeCount - 1) * (longitudeCount + 1);

            int i = 0;
            for (ushort longitude = 0; longitude < longitudeCount; longitude++)
            {
                triangles[i++] = longitude;
                triangles[i++] = (ushort) (longitude + longitudeCount + 1);
                triangles[i++] = (ushort) (longitude + longitudeCount + 2);

                for (int latitude = 1; latitude < latitudeCount; latitude++)
                {
                    int k1 = longitude + latitude * (longitudeCount + 1);
                    int k2 = k1 + longitudeCount + 1;

                    // 2 triangles per patch
                    // k1 => k2 => k1+1
                    triangles[i++] = (ushort) k1;
                    triangles[i++] = (ushort) k2;
                    triangles[i++] = (ushort) (k1 + 1);

                    // k1+1 => k2 => k2+1
                    triangles[i++] = (ushort) (k1 + 1);
                    triangles[i++] = (ushort) k2;
                    triangles[i++] = (ushort) (k2 + 1);
                }

                triangles[i++] = (ushort) (upperCapOffset + longitude);
                triangles[i++] = (ushort) (upperCapOffset + longitude + 1);
                triangles[i++] = (ushort) (upperCapOffset + longitude + longitudeCount + 1);
            }

            return triangles;
        }

        public static float[] GetIsoVertices(int rings)
        {
            int length = 3 * FullIsoVerticesCount(rings);
            var vertices = new float[length];
            SetHemiIsoVertices(rings, 1, vertices, 0);
            SetHemiIsoVertices(rings,-1, vertices, 3 * HemiIsoVerticesCount(rings));
            return vertices;
        }

        private static void SetHemiIsoVertices(int rings, int sign, float[] vertices, int i)
        {
            vertices[i++] = 0;
            vertices[i++] = sign;
            vertices[i++] = 0;

            float latiAngle = 0.5f * MathF.PI;
            float deltaLatiAngle = latiAngle / rings;
            int sectors = 6;
            for (int ring = 1; ring <= rings - (1-sign) / 2; ring++)
            {
                latiAngle -= deltaLatiAngle;
                float sinLatiAngle = MathF.Sin(latiAngle);
                float cosLatiAngle = MathF.Cos(latiAngle);
                float longAngle = 0;
                var deltaLongAngle = 2 * MathF.PI / sectors;
                for (int sector = 0; sector < sectors; sector++)
                {
                    vertices[i++] = cosLatiAngle * MathF.Cos(longAngle);
                    vertices[i++] = sinLatiAngle * sign;
                    vertices[i++] = cosLatiAngle * MathF.Sin(longAngle);
                    longAngle += deltaLongAngle;
                }

                sectors += 6;
            }
        }

        public static ushort[] GetIsoTriangles(int rings)
        {
            var triangles = new List<ushort>();
            Circle.AddIsoTriangles(rings, triangles);

            int fullHemiIndicesCount = triangles.Count;
            int hemiIsoVerticesCount = HemiIsoVerticesCount(rings);
            int equatorVertexIndex = hemiIsoVerticesCount - EquatorIsoVerticesCount(rings);
            for (int i = 0; i < fullHemiIndicesCount; i++)
            {
                int index = triangles[i];
                if (triangles[i] < equatorVertexIndex)
                {
                    index = hemiIsoVerticesCount + triangles[i];
                }
                triangles.Add((ushort) index);
            }
            return triangles.ToArray();
        }

        private static int FullIsoVerticesCount(int rings)
        {
            int hemiCount = HemiIsoVerticesCount(rings);
            int equatorCount = EquatorIsoVerticesCount(rings);
            return 2 * hemiCount - equatorCount;
        }

        private static int EquatorIsoVerticesCount(int rings)
        {
            return rings * 6;
        }

        private static int HemiIsoVerticesCount(int rings)
        {
            return rings * (rings + 1) / 2 * 6 + 1;
        }
    }
}
