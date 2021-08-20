using System;
using System.Collections.Generic;

namespace SharpGfx.Geometry
{
    public static class Circle
    {
        public static float[] GetTriangleVertices(int sectors)
        {
            var vertices = new List<float>();

            float angle = 0;
            var deltaAngle = 2 * MathF.PI / sectors;
            for (int sector = 0; sector < sectors; sector++)
            {
                vertices.AddRange(new[] { 0f, 0f, 0f });
                AddCirclePoint(vertices, angle);
                angle += deltaAngle;
                AddCirclePoint(vertices, angle);
            }

            return vertices.ToArray();
        }

        private static void AddCirclePoint(List<float> vertices, float angle)
        {
            vertices.Add(MathF.Cos(angle));
            vertices.Add(0);
            vertices.Add(MathF.Sin(angle));
        }

        public static float[] GetCakeVertices(int sectors)
        {
            var vertices = new float[3 * (sectors + 1)];
            int i = 0;
            vertices[i++] = 0;
            vertices[i++] = 0;
            vertices[i++] = 0;

            float angle = 0;
            var deltaAngle = 2 * MathF.PI / sectors;
            for (int sector = 0; sector < sectors; sector++)
            {
                vertices[i++] = MathF.Cos(angle);
                vertices[i++] = MathF.Sin(angle);
                vertices[i++] = 0;
                angle += deltaAngle;
            }

            return vertices;
        }

        public static ushort[] GetCakeIndices(int sectors)
        {
            var indices = new ushort[3 * sectors];
            int i = 0;
            for (int sector = 0; sector < sectors; sector++)
            {
                indices[i++] = 0;
                indices[i++] = (ushort)(1 + sector);
                indices[i++] = (ushort)(1 + (sector + 1) % sectors);
            }

            return indices;
        }

        public static float[] GetIsoVertices(int rings)
        {
            var vertices = new float[(rings * (rings + 1) / 2 * 6) * 3 + 3];
            int i = 0;
            vertices[i++] = 0;
            vertices[i++] = 0;
            vertices[i++] = 0;

            float r = 0;
            float deltaR = 0.5f / rings;
            int sectors = 6;
            for (int ring = 1; ring <= rings; ring++)
            {
                r += deltaR;
                float angle = 0;
                var deltaAngle = 2 * MathF.PI / sectors;
                for (int sector = 0; sector < sectors; sector++)
                {
                    vertices[i++] = MathF.Cos(angle) * r;
                    vertices[i++] = 0;
                    vertices[i++] = MathF.Sin(angle) * r;
                    angle += deltaAngle;
                }

                sectors += 6;
            }

            return vertices;
        }

        public static ushort[] GetIsoIndices(int rings)
        {
            var indices = new List<ushort>();
            SetIsoIndices(rings, indices);
            return indices.ToArray();
        }

        internal static void SetIsoIndices(int rings, List<ushort> indices)
        {
            int sectors = 6;
            for (int sector = 0; sector < sectors; sector++)
            {
                indices.Add(0);
                indices.Add((ushort) (1 + sector));
                indices.Add((ushort) (1 + (sector + 1) % sectors));
            }

            int innerStart = 1;
            for (int ring = 2; ring <= rings; ring++)
            {
                int outerStart = innerStart + sectors;
                int outerSector = 0;
                int moduloCorders = sectors / 6;
                for (int innerSector = 0; innerSector < sectors; innerSector++)
                {
                    int innerNext = (innerSector + 1) % sectors;
                    int outerNext = (outerSector + 1) % (sectors + 6);

                    indices.Add((ushort) (innerStart + innerSector));
                    indices.Add((ushort) (outerStart + outerSector));
                    indices.Add((ushort) (outerStart + outerNext));

                    indices.Add((ushort) (innerStart + innerSector));
                    indices.Add((ushort) (outerStart + outerNext));
                    indices.Add((ushort) (innerStart + innerNext));

                    outerSector = outerNext;
                    if (innerSector % moduloCorders == moduloCorders - 1)
                    {
                        indices.Add((ushort) (innerStart + innerNext));
                        indices.Add((ushort) (outerStart + outerSector));
                        outerSector = (outerSector + 1) % (sectors + 6);
                        indices.Add((ushort) (outerStart + outerSector));
                    }
                }

                sectors += 6;
                innerStart = outerStart;
            }
        }
    }
}
