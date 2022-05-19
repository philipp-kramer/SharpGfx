using System;
using System.Collections.Generic;
using System.Text;

namespace SharpGfx.Geometry
{
    public readonly struct Interval
    {
        public readonly float Min;
        public readonly float Max;

        public Interval(float min, float max)
        {
            Min = min;
            Max = max;
        }
    }

    public static class BoundingBox
    {
        public static Interval[] GetAxisAligned(Space space, List<float> vertices, List<uint> triangles)
        {
            if (triangles.Count == 0) return null;

            uint t = triangles[0];
            int index = 3 * (int)t;
            var vx = vertices[index];
            var vy = vertices[index + 1];
            var vz = vertices[index + 2];
            var x = new Interval(vx, vx);
            var y = new Interval(vy, vy);
            var z = new Interval(vz, vz);

            for (var i = 1; i < triangles.Count; i++)
            {
                t = triangles[i];
                index = 3 * (int) t;
                vx = vertices[index];
                vy = vertices[index + 1];
                vz = vertices[index + 2];
                x = new Interval(MathF.Min(x.Min, vx), MathF.Max(x.Max, vx));
                y = new Interval(MathF.Min(y.Min, vy), MathF.Max(y.Max, vy));
                z = new Interval(MathF.Min(z.Min, vz), MathF.Max(z.Max, vz));
            }

            return new[] { x, y, z };
        }
    }
}
