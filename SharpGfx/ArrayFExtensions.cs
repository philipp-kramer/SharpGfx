using SharpGfx.Primitives;

namespace SharpGfx
{
    internal static class ArrayFExtensions
    {
        public static float[] ToFloat(this short[] values)
        {
            var converted = new float[values.Length];
            for (long i = 0; i < values.Length; i++)
            {
                const float maxAmplitude = short.MaxValue + 1;
                converted[i] = values[i] / maxAmplitude;
            }

            return converted;
        }

        public static Point3 GetPoint3(this float[] values, Space space, long offset)
        {
            return new Point3(
                space.Vector3(
                    values[offset + 0],
                    values[offset + 1],
                    values[offset + 2]));
        }

        public static void Set(this float[] values, long offset, Vector3 value)
        {
            values[offset + 0] = value[0];
            values[offset + 1] = value[1];
            values[offset + 2] = value[2];
        }

        public static void Set(this float[] values, long offset, Point3 value)
        {
            values[offset + 0] = value.X;
            values[offset + 1] = value.Y;
            values[offset + 2] = value.Z;
        }
    }
}
