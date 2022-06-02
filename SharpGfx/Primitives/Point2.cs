namespace SharpGfx.Primitives
{
    public readonly struct Point2
    {
        public readonly IVector2 Relative;

        public float X => Relative.X;
        public float Y => Relative.Y;

        public Point2(IVector2 relative)
        {
            Relative = relative;
        }

        public static Point2 operator +(Point2 l, IVector2 r)
        {
            return new Point2(l.Relative + r);
        }

        public static Point2 operator -(Point2 l, IVector2 r)
        {
            return new Point2(l.Relative - r);
        }

        public static IVector2 operator -(Point2 l, Point2 r)
        {
            return l.Relative - r.Relative;
        }

        public static Point2 Combine(float wa, Point2 a, Point2 b)
        {
            return a + wa * (b - a);
        }
    }
}
