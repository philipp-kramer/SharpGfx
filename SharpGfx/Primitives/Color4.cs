namespace SharpGfx.Primitives
{
    public readonly struct Color4
    {
        public readonly Vector4 Vector;

        public float R => Vector.X;
        public float G => Vector.Y;
        public float B => Vector.Z;
        public float A => Vector.W;

        internal Color4(Vector4 vector)
        {
            Vector = vector;
        }

        public static Color4 operator *(Color4 l, float r)
        {
            return new Color4(l.Vector * r);
        }

        public static Color4 Combine(float wa, Color4 a, Color4 b)
        {
            return new Color4(a.Vector + wa * (b.Vector - a.Vector));
        }
    }
}