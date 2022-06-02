namespace SharpGfx.Primitives
{
    public interface Matrix2 : IPrimitive
    {
        public float[,] Elements { get; }
        public float this[int row, int col] { get; }
    }
}
