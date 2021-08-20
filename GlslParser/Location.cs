namespace GlslParser
{
    public struct Location
    {
        public int Start { get; }
        public int End { get; }

        public Location(int start, int end)
        {
            Start = start;
            End = end;
        }

        public override string ToString()
        {
            return $"({Start}, {End})";
        }
    }
}