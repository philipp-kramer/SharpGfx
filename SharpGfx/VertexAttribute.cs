namespace SharpGfx
{
    public readonly struct VertexAttribute
    {
        public readonly string Parameter;
        public readonly VertexBuffer Buffer;
        public readonly int Rank;

        public VertexAttribute(string parameter, VertexBuffer buffer, int rank)
        {
            Parameter = parameter;
            Buffer = buffer;
            Rank = rank;
        }
    }
}