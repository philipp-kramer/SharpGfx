namespace SharpGfx.OpenGL;

public readonly struct GlVertexAttribute : IVertexAttribute
{
    public string Name { get; }
    public float[] Values { get; }
    public int Rank { get; }

    public GlVertexAttribute(string name, float[] values, int rank)
    {
        Name = name;
        Values = values;
        Rank = rank;
    }
}