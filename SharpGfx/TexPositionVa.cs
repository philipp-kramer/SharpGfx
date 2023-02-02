namespace SharpGfx;

public readonly struct TexPositionVa : IVertexAttribute
{
    public float[] Values { get; }
    public int Rank => 2;

    public TexPositionVa(float[] values)
    {
        Values = values;
    }
}