namespace SharpGfx;

public readonly struct PositionVa : IVertexAttribute
{
    public float[] Values { get; }
    public int Rank { get; }

    public PositionVa(float[] values, int rank)
    {
        Values = values;
        Rank = rank;
    }
}