namespace SharpGfx;

public readonly struct NormalVa : IVertexAttribute
{
    public float[] Values { get; }
    public int Rank => 3;

    public NormalVa(float[] values)
    {
        Values = values;
    }
}