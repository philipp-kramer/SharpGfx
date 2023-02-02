using System;

namespace SharpGfx;

public abstract class Body : IDisposable
{
    public Material Material { get; }
    public int VertexCount { get; }

    private int _referenceCount;

    protected Body(Material material, int vertexCount)
    {
        Material = material;
        VertexCount = vertexCount;
    }

    public void Use()
    {
        _referenceCount++;
    }

    public void Unuse()
    {
        _referenceCount--;

        switch (_referenceCount)
        {
            case 0:
                Dispose();
                break;

            case < 0:
                throw new InvalidOperationException();
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing) Material.Dispose();
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(true);
    }
}