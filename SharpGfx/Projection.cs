namespace SharpGfx;

public abstract class Projection
{
    public float Near { get; set; } = 0.1f;
    public float Far { get; set; } = 100f;
}

public class PerspectiveProjection : Projection
{
    public float FovY { get; set; }

    public PerspectiveProjection(float fovY)
    {
        FovY = fovY;
    }
}

public class OrthographicProjection : Projection
{
    public float PixelScale { get; set; }

    public OrthographicProjection(float pixelScale)
    {
        PixelScale = pixelScale;
    }
}
