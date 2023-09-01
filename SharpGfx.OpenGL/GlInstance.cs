namespace SharpGfx.OpenGL;

internal class GlInstance : Instance
{
    private readonly GlApi _gl;

    internal GlSurface Surface { get; }

    public GlInstance(GlApi gl, Space space, string name, GlSurface surface)
        : base(space, name)
    {
        _gl = gl;
        Surface = surface;
        surface.Use();
    }

    public override void Render()
    {
        Surface.Draw();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Surface.Unuse();
            base.Dispose(true);
        }
    }

    ~GlInstance()
    {
        _gl.Add(() => Dispose(false));
    }
}