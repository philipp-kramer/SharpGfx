namespace SharpGfx.OpenGL;

internal class GlInstance : Instance
{
    private readonly GlApi _gl;

    internal GlBody Body { get; }

    public GlInstance(GlApi gl, Space space, string name, GlBody body)
        : base(space, name)
    {
        _gl = gl;
        Body = body;
        body.Use();
    }

    public override void Render()
    {
        Body.Draw();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Body.Unuse();
            base.Dispose(true);
        }
    }

    ~GlInstance()
    {
        _gl.Add(() => Dispose(false));
    }
}