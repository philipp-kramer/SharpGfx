namespace SharpGfx.OpenGL.Materials;

public class TextureMaterial : OpenGlMaterial
{
    private readonly TextureHandle _handle;
    private readonly int _unit;


    public TextureMaterial(GlApi gl, string vertexShader, string fragShader, TextureHandle handle, int unit)
        : base(gl, vertexShader, fragShader)
    {
        _handle = handle;
        _unit = unit;
        DoInContext(() => Set("texUnit", _unit));
    }

    public override void Apply()
    {
        _handle.ActivateTexture(_unit);
        base.Apply();
    }

    public override void UnApply()
    {
        base.UnApply();
        GL.ClearTexture(_unit);
    }

    protected override void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
        base.Dispose(disposing);
    }

    private void ReleaseUnmanagedResources()
    {
        _handle.DeleteTexture();
    }
}