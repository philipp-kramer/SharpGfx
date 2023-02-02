namespace SharpGfx.OpenGL;

internal class GlTextureHandle : TextureHandle
{
    private readonly GlApi _gl;
    internal readonly uint Handle;

    public GlTextureHandle(GlApi gl, uint handle)
    {
        _gl = gl;
        Handle = handle;
    }

    public override void ActivateTexture(int unit)
    {
        _gl.ActiveTexture(GlTextureUnit.Texture0 + unit);
        _gl.BindTexture(GlTextureTarget.Texture2D, Handle);
    }

    public override void DeleteTexture()
    {
        _gl.DeleteTexture(Handle);
    }
}