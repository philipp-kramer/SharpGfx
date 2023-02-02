namespace SharpGfx.OpenGL.Materials;

public class LambertColorTextureMaterial : LambertMaterial
{
    private readonly TextureHandle _handle;
    private readonly int _unit;

    public LambertColorTextureMaterial(GlApi gl, TextureHandle handle, Lighting lighting, int unit)
        : base(
            gl,
            Resources.GetShader("normal_texture.vert"),
            Resources.GetShader("lambert_texture.frag"),
            lighting)
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
        _handle.DeleteTexture();
        base.Dispose(disposing);
    }
}