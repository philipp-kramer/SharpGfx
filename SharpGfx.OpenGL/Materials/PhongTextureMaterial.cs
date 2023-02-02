namespace SharpGfx.OpenGL.Materials;

public class PhongTextureMaterial : PhongMaterial
{
    private readonly TextureHandle _handle;
    private readonly int _unit;

    public PhongTextureMaterial(OpenGlDevice device, Reflectance<TextureHandle> reflectance, Lighting lighting, int unit)
        : base(
            device, 
            Resources.GetShader("normal_texture.vert"),
            Resources.GetShader("phong_texture.frag"),
            reflectance, lighting)
    {
        _handle = reflectance.Diffuse;
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