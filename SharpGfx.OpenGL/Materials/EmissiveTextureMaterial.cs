namespace SharpGfx.OpenGL.Materials;

public class EmissiveTextureMaterial : TextureMaterial
{
    public EmissiveTextureMaterial(GlApi gl, TextureHandle handle, int unit) 
        : base(
            gl,
            Resources.GetShader("texture.vert"),
            Resources.GetShader("uniform_texture.frag"),
            handle,
            unit)
    {
    }
}