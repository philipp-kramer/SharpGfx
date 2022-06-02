namespace SharpGfx.OpenGL.Shading
{
    public class FlatTextureMaterial : TextureMaterial
    {
        public FlatTextureMaterial(Device device, TextureHandle handle, int unit) 
            : base(
                device,
                Resources.GetSource("texture.vert"),
                Resources.GetSource("texture.frag"),
                handle,
                unit)
        {
        }
    }
}
