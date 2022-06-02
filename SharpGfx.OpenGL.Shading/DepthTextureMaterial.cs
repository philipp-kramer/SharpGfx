namespace SharpGfx.OpenGL.Shading
{
    public class DepthTextureMaterial : TextureMaterial
    {
        public DepthTextureMaterial(Device device, TextureHandle handle, int unit)
            : base(
                device,
                Resources.GetSource("texture.vert"),
                Resources.GetSource("depth_texture.frag"),
                handle,
                unit)
        {
        }
    }
}