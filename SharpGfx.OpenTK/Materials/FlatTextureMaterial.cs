namespace SharpGfx.OpenTK.Materials
{
    public class FlatTextureMaterial : OtkTextureMaterial
    {
        public FlatTextureMaterial(Device device, TextureHandle handle, int unit) 
            : base(
                device,
                Resources.Get<FlatTextureMaterial>("Shaders.flat_texture.vert"),
                Resources.Get<FlatTextureMaterial>("Shaders.texture.frag"),
                handle,
                unit)
        {
        }
    }
}
