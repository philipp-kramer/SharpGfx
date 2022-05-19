namespace SharpGfx.OpenTK.Materials
{
    public class CameraFacingTextureMaterial : OtkTextureMaterial
    {
        public CameraFacingTextureMaterial(Device device, TextureHandle handle, int unit) 
            : base(
                device,
                Resources.Get<FlatTextureMaterial>("Resources.Shaders.camera_facing_texture.vert"),
                Resources.Get<FlatTextureMaterial>("Resources.Shaders.texture.frag"),
                handle,
                unit)
        {
        }
    }
}
