namespace SharpGfx.OpenGL.Shading
{
    public class CameraFacingTextureMaterial : TextureMaterial
    {
        public CameraFacingTextureMaterial(Device device, TextureHandle handle, int unit) 
            : base(
                device,
                Resources.GetSource("camera_facing_texture.vert"),
                Resources.GetSource("texture.frag"),
                handle,
                unit)
        {
        }
    }
}
