namespace SharpGfx.OpenGL.Shading
{
    public class NopMaterial : OpenGlMaterial
    {
        public NopMaterial(Device device)
            : base(
                device,
                Resources.GetSource("basic.vert"),
                Resources.GetSource("nop.frag"))
        {
            CheckUndefinedChannels = false;
        }
    }
}
