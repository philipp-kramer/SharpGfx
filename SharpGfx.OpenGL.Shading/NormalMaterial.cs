namespace SharpGfx.OpenGL.Shading
{
    public sealed class NormalMaterial : OpenGlMaterial
    {
        public NormalMaterial(Device device)
            : base(
                device,
                Resources.GetSource("normal_lighting.vert"), 
                Resources.GetSource("normal.frag"))
        {
        }
    }
}