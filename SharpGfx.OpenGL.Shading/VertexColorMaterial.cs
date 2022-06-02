namespace SharpGfx.OpenGL.Shading
{
    public sealed class VertexColorMaterial : OpenGlMaterial
    {
        public VertexColorMaterial(Device device)
            : base(
                device,
                Resources.GetSource("vertex_color.vert"), 
                Resources.GetSource("vertex_color.frag"))
        {
        }
    }
}