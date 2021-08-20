namespace SharpGfx.OpenTK.Materials
{
    public sealed class VertexColorMaterial : OtkShadedMaterial
    {
        public VertexColorMaterial()
            : base(
                Resources.Get<VertexColorMaterial>("Shaders.vertex_color.vert"), 
                Resources.Get<VertexColorMaterial>("Shaders.vertex_color.frag"))
        {
        }
    }
}