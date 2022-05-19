namespace SharpGfx.OpenTK.Materials
{
    public sealed class VertexColorMaterial : OtkShadedMaterial
    {
        public VertexColorMaterial()
            : base(
                Resources.Get<VertexColorMaterial>("Resources.Shaders.vertex_color.vert"), 
                Resources.Get<VertexColorMaterial>("Resources.Shaders.vertex_color.frag"))
        {
        }
    }
}