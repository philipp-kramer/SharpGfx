using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public sealed class NormalMaterial : OtkShadedMaterial
    {
        public NormalMaterial()
            : base(
                Resources.Get<UniformMaterial>("Resources.Shaders.normal_lighting.vert"), 
                Resources.Get<UniformMaterial>("Resources.Shaders.normal.frag"))
        {
        }
    }
}