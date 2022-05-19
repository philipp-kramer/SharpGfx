using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public sealed class UniformMaterial : OtkShadedMaterial
    {
        public UniformMaterial(Color4 color)
            : base(
                Resources.Get<UniformMaterial>("Resources.Shaders.basic.vert"), 
                Resources.Get<UniformMaterial>("Resources.Shaders.single_color.frag"))
        {
            Shading.DoInContext(() =>
            {
                Shading.Set("color", color.Vector);
            });
            Shading.CheckUndefinedChannels = false;
        }
    }
}