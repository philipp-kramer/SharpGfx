namespace SharpGfx.OpenTK.Materials
{
    public class NopMaterial : OtkShadedMaterial
    {
        public NopMaterial()
            : base(
                Resources.Get<UniformMaterial>("Resources.Shaders.basic.vert"),
                Resources.Get<NopMaterial>("Resources.Shaders.nop.frag"))
        {
            Shading.CheckUndefinedChannels = false;
        }
    }
}
