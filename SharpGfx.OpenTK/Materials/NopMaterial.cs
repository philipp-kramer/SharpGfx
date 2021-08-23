namespace SharpGfx.OpenTK.Materials
{
    public class NopMaterial : OtkShadedMaterial
    {
        public NopMaterial()
            : base(
                Resources.Get<UniformMaterial>("Shaders.basic.vert"),
                Resources.Get<NopMaterial>("Shaders.nop.frag"))
        {
            Shading.UndefinedChannels = false;
        }
    }

}
