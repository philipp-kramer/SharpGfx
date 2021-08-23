using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public sealed class UniformMaterial : OtkShadedMaterial
    {
        private readonly Color4 _color;

        public UniformMaterial(Color4 color)
            : base(
                Resources.Get<UniformMaterial>("Shaders.basic.vert"), 
                Resources.Get<UniformMaterial>("Shaders.single_color.frag"))
        {
            _color = color;

            Shading.UndefinedChannels = false;
        }

        public override void Apply(Point3 cameraPosition)
        {
            Shading.Set("color", _color.Vector);
        }

        public override void UnApply()
        {
            Shading.ResetVector4("color");
        }
    }
}