using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public sealed class UniformMaterial : OtkShadedMaterial
    {
        private readonly Color4 _color;

        public UniformMaterial(Color4 color)
            : base(
                Resources.Get<UniformMaterial>("Shaders.basic.vert"), 
                Resources.Get<UniformMaterial>("Shaders.single_color.frag"),
                false)
        {
            _color = color;

            Shading.CheckUndefinedChannels = false;
        }

        protected internal override void Apply(Point3 cameraPosition)
        {
            Shading.Set("color", _color.Vector);
            base.Apply(cameraPosition);
        }

        protected internal override void UnApply()
        {
            base.UnApply();
            Shading.ResetVector4("color");
        }
    }
}