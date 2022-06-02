using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public sealed class UniformMaterial : OpenGlMaterial
    {
        public UniformMaterial(Device device, Color4 color)
            : base(
                device,
                Resources.GetSource("basic.vert"), 
                Resources.GetSource("single_color.frag"))
        {
            DoInContext(() =>
            {
                Set("color", color.Vector);
            });
            CheckUndefinedChannels = false;
        }
    }
}