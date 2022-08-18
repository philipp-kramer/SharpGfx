using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class DirectMaterial : OpenGlMaterial
    {
        public DirectMaterial(Device device, Color4 color)
            : base(
                device,
                Resources.GetSource("direct.vert"), 
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