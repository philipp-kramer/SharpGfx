using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Materials;

public sealed class EmissiveMaterial : OpenGlMaterial
{
    public EmissiveMaterial(GlApi gl, Color4 color)
        : base(
            gl,
            Resources.GetShader("flat.vert"), 
            Resources.GetShader("uniform.frag"))
    {
        if (color.A < 1) Transparent = true;

        DoInContext(() =>
        {
            Set("color", color.Vector);
        });
        CheckUndefinedChannels = false;
    }
}