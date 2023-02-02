using System.Linq;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Materials;

public class LambertMaterial : OpenGlMaterial
{
    public LambertMaterial(GlApi gl, Color4 color, Lighting lighting)
        : this(
            gl, 
            Resources.GetShader("normal.vert"), 
            Resources.GetShader("lambert.frag"), 
            color, 
            lighting)
    {
    }

    protected LambertMaterial(GlApi gl, string vertexShader, string fragShader, Color4 color, Lighting lighting)
        : this(gl, vertexShader, fragShader, lighting)
    {
        if (color.A < 1) Transparent = true;

        DoInContext(() =>
        {
            Set("material", color.Vector);
        });
    }

    protected LambertMaterial(GlApi gl, string vertexShader, string fragShader, Lighting lighting)
        : base(gl, vertexShader, fragShader)
    {
        Update(lighting);
    }

    public void Update(Lighting lighting)
    {
        DoInContext(() =>
        {
            Set("ambient", lighting.Ambient.Vector);
            Set("lightPositions", lighting.Lights.Select(l => l.Position.Vector).ToArray());
            Set("lightColors", lighting.Lights.Select(l => l.Color.Vector).ToArray());
            Set("lightCount", lighting.Lights.Length);
        });
    }
}