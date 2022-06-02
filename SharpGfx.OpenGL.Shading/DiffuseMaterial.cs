using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class DiffuseMaterial : OpenGlMaterial
    {
        protected readonly Light Light;
        protected readonly Color3 MaterialDiffuse;

        public DiffuseMaterial(Device device, Light light, Color3 materialDiffuse)
            : this(
                device,
                Resources.GetSource("normal_lighting.vert"),
                Resources.GetSource("diffuse_lighting.frag"), 
                light, 
                materialDiffuse)
        {
        }

        protected DiffuseMaterial(Device device, string vertexShader, string fragShader, Light light, Color3 materialDiffuse)
            : base(device, vertexShader, fragShader)
        {
            Light = light;
            MaterialDiffuse = materialDiffuse;

            DoInContext(() =>
            {
                Set("light.position", Light.Position.Vector);
                Set("light.ambient", Light.Ambient.Vector);
                Set("light.diffuse", Light.Diffuse.Vector);
                Set("material.diffuse", MaterialDiffuse.Vector);
            });
        }
    }
}