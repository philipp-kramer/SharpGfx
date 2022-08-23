using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class DiffuseMaterial : OpenGlMaterial
    {
        public DiffuseMaterial(Device device, Point3 lightPosition, Light light, Color3 materialDiffuse)
            : this(
                device,
                Resources.GetSource("normal_lighting.vert"),
                Resources.GetSource("diffuse_lighting.frag"),
                lightPosition,
                light, 
                materialDiffuse)
        {
        }

        protected DiffuseMaterial(Device device, string vertexShader, string fragShader, Point3 lightPosition, Light light, Color3 materialDiffuse)
            : base(device, vertexShader, fragShader)
        {
            DoInContext(() =>
            {
                Set("light.position", lightPosition.Vector);
                Set("light.ambient", light.Ambient.Vector);
                Set("light.diffuse", light.Diffuse.Vector);
                Set("material.diffuse", materialDiffuse.Vector);
            });
        }
    }
}