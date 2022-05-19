using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public class DiffuseMaterial : OtkShadedMaterial
    {
        protected readonly Light Light;
        protected readonly Color3 MaterialDiffuse;

        public DiffuseMaterial(Light light, Color3 materialDiffuse)
            : this(
                Resources.Get<DiffuseMaterial>("Resources.Shaders.normal_lighting.vert"),
                Resources.Get<DiffuseMaterial>("Resources.Shaders.diffuse_lighting.frag"), 
                light, 
                materialDiffuse)
        {
        }

        protected DiffuseMaterial(string vertexShader, string fragmentShader, Light light, Color3 materialDiffuse)
            : base(vertexShader, fragmentShader)
        {
            Light = light;
            MaterialDiffuse = materialDiffuse;

            Shading.DoInContext(() =>
            {
                Shading.Set("light.position", Light.Position.Vector);
                Shading.Set("light.ambient", Light.Ambient.Vector);
                Shading.Set("light.diffuse", Light.Diffuse.Vector);
                Shading.Set("material.diffuse", MaterialDiffuse.Vector);
            });
        }
    }
}