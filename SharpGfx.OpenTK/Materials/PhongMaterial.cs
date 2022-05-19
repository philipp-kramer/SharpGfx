using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public class PhongMaterial : DiffuseMaterial
    {
        public PhongMaterial(Light light, Color3 materialDiffuse, Color3 specular, float shininess)
            : this(
                Resources.Get<PhongMaterial>("Resources.Shaders.normal_lighting.vert"),
                Resources.Get<PhongMaterial>("Resources.Shaders.phong_lighting.frag"),
                light, 
                materialDiffuse,
                specular, 
                shininess)
        {
        }

        protected PhongMaterial(
            string vertexShader, 
            string fragmentShader, 
            Light light, 
            Color3 materialDiffuse, 
            Color3 specular, 
            float shininess)
            : base(
                vertexShader,
                fragmentShader,
                light,
                materialDiffuse)
        {
            Shading.DoInContext(() =>
            {
                Shading.Set("light.specular", Light.Specular.Vector);
                Shading.Set("material.specular", specular.Vector);
                Shading.Set("material.shininess", shininess);
            });
        }

        protected internal override void Apply(Point3 cameraPosition)
        {
            Shading.Set("cameraPosition", cameraPosition.Vector);
            base.Apply(cameraPosition);
        }
    }
}
