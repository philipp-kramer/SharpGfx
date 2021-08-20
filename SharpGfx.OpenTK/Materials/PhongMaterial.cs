using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public class PhongMaterial : DiffuseMaterial
    {
        private readonly Color3 _specular;
        private readonly float _shininess;

        public PhongMaterial(Light light, Color3 materialDiffuse, Color3 specular, float shininess)
            : this(
                Resources.Get<PhongMaterial>("Shaders.normal_lighting.vert"),
                Resources.Get<PhongMaterial>("Shaders.phong_lighting.frag"),
                light, 
                materialDiffuse,
                specular, 
                shininess)
        {
        }

        protected PhongMaterial(string vertexShader, string fragmentShader, Light light, Color3 materialDiffuse, Color3 specular, float shininess)
            : base(
                vertexShader,
                fragmentShader,
                light,
                materialDiffuse)
        {
            _specular = specular;
            _shininess = shininess;
        }

        public override void Apply(Point3 cameraPosition)
        {
            base.Apply(cameraPosition);
            Shading.Set("light.specular", Light.Specular.Vector);
            Shading.Set("material.specular", _specular.Vector);
            Shading.Set("material.shininess", _shininess);

            Shading.Set("cameraPosition", cameraPosition.Vector);
        }

        public override void UnApply()
        {
            base.UnApply();
            Shading.ResetVector3("light.specular");
            Shading.ResetVector3("material.specular");
            Shading.ResetFloat("material.shininess");
        }
    }
}
