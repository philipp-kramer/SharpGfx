using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class PhongMaterial : DiffuseMaterial
    {
        public PhongMaterial(Device device, Light light, Color3 materialDiffuse, Color3 specular, float shininess)
            : this(
                device,
                Resources.GetSource("normal_lighting.vert"),
                Resources.GetSource("phong_lighting.frag"),
                light, 
                materialDiffuse,
                specular, 
                shininess)
        {
        }

        protected PhongMaterial(
            Device device,
            string vertexShader, 
            string fragShader, 
            Light light, 
            Color3 materialDiffuse, 
            Color3 specular, 
            float shininess)
            : base(
                device,
                vertexShader,
                fragShader,
                light,
                materialDiffuse)
        {
            DoInContext(() =>
            {
                Set("light.specular", Light.Specular.Vector);
                Set("material.specular", specular.Vector);
                Set("material.shininess", shininess);
            });
        }
    }
}
