using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class PhongMaterial : DiffuseMaterial
    {
        public PhongMaterial(
            Device device,
            Point3 lightPosition,
            Light light,
            Reflectance reflectance)
            : this(
                device,
                Resources.GetSource("normal_lighting.vert"),
                Resources.GetSource("phong_lighting.frag"),
                lightPosition,
                light, 
                reflectance)
        {
        }

        public PhongMaterial(
            Device device,
            string vertexShader, 
            string fragShader,
            Point3 lightPosition,
            Light light, 
            Reflectance reflectance)
            : base(
                device,
                vertexShader,
                fragShader,
                lightPosition,
                light,
                reflectance.Diffuse)
        {
            DoInContext(() =>
            {
                Set("light.specular", light.Specular.Vector);
                Set("material.specular", reflectance.Specular.Vector);
                Set("material.shininess", reflectance.Shininess);
            });
        }
    }
}
