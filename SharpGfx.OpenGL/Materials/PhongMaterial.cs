using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Materials;

public class PhongMaterial : LambertMaterial
{
    public PhongMaterial(
        OpenGlDevice device,
        Reflectance<Color3> reflectance,
        Lighting lighting)
        : this(
            device,
            Resources.GetShader("normal.vert"),
            Resources.GetShader("phong.frag"),
            reflectance, 
            lighting)
    {
    }

    public PhongMaterial(
        OpenGlDevice device,
        string vertexShader,
        string fragShader,
        Reflectance<Color3> reflectance,
        Lighting lighting)
        : base(
            device.GL,
            vertexShader,
            fragShader,
            device.Color4(reflectance.Diffuse, reflectance.Alpha),
            lighting)
    {
        DoInContext(() =>
        {
            Set("materialSpecular", reflectance.Specular.Vector);
            Set("materialShininess", reflectance.Shininess);
        });
    }

    public PhongMaterial(
        OpenGlDevice device,
        string vertexShader,
        string fragShader,
        Reflectance<TextureHandle> reflectance,
        Lighting lighting)
        : base(
            device.GL,
            vertexShader,
            fragShader,
            lighting)
    {
        DoInContext(() =>
        {
            Set("materialSpecular", reflectance.Specular.Vector);
            Set("materialShininess", reflectance.Shininess);
        });
    }

}