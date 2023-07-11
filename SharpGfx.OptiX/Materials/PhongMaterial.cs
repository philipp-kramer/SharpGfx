using System;
using System.Linq;
using System.Runtime.InteropServices;
using SharpGfx.Primitives;

namespace SharpGfx.OptiX.Materials;

public class PhongMaterial : OptixMaterial
{
    [DllImport(@".\optix.dll", EntryPoint = "PhongMaterial_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe MaterialPtr CreateMaterial(
        ContextPtr context,
        string rays_cu,
        float* material_color,
        float* ambient_color,
        float* lights_position,
        float* lights_color,
        int point_lights_count,
        float* material_color_specular,
        float material_shininess);


    private static unsafe MaterialPtr CreateMaterial(OptixDevice device, string program, Reflectance<Color3> reflectance, Lighting lighting)
    {
        var lightPositions = lighting.Lights
            .SelectMany(l => (float[]) l.Position.Vector.Values)
            .ToArray();
        var lightColors= lighting.Lights
            .SelectMany(l => (float[])l.Color.Vector.Values)
            .ToArray();
        fixed (float* materialColor = new[] { reflectance.Diffuse.R, reflectance.Diffuse.G, reflectance.Diffuse.B, reflectance.Alpha })
        fixed (float* ambientColor = (float[]) lighting.Ambient.Vector.Values)
        fixed (float* positions = lightPositions)
        fixed (float* lights = lightColors)
        fixed (float* specular = (float[]) reflectance.Specular.Vector.Values)
        {
            return CreateMaterial(device.Context, program, materialColor, ambientColor, positions, lights, lighting.Lights.Length, specular, reflectance.Shininess);
        }
    }

    public PhongMaterial(OptixDevice device, Reflectance<Color3> reflectance, Lighting lighting)
        : base(CreateMaterial(device, Resources.GetProgram("ray_phong.cu"), reflectance, lighting))
    {
    }
}