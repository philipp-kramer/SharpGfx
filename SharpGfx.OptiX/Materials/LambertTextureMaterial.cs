using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace SharpGfx.OptiX.Materials;

public class LambertTextureMaterial : TextureMaterial
{
    [DllImport(@".\optix.dll", EntryPoint = "LambertTextureMaterial_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe MaterialPtr CreateMaterial(ContextPtr context, string raysCu, TexturePtr texture, float* ambient_color, float* light_positions, float* light_colors, int point_light_count);

    private static unsafe MaterialPtr CreateMaterial(OptixDevice device, string program, OptixTextureHandle texture, Lighting lighting)
    {
        var lightPositions = lighting.Lights
            .SelectMany(l => (float[]) l.Position.Vector.Values)
            .ToArray();
        var lightColors= lighting.Lights
            .SelectMany(l => (float[])l.Color.Vector.Values)
            .ToArray();
        fixed (float* ambientColor = (float[]) lighting.Ambient.Vector.Values)
        fixed (float* positions = lightPositions)
        fixed (float* colors = lightColors)
        {
            return CreateMaterial(device.Context, program, texture.Handle, ambientColor, positions, colors, lighting.Lights.Length);
        }
    }

    public LambertTextureMaterial(OptixDevice device, TextureHandle texture, Lighting lighting)
        : base(CreateMaterial(device, Resources.GetProgram("ray_lambert_texture.cu"), (OptixTextureHandle) texture, lighting), texture)
    {
    }
}