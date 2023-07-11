using System;
using System.Linq;
using System.Runtime.InteropServices;
using SharpGfx.Primitives;

namespace SharpGfx.OptiX.Materials;

public class LambertMaterial : OptixMaterial
{
    [DllImport(@".\optix.dll", EntryPoint = "LambertMaterial_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe MaterialPtr CreateMaterial(ContextPtr context, string raysCu, float* material_color, float* ambient_color, float* light_positions, float* light_colors, int point_light_count);

    private static unsafe MaterialPtr CreateMaterial(OptixDevice device, string program, Color4 color, Lighting lighting)
    {
        var lightPositions = lighting.Lights
            .SelectMany(l => (float[]) l.Position.Vector.Values)
            .ToArray();
        var lightColors= lighting.Lights
            .SelectMany(l => (float[])l.Color.Vector.Values)
            .ToArray();
        fixed (float* materialColor = (float[]) color.Vector.Values)
        fixed (float* ambientColor = (float[]) lighting.Ambient.Vector.Values)
        fixed (float* positions = lightPositions)
        fixed (float* colors = lightColors)
        {
            return CreateMaterial(device.Context, program, materialColor, ambientColor, positions, colors, lighting.Lights.Length);
        }
    }

    public LambertMaterial(OptixDevice device, Color4 color, Lighting lighting)
        : base(CreateMaterial(device, Resources.GetProgram("ray_lambert.cu"), color, lighting))
    {
    }
}