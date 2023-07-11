using System.Runtime.InteropServices;
using SharpGfx.Primitives;

namespace SharpGfx.OptiX.Materials;

public class EmissiveMaterial : OptixMaterial
{
    [DllImport(@".\optix.dll", EntryPoint = "EmissiveMaterial_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe MaterialPtr CreateMaterial(ContextPtr context, string raysCu, float* color);

    private static unsafe MaterialPtr CreateMaterial(OptixDevice device, string program, Color4 color4)
    {
        fixed (float* color = (float[])color4.Vector.Values)
        {
            return CreateMaterial(device.Context, program, color);
        }
    }

    public EmissiveMaterial(OptixDevice device, Color4 color4)
        : base(CreateMaterial(device, Resources.GetProgram("ray_uniform.cu"), color4))
    {
    }
}