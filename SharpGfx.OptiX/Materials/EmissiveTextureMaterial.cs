using System;
using System.Runtime.InteropServices;

namespace SharpGfx.OptiX.Materials;

public class EmissiveTextureMaterial : TextureMaterial
{
    [DllImport(@".\optix.dll", EntryPoint = "EmissiveTextureMaterial_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern MaterialPtr CreateMaterial(ContextPtr context, string raysCu, TexturePtr texture);

    private static MaterialPtr CreateMaterial(OptixDevice device, string program, OptixTextureHandle texture)
    {
        return CreateMaterial(device.Context, program, texture.Handle);
    }

    public EmissiveTextureMaterial(OptixDevice device, TextureHandle texture)
        : base(CreateMaterial(device, Resources.GetProgram("ray_uniform_texture.cu"), (OptixTextureHandle) texture), texture)
    {
    }
}