using System;
using System.Runtime.InteropServices;

namespace SharpGfx.OptiX.Materials;

public class OptixMaterial : Material, IDisposable
{
    [DllImport(@".\optix.dll", EntryPoint = "Material_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern MaterialPtr CreateMaterial(ContextPtr context, string raysCu);
    [DllImport(@".\optix.dll", EntryPoint = "Material_Destroy", CallingConvention = CallingConvention.StdCall)]
    private static extern void DestroyMaterial(MaterialPtr handle);

    public MaterialPtr Handle { get; }

    public OptixMaterial(Device device, string raysCu)
    {
        Handle = CreateMaterial(((OptixDevice)device).Context, raysCu);
    }

    protected OptixMaterial(MaterialPtr handle)
    {
        Handle = handle;
    }

    protected override void DoInContext(Action action)
    {
    }

    public override void Apply()
    {
    }

    public override void UnApply()
    {
    }

    private void ReleaseUnmanagedResources()
    {
        DestroyMaterial(Handle);
    }

    protected virtual void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
    }

    public override void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    ~OptixMaterial()
    {
        Dispose(false);
    }
}