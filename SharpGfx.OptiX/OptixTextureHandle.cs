using System;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpGfx.OptiX;

internal class OptixTextureHandle : TextureHandle, IDisposable
{
    [DllImport(@".\optix.dll", EntryPoint = "Texture_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe TexturePtr CreateTexture(byte* texPixels, int texWidth, int texHeight);
    [DllImport(@".\optix.dll", EntryPoint = "Texture_Destroy", CallingConvention = CallingConvention.StdCall)]
    private static extern void DestroyTexture(TexturePtr handle);

    internal byte[] Buffer { get; }

    internal TexturePtr Handle { get; }

    public unsafe OptixTextureHandle(Image<Rgba32> image)
    {
        Buffer = new byte[4 * image.Height * image.Width];
        image.CopyPixelDataTo(Buffer);

        fixed (byte* buffer = Buffer)
        {
            Handle = CreateTexture(buffer, image.Width, image.Height);
        }
    }

    public override void ActivateTexture(int unit)
    {
    }

    public override void DeleteTexture()
    {
    }

    private void ReleaseUnmanagedResources()
    {
        DestroyTexture(Handle);
    }

    public void Dispose()
    {
        ReleaseUnmanagedResources();
        GC.SuppressFinalize(this);
    }

    ~OptixTextureHandle()
    {
        ReleaseUnmanagedResources();
    }
}