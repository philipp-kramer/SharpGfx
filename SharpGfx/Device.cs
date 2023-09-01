using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx;

public abstract class Device
{
    public abstract Space Color { get; }
    public abstract Space World { get; }
    protected abstract Space View { get; }
    public abstract Space Model();

    public void CheckSpaces(ICollection<Instance> scene)
    {
        foreach (var instance in scene)
        {
            if (instance.Space != World) throw new InvalidOperationException("object not in world-space");
        }
    }

    public abstract Material Emissive(Color4 color);
    public abstract Material Emissive(TextureHandle color);
    public abstract Material Lambert(Color4 color, Lighting lighting);
    public abstract Material Lambert(TextureHandle color, Lighting lighting);
    public abstract Material Phong(Reflectance<Color3> reflectance, Lighting lighting);
    public abstract Material Phong(Reflectance<TextureHandle> reflectance, Lighting lighting);

    public abstract Surface Surface(Material material, params SurfaceAttribute[] attributes);
    public abstract Surface Surface(Material material, uint[] triangles, params SurfaceAttribute[] attributes);
    public abstract Surface Surface(Material material, ushort[] triangles, params SurfaceAttribute[] attributes);

    public Instance Group(Space space, string name) => new Instance(space, name);
    public abstract Instance Instance(Space space, string name, Surface surface);

    public abstract TextureHandle Texture(Image<Rgba32> image);

    public abstract void Render(Window window, Rendering rendering);
    public abstract void RenderWithCamera(Window window, CameraRendering rendering);
}

public static class DeviceExtensions
{
    public static Color3 Color3(this Device device, float r, float g, float b)
    {
        return new Color3(device.Color.Vector3(r, g, b));
    }

    public static Color4 Color4(this Device space, float r, float g, float b, float a)
    {
        return new Color4(space.Color.Vector4(r, g, b, a));
    }

    public static Color4 Color4(this Device space, Color3 color, float a)
    {
        return new Color4(space.Color.Vector4(color.R, color.G, color.B, a));
    }
}