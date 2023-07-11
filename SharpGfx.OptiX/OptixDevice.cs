using System;
using System.Runtime.InteropServices;
using SharpGfx.Host;
using SharpGfx.OptiX.Materials;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// TODO: refactor logic out to reduce this to a method API
namespace SharpGfx.OptiX;

public class OptixDevice : Device
{
    [DllImport(@".\optix.dll", EntryPoint = "create_context", CallingConvention = CallingConvention.StdCall)]
    private static extern ContextPtr CreateContext();
    [DllImport(@".\optix.dll", EntryPoint = "set_cam_eye", CallingConvention = CallingConvention.StdCall)]
    private static extern void SetCamEye(CameraPtr camera, Float3 val);
    [DllImport(@".\optix.dll", EntryPoint = "set_cam_look_at", CallingConvention = CallingConvention.StdCall)]
    private static extern void SetCamLookAt(CameraPtr camera, Float3 val);
    [DllImport(@".\optix.dll", EntryPoint = "set_cam_up", CallingConvention = CallingConvention.StdCall)]
    private static extern void SetCamUp(CameraPtr camera, Float3 val);

    [DllImport(@".\optix.dll", EntryPoint = "render", CallingConvention = CallingConvention.StdCall)]
    private static extern int Render(WindowPtr window, CameraPtr camera, ContextPtr context);

    public override Space Color { get; } = new HostSpace(Domain.Color);
    public override Space World { get; } = new HostSpace(Domain.World);
    protected override Space View { get; } = new HostSpace(Domain.View);

    internal ContextPtr Context { get; }

    public OptixDevice()
    {
        Context = CreateContext();
    }

    public override Space Model() { return new HostSpace(Domain.Model); }


    public override Material Emissive(Color4 color)
    {
        if (color.A < 1) throw new NotImplementedException("alpha blending");
        return new EmissiveMaterial(this, color);
    }

    public override Material Emissive(TextureHandle color)
    {
        return new EmissiveTextureMaterial(this, color);
    }

    public override Material Lambert(Color4 color, Lighting lighting)
    {
        if (color.A < 1) throw new NotImplementedException("alpha blending");
        return new LambertMaterial(this, color, lighting);
    }

    public override Material Lambert(TextureHandle color, Lighting lighting)
    {
        return new LambertTextureMaterial(this, color, lighting);
    }

    public override Material Phong(Reflectance<Color3> reflectance, Lighting lighting)
    {
        if (reflectance.Alpha < 1) throw new NotImplementedException("alpha blending");
        return new PhongMaterial(this, reflectance, lighting);
    }

    public override Material Phong(Reflectance<TextureHandle> reflectance, Lighting lighting)
    {
        throw new NotImplementedException("phong texture");
    }

    public override Body Body(Material material, params IVertexAttribute[] attributes)
    {
        return new OptixBody(Context, material, attributes);
    }

    public override Body Body(Material material, ushort[] indices, params IVertexAttribute[] attributes)
    {
        return new OptixBody(Context, material, indices, attributes);
    }

    public override Body Body(Material material, uint[] indices, params IVertexAttribute[] attributes)
    {
        throw new NotImplementedException("uint indices not handled in ray programs, try using C++ template");
        return new OptixBody(Context, material, indices, attributes);
    }

    public override Instance Instance(Space space, string name, Body body)
    {
        return new OptixInstance(space, name, (OptixBody) body);
    }

    public override TextureHandle Texture(Image<Rgba32> image)
    {
        return new OptixTextureHandle(image);
    }

    public readonly struct Float3
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Z;

        public Float3(IVector3 vector)
        {
            X = vector.X;
            Y = vector.Y;
            Z = vector.Z;
        }
    }

    public override void Render(Window window, Rendering rendering)
    {
        var optiXWindow = (OptixWindow) window;
        int errorCode = Render(optiXWindow.Window, optiXWindow.Camera, Context);
        if (errorCode != 0) throw new InvalidOperationException();
    }

    public override void RenderWithCamera(Window window, CameraRendering rendering)
    {
        var camera = rendering.Camera;

        var optixCamera = ((OptixWindow) window).Camera;
        SetCamEye(optixCamera, new Float3(camera.Position.Vector));
        SetCamLookAt(optixCamera, new Float3(camera.LookAt));
        SetCamUp(optixCamera, new Float3(rendering.View.Up));

        Render(window, rendering);
    }
}