using SharpGfx.Primitives;

namespace SharpGfx;

public abstract class CameraRendering : Rendering
{
    public Camera Camera { get; }
    public CameraView View => new(Camera.Position, Camera.LookAt, Device.World.Unit3Y);

    protected CameraRendering(Device device, Color3 background, Camera camera)
        : base(device, background)
    {
        Camera = camera;
    }

    public override void OnRenderFrame(Window window)
    {
        Device.RenderWithCamera(window, this);
    }
}