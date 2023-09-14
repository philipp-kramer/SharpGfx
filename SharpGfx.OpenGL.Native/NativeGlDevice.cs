using System;
using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Native;

public class NativeGlDevice : OpenGlDevice
{
    public NativeGlDevice() 
        : base(new NativeGlApi())
    {
    }

    public override Space Color { get; } = new HostSpace(Domain.Color);
    public override Space World { get; } = new HostSpace(Domain.World);
    protected override Space View { get; } = new HostSpace(Domain.View);

    public override Space Model() { return new HostSpace(Domain.Model); }

    public override Matrix4 GetViewMatrix(CameraView cameraView)
    {
        return HostMatrix4.GetView(
            View,
            (HostVector3)cameraView.Eye.Vector,
            (HostVector3)cameraView.LookAt,
            (HostVector3)cameraView.Up);
    }

    public override Matrix4 GetProjection(float width, float height, Camera camera)
    {
        switch (camera.Projection)
        {
            case OrthographicProjection o:
                return HostMatrix4.GetOrthographicProjection(View, o.PixelScale * width, o.PixelScale * height, o.Near, o.Far);

            case PerspectiveProjection p:
                return HostMatrix4.GetPerspectiveProjection(View, p.FovY, width / height, p.Near, p.Far);

            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    protected override Matrix4 GetOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
    {
        return HostMatrix4.GetOffCenterPerspectiveProjection(View, left, right, bottom, top, near, far);
    }
}