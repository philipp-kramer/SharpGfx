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

    public override Matrix4 GetProjection(float aspect, Camera camera)
    {
        return HostMatrix4.GetProjection(View, camera.FovY, aspect, camera.Near, camera.Far);
    }

    protected override Matrix4 GetOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
    {
        return HostMatrix4.GetProjection(View, left, right, bottom, top, near, far);
    }
}