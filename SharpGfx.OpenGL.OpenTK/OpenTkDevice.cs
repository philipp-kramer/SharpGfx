namespace SharpGfx.OpenGL.OpenTK;

public class OpenTkDevice : OpenGlDevice
{
    public OpenTkDevice() 
        : base(new OpenTkApi())
    {
    }

    public override Space Color { get; } = new OpenTkSpace(Domain.Color);
    public override Space World { get; } = new OpenTkSpace(Domain.World);
    protected override Space View { get; } = new OpenTkSpace(Domain.View);

    public override Space Model()
    {
        return new OpenTkSpace(Domain.Model);
    }

    public override Primitives.Matrix4 GetViewMatrix(CameraView cameraView)
    {
        var eyeVector = (Vector3) cameraView.Eye.Vector;
        return new Matrix4(View, global::OpenTK.Mathematics.Matrix4.LookAt(
            eyeVector.Value,
            ((Vector3) (cameraView.Eye + cameraView.LookAt).Vector).Value,
            ((Vector3) cameraView.Up).Value));
    }

    public override Primitives.Matrix4 GetProjection(float aspect, Camera camera)
    {
        var perspective = global::OpenTK.Mathematics.Matrix4.CreatePerspectiveFieldOfView(camera.FovY, aspect, camera.Near, camera.Far);
        return new Matrix4(View, perspective);
    }

    protected override Primitives.Matrix4 GetOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
    {
        var perspective = global::OpenTK.Mathematics.Matrix4.CreatePerspectiveOffCenter(left, right, bottom, top, near, far);
        return new Matrix4(View, perspective);
    }
}