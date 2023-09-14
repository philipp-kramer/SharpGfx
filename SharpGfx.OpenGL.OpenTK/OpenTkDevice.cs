using System;

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

    public override Primitives.Matrix4 GetProjection(float width, float height, Camera camera)
    {
        switch (camera.Projection)
        {
            case OrthographicProjection o:
                var om = global::OpenTK.Mathematics.Matrix4.CreateOrthographic(o.PixelScale * width, o.PixelScale * height, o.Near, o.Far);
                return new Matrix4(View, om);

            case PerspectiveProjection p:
                var pm = global::OpenTK.Mathematics.Matrix4.CreatePerspectiveFieldOfView(p.FovY, width / height, p.Near, p.Far);
                return new Matrix4(View, pm);

            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    protected override Primitives.Matrix4 GetOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
    {
        var perspective = global::OpenTK.Mathematics.Matrix4.CreatePerspectiveOffCenter(left, right, bottom, top, near, far);
        return new Matrix4(View, perspective);
    }
}