using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class CameraRendering : Rendering
    {
        public Camera Camera { get; }

        protected CameraRendering(Device device, IVector2 size, Color3 ambientColor, Camera camera)
            : base(device, size, ambientColor)
        {
            Camera = camera;
        }

        public override void OnResize(IVector2 size)
        {
            base.OnResize(size);
            Device.SetProjection(Scene, Device.GetPerspectiveProjection(FovY, Aspect, Near, Far));
        }

        public override void OnRenderFrame()
        {
            Device.SetCameraView(Scene, GetView());
            Device.Render(Scene, Size, Camera.Position, AmbientColor.GetColor4(1));
        }

        protected CameraView GetView()
        {
            return new CameraView(Camera.Position, Camera.LookAt, Device.World.Unit3Y);
        }
    }
}
