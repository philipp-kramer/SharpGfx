using System.Drawing;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class CameraRendering : Rendering
    {
        public Camera Camera { get; }

        protected CameraRendering(Device device, Vector2 size, Color3 ambientColor, Camera camera)
            : base(device, size, ambientColor)
        {
            Camera = camera;
        }

        public override void OnRenderFrame()
        {
            Device.SetCameraView(Scene, GetView());
            Device.Render(Scene, Size, Camera.Position, AmbientColor.GetColor4(1));
        }

        protected CameraView GetView()
        {
            return new CameraView(Device, Camera.Position, Camera.LookAt, Device.World.Unit3Y);
        }
    }
}
