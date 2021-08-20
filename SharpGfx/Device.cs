using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.CompilerServices;
using SharpGfx.Host;
using SharpGfx.Primitives;


[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx
{
    public abstract class Device
    {
        public Space Color { get; }
        public Space World { get; }
        protected Space View { get; }

        protected Device(Space color, Space world)
        {
            Color = color;
            World = world;
            View = new HostSpace(Domain.View);
        }

        public abstract Space Model();

        public abstract RenderObject Object(Space space, Material material, params (string, float[], int)[] vertexData);
        public abstract RenderObject Object(Space space, Material material, uint[] indices, params (string, float[], int)[] vertexData);
        public abstract RenderObject Object(Space space, Material material, ushort[] indices, params (string, float[], int)[] vertexData);

        public abstract TextureHandle Texture(Bitmap bitmap, bool manualLevels = false);
        public abstract TextureHandle RgbTexture(Size pixels);
        public abstract TextureHandle DepthTexture(Size pixels);
        public abstract void ClearTexture(int unit);

        public abstract FrameBuffer FrameBuffer();
        public abstract FrameBuffer FrameRenderBuffer(Size pixels);

        public void CheckSpaces(ICollection<RenderObject> scene)
        {
            foreach (var obj in scene)
            {
                if (obj.Space != World) throw new InvalidOperationException("object not in world-space");
            }
        }

        // TODO: consider computing using HostMatrix and move the following to into Matrix4
        public abstract Matrix4 GetViewMatrix(CameraView cameraView);
        public abstract Matrix4 GetPerspectiveProjection(float fovy, float aspect, float near, float far);
        public abstract Matrix4 GetPerspectivOffCenterProjection(float left, float right, float bottom, float top,  float near, float far);

        public abstract void SetCameraView(ICollection<RenderObject> scene, CameraView cameraView);
        public abstract void SetProjection(ICollection<RenderObject> scene, Matrix4 projection);

        public abstract void Render(ICollection<RenderObject> scene, Size pixels, Point3 cameraPosition, Color4 ambientColor);
        public abstract void TakeColorPicture(ICollection<RenderObject> scene, Size pixels, Color4 ambientColor, Point3 cameraPosition, CameraView cameraView, TextureHandle texture);
        public abstract TextureHandle TakeDepthPicture(ICollection<RenderObject> scene, Size pixels, Color4 ambientColor, Point3 cameraPosition, CameraView cameraView);
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
    }
}
