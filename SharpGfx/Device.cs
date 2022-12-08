using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using SharpGfx.Host;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;


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

        public abstract RenderObject Object(Space space, string name, Material material, params VertexAttribute[] attributes);
        public abstract RenderObject Object(Space space, string name, Material material, uint[] triangles, params VertexAttribute[] attributes);
        public abstract RenderObject Object(Space space, string name, Material material, ushort[] triangles, params VertexAttribute[] attributes);
        public abstract void SetVertexArrayAttributes(uint arrayHandle, uint shaderHandle, VertexAttribute[] attributes, VertexBuffer[] buffers);

        public abstract TextureHandle Texture(Image<Rgba32> image);
        public abstract TextureHandle RgbTexture(IVector2 pixels);
        public abstract TextureHandle DepthTexture(IVector2 pixels);
        public abstract void ClearTexture(int unit);

        public abstract uint Compile(string vertexShader, string fragShader, string fragColorChannel, List<string> errors);
        public abstract void UseProgram(uint handle);
        public abstract void DeleteProgram(uint handle);

        public void CheckSpaces(ICollection<RenderObject> scene)
        {
            foreach (var obj in scene)
            {
                if (obj.Space != World) throw new InvalidOperationException("object not in world-space");
            }
        }

        public abstract void EnableBlend();
        public abstract void DisableBlend();

        public abstract Matrix4 GetViewMatrix(CameraView cameraView);
        public abstract Matrix4 GetPerspectiveProjection(float fovy, float aspect, float near, float far);
        public abstract Matrix4 GetPerspectiveOffCenterProjection(float left, float right, float bottom, float top,  float near, float far);
        public abstract void SetCameraView(ICollection<RenderObject> scene, CameraView cameraView);
        public abstract void SetProjection(ICollection<RenderObject> scene, Matrix4 projection);
        public abstract void Render(ICollection<RenderObject> scene, IVector2 pixels, Color4 ambientColor);
        public abstract void Render(ICollection<RenderObject> scene, IVector2 pixels, Point3 cameraPosition, Color4 ambientColor);
        public abstract void TakeColorPicture(ICollection<RenderObject> scene, IVector2 pixels, Color4 ambientColor, Point3 cameraPosition, CameraView cameraView, TextureHandle texture);
        public abstract void TakeDepthPicture(ICollection<RenderObject> scene, IVector2 pixels, Color4 ambientColor, Point3 cameraPosition, CameraView cameraView, Matrix4 projection, TextureHandle texture);

        public abstract uint GetUniformLocation(uint shader, string name);
        public abstract void Uniform1(uint location, int value);
        public abstract void Uniform1(uint location, int count, int[] values);
        public abstract void Uniform1(uint location, float value);
        public abstract void Uniform1(uint location, int count, float[] values);
        public abstract void Uniform2(uint location, float valueX, float valueY);
        public abstract void Uniform3(uint location, float valueX, float valueY, float valueZ);
        public abstract void Uniform3(uint location, ICollection<IVector3> values);
        public abstract void Uniform4(uint location, float valueX, float valueY, float valueZ, float valueW);
        public abstract void UniformMatrix4(uint location, Matrix4 value);
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
