using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    /// <summary>
    /// left-handed
    /// </summary>
    public sealed class OtkDevice : Device
    {
        public OtkDevice()
            : base(new OtkSpace(Domain.Color), new OtkSpace(Domain.World))
        {
        }

        public override Space Model()
        {
            return new OtkSpace(Domain.Model);
        }

        public override RenderObject Object(Space space, Material material, params (string, float[], int)[] vertexData)
        {
            return new OtkRenderObject(space, material, GetVertexAttributes(vertexData));
        }

        public override RenderObject Object(Space space, Material material, uint[] indices, params (string, float[], int)[] vertexData) 
        {
            return new OtkIndexedRenderObject<uint>(space, material, indices, GetVertexAttributes(vertexData));
        }

        public override RenderObject Object(Space space, Material material, ushort[] indices, params (string, float[], int)[] vertexData)
        {
            return new OtkIndexedRenderObject<ushort>(space, material, indices, GetVertexAttributes(vertexData));
        }

        private static VertexAttribute[] GetVertexAttributes((string, float[], int)[] vertexData)
        {
            var attributes = new VertexAttribute[vertexData.Length];
            for (var i = 0; i < vertexData.Length; i++)
            {
                var attributeData = vertexData[i];
                attributes[i] = new VertexAttribute(
                    attributeData.Item1,
                    new OtkVertexBuffer<float>(attributeData.Item2),
                    attributeData.Item3);
            }
            return attributes;
        }

        public override TextureHandle Texture(Bitmap bitmap, bool manualLevels = false)
        {
            return manualLevels
                ? new OtkTextureHandle(OtkTextures.CreateMipmapTexture(bitmap))
                : new OtkTextureHandle(OtkTextures.CreateAutoMipmapTexture(bitmap));
        }

        public override TextureHandle RgbTexture(Size pixels)
        {
            return new OtkTextureHandle(OtkTextures.CreateTexture(pixels, PixelInternalFormat.Rgb, PixelFormat.Rgb, PixelType.UnsignedByte));
        }

        public override TextureHandle DepthTexture(Size pixels)
        {
            return new OtkTextureHandle(OtkTextures.CreateTexture(pixels, PixelInternalFormat.DepthComponent, PixelFormat.DepthComponent, PixelType.Float));
        }

        public override void ClearTexture(int unit)
        {
            OtkTextures.ClearTexture(unit);
        }

        public override FrameBuffer FrameBuffer()
        {
            return new OtkFrameBuffer();
        }

        public override FrameBuffer FrameRenderBuffer(Size pixels)
        {
            return new OtkFrameRenderBuffer(pixels);
        }

        public override Matrix4 GetViewMatrix(CameraView cameraView)
        {
            var eyeVector = (OtkVector3)cameraView.Eye.Vector;
            return new OtkMatrix4(View, global::OpenTK.Matrix4.LookAt(
                eyeVector.Value,
                ((OtkVector3)(cameraView.Eye + cameraView.LookAt).Vector).Value,
                ((OtkVector3)cameraView.Up).Value));
        }

        public override Matrix4 GetPerspectiveProjection(float fovy, float aspect, float near, float far)
        {
            return new OtkMatrix4(View, global::OpenTK.Matrix4.CreatePerspectiveFieldOfView(fovy, aspect, near, far));
        }

        public override Matrix4 GetPerspectivOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
        {
            return new OtkMatrix4(View, global::OpenTK.Matrix4.CreatePerspectiveOffCenter(left, right, bottom, top, near, far));
        }

        public override void SetCameraView(ICollection<RenderObject> scene, CameraView cameraView)
        {
            OtkRenderer.SetCameraView(this, scene.GroupBy(obj => (OtkShadedMaterial)obj.Material), cameraView);
        }

        public override void SetProjection(ICollection<RenderObject> scene, Matrix4 projection)
        {
            if (!projection.In(View)) throw new ArgumentException("needs to be in view-space", nameof(projection));
            OtkRenderer.SetProjection(scene.GroupBy(obj => (OtkShadedMaterial)obj.Material), projection);
        }

        public override void Render(
            ICollection<RenderObject> scene,
            Size pixels,
            Point3 cameraPosition,
            Color4 ambientColor)
        {
            OtkRenderer.Render(scene.GroupBy(obj => (OtkShadedMaterial)obj.Material), pixels, cameraPosition, ambientColor);
        }

        public void UndefinedChannels(ICollection<RenderObject> scene, bool check)
        {
            foreach (var obj in scene)
            {
                ((OtkShadedMaterial) obj.Material).Shading.UndefinedChannels = check;
            }
        }

        public override void TakeColorPicture(
            ICollection<RenderObject> scene,
            Size pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            TextureHandle texture)
        {
            if (!cameraPosition.Vector.In(World)) throw new ArgumentException("needs to be in world-space", nameof(cameraPosition));

            OtkRenderer.TakeColorPicture(this, scene, pixels, ambientColor, cameraPosition, cameraView, texture);
        }

        public override TextureHandle TakeDepthPicture(
            ICollection<RenderObject> scene,
            Size pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            Matrix4 projection)
        {
            if (!cameraPosition.Vector.In(World)) throw new ArgumentException("needs to be in world-space", nameof(cameraPosition));

            return OtkRenderer.TakeDepthPicture(this, scene, pixels, ambientColor, cameraPosition, cameraView, projection);
        }
    }
}
