using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using SharpGfx.OpenGL.Shading;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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

        public override RenderObject Object(Space space, string name, Material material, params VertexAttribute[] attributes)
        {
            return new OtkRenderObject(space, name, (OpenGlMaterial) material, attributes);
        }

        public override RenderObject Object(Space space, string name, Material material, uint[] triangles, params VertexAttribute[] attributes) 
        {
            return new OtkIndexedRenderObject<uint>(space, name, (OpenGlMaterial) material, triangles, attributes);
        }

        public override RenderObject Object(Space space, string name, Material material, ushort[] triangles, params VertexAttribute[] attributes)
        {
            return new OtkIndexedRenderObject<ushort>(space, name, (OpenGlMaterial) material, triangles, attributes);
        }

        public override void SetVertexArrayAttributes(uint arrayHandle, uint shaderHandle, VertexAttribute[] attributes, VertexBuffer[] buffers)
        {
            GL.BindVertexArray(arrayHandle);

            for (int i = 0; i < attributes.Length; i++)
            {
                var attribute = attributes[i];

                GL.BindBuffer(BufferTarget.ArrayBuffer, ((OtkVertexBuffer<float>) buffers[i]).Handle);

                int location = GL.GetAttribLocation(shaderHandle, attribute.Channel);
                GL.EnableVertexAttribArray(location);
                GL.VertexAttribPointer(location, attribute.Size, VertexAttribPointerType.Float, false, attribute.Stride * sizeof(float), attribute.Offset * sizeof(float));

                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            }

            GL.BindVertexArray(0);
        }

        public override TextureHandle Texture(Image<Rgba32> image)
        {
            return new OtkTextureHandle(OtkTextures.CreateAutoMipmapTexture(image));
        }

        public override TextureHandle RgbTexture(IVector2 pixels)
        {
            return new OtkTextureHandle(OtkTextures.CreateTexture(pixels, PixelInternalFormat.Rgb, PixelFormat.Rgb, PixelType.UnsignedByte));
        }

        public override TextureHandle DepthTexture(IVector2 pixels)
        {
            return new OtkTextureHandle(OtkTextures.CreateTexture(pixels, PixelInternalFormat.DepthComponent, PixelFormat.DepthComponent, PixelType.Float));
        }

        public override void ClearTexture(int unit)
        {
            var textureUnit = OtkTextures.GetTextureUnit(unit);
            GL.ActiveTexture(textureUnit);
            GL.BindTexture(TextureTarget.Texture2D, 0);
        }

        public override uint Compile(string vertexShader, string fragShader, string fragColorChannel, List<string> errors)
        {
            return OtkCompilation.Compile(vertexShader, fragShader, fragColorChannel, errors);
        }

        public override void UseProgram(uint handle)
        {
            GL.UseProgram(handle);
        }

        public override void DeleteProgram(uint handle)
        {
            GL.DeleteProgram(handle);
        }

        public override void EnableBlend()
        {
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        }

        public override void DisableBlend()
        {
            GL.Disable(EnableCap.Blend);
        }

        public override Matrix4 GetViewMatrix(CameraView cameraView)
        {
            var eyeVector = (OtkVector3)cameraView.Eye.Vector;
            return new OtkMatrix4(View, global::OpenTK.Mathematics.Matrix4.LookAt(
                eyeVector.Value,
                ((OtkVector3)(cameraView.Eye + cameraView.LookAt).Vector).Value,
                ((OtkVector3)cameraView.Up).Value));
        }

        public override Matrix4 GetPerspectiveProjection(float fovy, float aspect, float near, float far)
        {
            return new OtkMatrix4(View, global::OpenTK.Mathematics.Matrix4.CreatePerspectiveFieldOfView(fovy, aspect, near, far));
        }

        public override Matrix4 GetPerspectiveOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
        {
            return new OtkMatrix4(View, global::OpenTK.Mathematics.Matrix4.CreatePerspectiveOffCenter(left, right, bottom, top, near, far));
        }

        public override void SetCameraView(ICollection<RenderObject> scene, CameraView cameraView)
        {
            var materials = GetMaterials(scene);
            var view = GetViewMatrix(cameraView);
            OpenGlMaterial.Set(materials, "cameraView", view);
        }

        public override void SetProjection(ICollection<RenderObject> scene, Matrix4 projection)
        {
            if (!projection.In(View)) throw new ArgumentException("needs to be in view-space", nameof(projection));

            var materials = GetMaterials(scene);
            OpenGlMaterial.Set(materials, "projection", projection);
        }

        public override void Render(ICollection<RenderObject> scene, IVector2 pixels, Color4 ambientColor)
        {
            var materialScene = scene.GroupBy(obj => (OpenGlMaterial) obj.Material);
            OtkRenderer.Render(materialScene, pixels, ambientColor);
            UnmanagedRelease.ExecutePending();
        }

        public override void Render(ICollection<RenderObject> scene, IVector2 pixels, Point3 cameraPosition, Color4 ambientColor)
        {
            var materials = GetMaterials(scene);
            var materialScene = scene.GroupBy(obj => (OpenGlMaterial) obj.Material);
            OpenGlMaterial.SetIfDefined(World, materials, "cameraPosition", cameraPosition.Vector);
            OtkRenderer.Render(materialScene, pixels, ambientColor);
            UnmanagedRelease.ExecutePending();
        }

        public override void TakeColorPicture(
            ICollection<RenderObject> scene,
            IVector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            TextureHandle texture)
        {
            if (!cameraPosition.Vector.In(World)) throw new ArgumentException("needs to be in world-space", nameof(cameraPosition));

            var view = GetViewMatrix(cameraView);
            OtkRenderer.TakeColorPicture(this, scene, pixels, ambientColor, cameraPosition, view, texture);
        }

        public override TextureHandle TakeDepthPicture(
            ICollection<RenderObject> scene,
            IVector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            Matrix4 projection)
        {
            if (!cameraPosition.Vector.In(World)) throw new ArgumentException("needs to be in world-space", nameof(cameraPosition));

            var view = GetViewMatrix(cameraView);
            return OtkRenderer.TakeDepthPicture(this, scene, pixels, ambientColor, cameraPosition, view, projection);
        }

        public override uint GetUniformLocation(uint shader, string name)
        {
            return (uint) GL.GetUniformLocation(shader, name);
        }

        public override void Uniform1(uint location, int value)
        {
            GL.Uniform1((int) location, value);
        }

        public override void Uniform1(uint location, int count, int[] values)
        {
            GL.Uniform1((int) location, values.Length, values);
        }

        public override void Uniform1(uint location, float value)
        {
            GL.Uniform1((int)location, value);
        }

        public override void Uniform1(uint location, int count, float[] values)
        {
            GL.Uniform1((int)location, values.Length, values);
        }

        public override void Uniform2(uint location, float valueX, float valueY)
        {
            GL.Uniform2((int) location, valueX, valueY);
        }

        public override void Uniform3(uint location, float valueX, float valueY, float valueZ)
        {
            GL.Uniform3((int) location, valueX, valueY, valueZ);
        }

        public override void Uniform3(uint location, ICollection<IVector3> values)
        {
            var floats = new float[values.Count * 3];
            int j = 0;
            foreach (var vector in values)
            {
                floats[j++] = vector.X;
                floats[j++] = vector.Y;
                floats[j++] = vector.Z;
            }
            GL.Uniform3((int) location, floats.Length, floats);
        }

        public override void Uniform4(uint location, float valueX, float valueY, float valueZ, float valueW)
        {
            GL.Uniform4((int) location, valueX, valueY, valueZ, valueW);
        }

        public override void UniformMatrix4(uint location, Matrix4 value)
        {
            var v = ((OtkMatrix4)value).Value;
            GL.UniformMatrix4((int) location, true, ref v);
        }

        internal static IEnumerable<OpenGlMaterial> GetMaterials(ICollection<RenderObject> scene) =>
            scene
                .Select(obj => (OpenGlMaterial) obj.Material)
                .Distinct();
    }
}
