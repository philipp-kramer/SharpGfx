using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.Host;
using SharpGfx.OpenGL.Shading;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpGfx.OpenGL
{
    public class OglDevice : Device
    {
        public OglDevice() 
            : base(new HostSpace(Domain.Color), new HostSpace(Domain.World))
        {
        }

        public override Space Model()
        {
            return new HostSpace(Domain.Model);
        }

        public override RenderObject Object(Space space, string name, Material material, params VertexAttribute[] attributes)
        {
            return new OglRenderObject(space, name, (OpenGlMaterial) material, attributes);
        }

        public override RenderObject Object(Space space, string name, Material material, uint[] triangles, params VertexAttribute[] attributes)
        {
            return new OglIndexedRenderObject<uint>(space, name, (OpenGlMaterial) material, triangles, attributes);
        }

        public override RenderObject Object(Space space, string name, Material material, ushort[] triangles, params VertexAttribute[] attributes)
        {
            return new OglIndexedRenderObject<ushort>(space, name, (OpenGlMaterial)material, triangles, attributes);
        }

        public override void SetVertexArrayAttributes(uint arrayHandle, uint shaderHandle, VertexAttribute[] attributes, VertexBuffer[] buffers)
        {
            GL.BindVertexArray(arrayHandle);

            for (int i = 0; i < attributes.Length; i++)
            {
                var attribute = attributes[i];
                GL.BindBuffer(GlBufferTarget.ArrayBuffer, ((OglVertexBuffer<float>)buffers[i]).Handle);

                uint location = GL.GetAttributeLocation(shaderHandle, attribute.Channel);
                GL.EnableVertexAttribArray(location);
                GL.FloatVertexAttribPointer(location, attribute.Size, attribute.Stride * sizeof(float), attribute.Offset * sizeof(float));

                GL.BindBuffer(GlBufferTarget.ArrayBuffer, 0);
            }

            GL.BindVertexArray(0);
        }

        public override TextureHandle Texture(Image<Rgba32> image)
        {
            return new OglTextureHandle(OglTextures.CreateMipmapTexture(image));
        }

        public override TextureHandle RgbTexture(IVector2 pixels)
        {
            return new OglTextureHandle(OglTextures.CreateTexture(pixels, GlPixelFormat.Rgb, GlPixelType.UnsignedByte));
        }

        public override TextureHandle DepthTexture(IVector2 pixels)
        {
            return new OglTextureHandle(OglTextures.CreateTexture(pixels, GlPixelFormat.DepthComponent, GlPixelType.Float));
        }

        public override void ClearTexture(int unit)
        {
            GL.ActiveTexture(GlTextureUnit.Texture0 + unit);
            GL.BindTexture(GlTextureTarget.Texture2D, 0);
        }

        public override uint Compile(string vertexShader, string fragShader, string fragColorChannel, List<string> errors)
        {
            return GL.Compile(vertexShader, fragShader);
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
            GL.Enable(GlCap.Blend);
            GL.BlendFunc(GlBlendingFactor.SrcAlpha, GlBlendingFactor.OneMinusSrcAlpha);
        }

        public override void DisableBlend()
        {
            GL.Disable(GlCap.Blend);
        }

        public override Matrix4 GetViewMatrix(CameraView cameraView)
        {
            return HostMatrix4.GetView(
                View,
                (HostVector3) cameraView.Eye.Vector,
                (HostVector3) cameraView.LookAt,
                (HostVector3) cameraView.Up);
        }

        public override Matrix4 GetPerspectiveProjection(float fovy, float aspect, float near, float far)
        {
            return HostMatrix4.GetProjection(View, fovy, aspect, near, far);
        }

        public override Matrix4 GetPerspectiveOffCenterProjection(float left, float right, float bottom, float top, float near, float far)
        {
            throw new NotImplementedException();
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
            OglRenderer.Render(materialScene, pixels, ambientColor);
        }

        public override void Render(ICollection<RenderObject> scene, IVector2 pixels, Point3 cameraPosition, Color4 ambientColor)
        {
            var materials = GetMaterials(scene);
            var materialScene = scene.GroupBy(obj => (OpenGlMaterial) obj.Material);
            OpenGlMaterial.SetIfDefined(World, materials, "cameraPosition", cameraPosition.Vector);
            OglRenderer.Render(materialScene, pixels, ambientColor);
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
            OglRenderer.TakeColorPicture(this, scene, pixels, ambientColor, cameraPosition, view, texture);
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
            return OglRenderer.TakeDepthPicture(this, scene, pixels, ambientColor, cameraPosition, view, projection);
        }

        public override uint GetUniformLocation(uint shader, string name)
        {
            return GL.GetUniformLocation(shader, name);
        }

        public override void Uniform1(uint location, int value)
        {
            GL.Uniform1(location, value);
        }

        public override void Uniform1(uint location, int count, int[] values)
        {
            throw new NotImplementedException();
        }

        public override void Uniform1(uint location, float value)
        {
            GL.Uniform1(location, value);
        }

        public override void Uniform1(uint location, int count, float[] values)
        {
            throw new NotImplementedException();
        }

        public override void Uniform2(uint location, float valueX, float valueY)
        {
            GL.Uniform2(location, valueX, valueY);
        }

        public override void Uniform3(uint location, float valueX, float valueY, float valueZ)
        {
            GL.Uniform3(location, valueX, valueY, valueZ);
        }

        public override void Uniform3(uint location, ICollection<IVector3> values)
        {
            throw new NotImplementedException();
        }

        public override void Uniform4(uint location, float valueX, float valueY, float valueZ, float valueW)
        {
            GL.Uniform4(location, valueX, valueY, valueZ, valueW);
        }

        public override void UniformMatrix4(uint location, Matrix4 value)
        {
            unsafe
            {
                fixed (float* values = value.Elements)
                {
                    GL.UniformMatrix4(location, true, values);
                }
            }
        }

        internal static IEnumerable<OpenGlMaterial> GetMaterials(ICollection<RenderObject> scene) =>
            scene
                .Select(obj => (OpenGlMaterial) obj.Material)
                .Distinct();
    }
}
