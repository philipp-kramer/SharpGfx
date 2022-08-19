using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.OpenGL.Shading;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL
{
    internal static class OglRenderer
    {
        public static void Render(
            IEnumerable<IGrouping<OpenGlMaterial, RenderObject>> scene,
            IVector2 pixels,
            Color4 ambientColor)
        {
            GL.Enable(GlCap.DepthTest);
            GL.Viewport(0, 0, (int) pixels.X, (int) pixels.Y);
            GL.ClearColor(
                ambientColor.R,
                ambientColor.G,
                ambientColor.B,
                ambientColor.A);
            GL.Clear(GlBufferBit.Color | GlBufferBit.Depth);

            foreach (var materialObject in scene)
            {
                var material = materialObject.Key;
                material.Apply();

                foreach (var obj in materialObject)
                {
                    material.Set("model", obj.Transform);
                    material.CheckInputs();

                    obj.Render();

                    material.Set("model", obj.Space.Identity4);
                    material.Reset("model");
                }

                material.UnApply();
            }
        }

        public static void TakeColorPicture(
            Device device,
            ICollection<RenderObject> scene,
            IVector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            Matrix4 view,
            TextureHandle texture)
        {
            using (new OglFrameRenderBuffer(pixels))
            {
                GL.FramebufferTexture2D(
                    GlFramebufferTarget.Framebuffer, 
                    GlFramebufferAttachment.ColorAttachment0, 
                    GlTextureTarget.Texture2D, 
                    ((OglTextureHandle) texture).Handle, 0);

                device.CheckSpaces(scene);

                var materials = OglDevice
                    .GetMaterials(scene)
                    .ToArray();
                var materialScene = scene.GroupBy(obj => (OpenGlMaterial) obj.Material);

                OpenGlMaterial.SetIfDefined(device.World, materials, "cameraPosition", cameraPosition.Vector);
                OpenGlMaterial.Set(materials, "cameraView", view);
                Render(materialScene, pixels, ambientColor);

                GL.FramebufferTexture2D(
                    GlFramebufferTarget.Framebuffer, 
                    GlFramebufferAttachment.ColorAttachment0, 
                    GlTextureTarget.Texture2D, 
                    0, 0);
            }
        }

        public static TextureHandle TakeDepthPicture(
            Device device,
            ICollection<RenderObject> scene,
            IVector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            Matrix4 view,
            Matrix4 projection)
        {
            var depthTexture = device.DepthTexture(pixels);

            using (new OglFrameRenderBuffer(pixels))
            {
                GL.FramebufferTexture2D(
                    GlFramebufferTarget.Framebuffer,
                    GlFramebufferAttachment.DepthAttachment,
                    GlTextureTarget.Texture2D,
                    ((OglTextureHandle) depthTexture).Handle, 0);

                GL.DrawBuffer(0);
                GL.ReadBuffer(0);

                if (GL.CheckFramebufferStatus(GlFramebufferTarget.Framebuffer) != GlFramebufferErrorCode.FramebufferComplete)
                {
                    throw new InvalidOperationException("framebuffer not configured correctly");
                }

                device.CheckSpaces(scene);

                using var material = new NopMaterial(device);
                var materials = new[] {material};
                var nopMaterialScene = new[]
                {
                    new Grouping<OpenGlMaterial, RenderObject>(material, scene)
                };

                OpenGlMaterial.SetIfDefined(device.World, materials, "cameraPosition", cameraPosition.Vector); // supply scene also the same way as for Render
                OpenGlMaterial.Set(materials, "cameraView", view);
                OpenGlMaterial.Set(materials, "projection", projection);
                Render(nopMaterialScene, pixels, ambientColor);

                GL.FramebufferTexture2D(
                    GlFramebufferTarget.Framebuffer,
                    GlFramebufferAttachment.DepthAttachment,
                    GlTextureTarget.Texture2D,
                    0, 0); // detach

                return depthTexture;
            }
        }
    }
}