using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using SharpGfx.OpenGL.Shading;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal static class OtkRenderer
    {
        public static void Render(
            IEnumerable<IGrouping<OpenGlMaterial, RenderObject>> scene, 
            IVector2 pixels,
            Color4 ambientColor)
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Viewport(0, 0, (int) pixels.X, (int) pixels.Y);
            GL.ClearColor(
                    ambientColor.R, 
                    ambientColor.G, 
                    ambientColor.B,
                    ambientColor.A);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

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
            const FramebufferTarget bufferTarget = FramebufferTarget.Framebuffer;
            const FramebufferAttachment attachment = FramebufferAttachment.ColorAttachment0;
            const TextureTarget textureTarget = TextureTarget.Texture2D;

            using (new OtkFrameRenderBuffer(pixels, RenderbufferStorage.Depth24Stencil8, FramebufferAttachment.DepthStencilAttachment))
            {
                GL.FramebufferTexture2D(bufferTarget, attachment, textureTarget, ((OtkTextureHandle) texture).Handle, 0);

                device.CheckSpaces(scene);
                
                var materials = OtkDevice
                    .GetMaterials(scene)
                    .ToArray();
                var materialScene = scene.GroupBy(obj => (OpenGlMaterial) obj.Material);

                OpenGlMaterial.SetIfDefined(device.World, materials, "cameraPosition", cameraPosition.Vector);
                OpenGlMaterial.Set(materials, "cameraView", view);
                Render(materialScene, pixels, ambientColor);

                GL.FramebufferTexture2D(bufferTarget, attachment, textureTarget, 0, 0);

            }
        }

        public static void TakeDepthPicture(
            Device device,
            ICollection<RenderObject> scene,
            IVector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            Matrix4 view,
            Matrix4 projection,
            TextureHandle texture)
        {
            const FramebufferTarget bufferTarget = FramebufferTarget.Framebuffer;
            const FramebufferAttachment attachment = FramebufferAttachment.DepthAttachment;
            const TextureTarget textureTarget = TextureTarget.Texture2D;

            using (new OtkFrameRenderBuffer(pixels, RenderbufferStorage.DepthComponent, attachment))
            {
                GL.FramebufferTexture2D(bufferTarget, attachment, textureTarget,  ((OtkTextureHandle)texture).Handle, 0);

                device.CheckSpaces(scene);

                using var material = new NopMaterial(device);
                var materials = new[] { material };
                var nopMaterialScene = new[]
                {
                    new Grouping<OpenGlMaterial, RenderObject>(material, scene)
                };

                OpenGlMaterial.SetIfDefined(device.World, materials, "cameraPosition", cameraPosition.Vector); // supply scene also the same way as for Render
                OpenGlMaterial.Set(materials, "cameraView", view);
                OpenGlMaterial.Set(materials, "projection", projection);
                Render(nopMaterialScene, pixels, ambientColor);

                GL.FramebufferTexture2D(bufferTarget, attachment, textureTarget,  0, 0); // detach
            }
        }
    }
}
