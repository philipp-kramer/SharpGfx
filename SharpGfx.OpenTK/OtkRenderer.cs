using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using SharpGfx.OpenTK.Materials;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal static class OtkRenderer
    {
        public static void SetProjection(IEnumerable<IGrouping<OtkShadedMaterial, RenderObject>> scene, Matrix4 projection)
        {
            foreach (var shader in scene.Select(obj => obj.Key.Shading))
            {
                shader.DoInContext(() => shader.Set("projection", projection));
            }
        }

        public static void SetCameraView(Device device, IEnumerable<IGrouping<OtkShadedMaterial, RenderObject>> scene, CameraView cameraView)
        {
            var view = device.GetViewMatrix(cameraView);
            foreach (var shader in scene.Select(obj => obj.Key.Shading))
            {
                shader.DoInContext(() => shader.Set("cameraView", view));
            }
        }

        public static void Render(
            IEnumerable<IGrouping<OtkShadedMaterial, RenderObject>> scene, 
            Vector2 pixels,
            Point3 cameraPosition,
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
                var shading = material.Shading;
                shading.DoInContext(() =>
                {
                    material.Apply(cameraPosition);

                    foreach (var obj in materialObject)
                    {
                        shading.Set("model", obj.Transform);
                        shading.CheckInputs();

                        obj.Render();

                        shading.ResetIdentityMatrix4("model");
                    }

                    material.UnApply();
                });
            }
        }

        public static void TakeColorPicture(
            Device device,
            ICollection<RenderObject> scene,
            Vector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            TextureHandle texture)
        {
            using (new OtkFrameRenderBuffer(pixels))
            {
                if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                {
                    throw new InvalidOperationException("framebuffer not configured correctly");
                }
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, ((OtkTextureHandle)texture).Handle, 0);

                device.CheckSpaces(scene);
                
                var groupedScene = scene
                    .GroupBy(obj => (OtkShadedMaterial)obj.Material)
                    .ToList();

                SetCameraPosition(device.World, groupedScene, cameraPosition);
                SetCameraView(device, groupedScene, cameraView);
                Render(groupedScene, pixels, cameraPosition, ambientColor);

                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, 0, 0);
            }
        }

        public static TextureHandle TakeDepthPicture(
            Device device,
            ICollection<RenderObject> scene,
            Vector2 pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            Matrix4 projection)
        {
            var depthTexture = device.DepthTexture(pixels);

            using (new OtkFrameRenderBuffer(pixels))
            {
                GL.FramebufferTexture2D(
                    FramebufferTarget.Framebuffer, 
                    FramebufferAttachment.DepthAttachment,
                    TextureTarget.Texture2D, 
                    ((OtkTextureHandle) depthTexture).Handle, 0);

                GL.DrawBuffer(DrawBufferMode.None);
                GL.ReadBuffer(ReadBufferMode.None);

                if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                {
                    throw new InvalidOperationException("framebuffer not configured correctly");
                }

                device.CheckSpaces(scene);

                using var material = new NopMaterial();
                var nopMaterialScene = new[]
                {
                    new Grouping<OtkShadedMaterial, RenderObject>(material, scene)
                };
                SetProjection(nopMaterialScene, projection);
                SetCameraPosition(device.World, nopMaterialScene, cameraPosition); // supply scene also the same way as for Render
                SetCameraView(device, nopMaterialScene, cameraView);
                Render(nopMaterialScene, pixels, cameraPosition, ambientColor);

                GL.FramebufferTexture2D(
                    FramebufferTarget.Framebuffer, 
                    FramebufferAttachment.DepthAttachment,
                    TextureTarget.Texture2D, 
                    0, 0); // detach

                return depthTexture;
            }
        }

        private static void SetCameraPosition(Space world, IEnumerable<IGrouping<OtkShadedMaterial, RenderObject>> scene, Point3 position)
        {
            if (!position.Vector.In(world)) throw new ArgumentException("needs to be in world-space", nameof(position));
            foreach (var shader in scene.Select(obj => obj.Key.Shading))
            {
                shader.DoInContext(() => shader.SetUnchecked("cameraPosition", position.Vector));
            }
        }
    }
}
