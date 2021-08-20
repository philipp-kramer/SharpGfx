using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal static class OtkRenderer
    {
        public static void SetProjection(ICollection<RenderObject> scene, Matrix4 projection)
        {
            foreach (var shader in GetShaders(scene))
            {
                shader.DoInContext(() => shader.Set("projection", projection));
            }
        }

        public static void SetCameraView(Device device, ICollection<RenderObject> scene, CameraView cameraView)
        {
            var view = device.GetViewMatrix(cameraView);
            foreach (var shader in GetShaders(scene))
            {
                shader.DoInContext(() => shader.Set("cameraView", view));
            }
        }

        public static void Render(
            ICollection<RenderObject> scene, 
            Size pixels,
            Point3 cameraPosition,
            Color4 ambientColor)
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Viewport(0, 0, pixels.Width, pixels.Height);
            GL.ClearColor(
                    ambientColor.R, 
                    ambientColor.G, 
                    ambientColor.B,
                    ambientColor.A);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            foreach (var materialObjects in scene.GroupBy(o => (OtkShadedMaterial)o.Material))
            {
                var material = materialObjects.Key;
                material.Shading.DoInContext(() =>
                {
                    material.Apply(cameraPosition);
                    foreach (var o in materialObjects)
                    {
                        o.Render();
                    }
                    material.UnApply();
                });
            }
        }

        public static void TakeColorPicture(
            Device device,
            ICollection<RenderObject> scene,
            Size pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView,
            TextureHandle texture)
        {
            using (device.FrameRenderBuffer(pixels))
            {
                if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                {
                    throw new InvalidOperationException("framebuffer not configured correctly");
                }
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, ((OtkTextureHandle)texture).Handle, 0);

                device.CheckSpaces(scene);
                SetCameraPosition(device.World, scene, cameraPosition);
                SetCameraView(device, scene, cameraView);
                Render(scene, pixels, cameraPosition, ambientColor);

                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, 0, 0);
            }
        }

        public static TextureHandle TakeDepthPicture(
            Device device,
            ICollection<RenderObject> scene,
            Size pixels,
            Color4 ambientColor,
            Point3 cameraPosition,
            CameraView cameraView)
        {
            var depthTexture = device.DepthTexture(pixels);

            using (device.FrameBuffer())
            {
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, TextureTarget.Texture2D, ((OtkTextureHandle)depthTexture).Handle, 0);

                GL.DrawBuffer(DrawBufferMode.None);
                GL.ReadBuffer(ReadBufferMode.None);

                if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                {
                    throw new InvalidOperationException("framebuffer not configured correctly");
                }

                device.CheckSpaces(scene);
                SetCameraPosition(device.World, scene, cameraPosition);
                SetCameraView(device, scene, cameraView);
                Render(scene, pixels, cameraPosition, ambientColor);

                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, TextureTarget.Texture2D, 0, 0); // detach
            }

            return depthTexture;
        }

        private static void SetCameraPosition(Space world, ICollection<RenderObject> scene, Point3 position)
        {
            if (!position.Vector.In(world)) throw new ArgumentException("needs to be in world-space", nameof(position));
            foreach (var shader in GetShaders(scene))
            {
                shader.DoInContext(() => shader.Set("cameraPosition", position.Vector));
            }
        }

        private static IEnumerable<OtkShading> GetShaders(ICollection<RenderObject> scene)
        {
            return scene.Select(o => ((OtkShadedMaterial)o.Material).Shading).Distinct();
        }
    }
}
