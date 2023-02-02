using System.Collections.Generic;
using System.Linq;
using SharpGfx.OpenGL.Materials;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL;

internal static class GlRenderer
{
    public static void Render(
        GlApi gl,
        IEnumerable<IGrouping<OpenGlMaterial, Instance>> scene,
        int width, int height,
        Color3 background)
    {
        gl.Enable(GlCap.DepthTest);
        gl.Viewport(0, 0, width, height);
        gl.ClearColor(background.R, background.G, background.B, 1);
        gl.Clear(GlBufferBit.Color | GlBufferBit.Depth);

        foreach (var item in scene)
        {
            var material = item.Key;
            material.Apply();

            foreach (var instance in item)
            {
                material.Set("model", true, instance.Transform);
                material.CheckInputs();

                instance.Render();

                material.Set("model", true, instance.Space.Identity4);
                material.Reset("model");
            }

            material.UnApply();
        }
    }

    public static void TakeColorPicture(
        GlApi gl,
        IEnumerable<IGrouping<OpenGlMaterial, Instance>> materialScene,
        int width, int height,
        Color3 background,
        TextureHandle texture)
    {
        const GlFramebufferTarget bufferTarget = GlFramebufferTarget.Framebuffer;
        const GlFramebufferAttachment attachment = GlFramebufferAttachment.ColorAttachment0;
        const GlTextureTarget textureTarget = GlTextureTarget.Texture2D;

        using (new GlFrameRenderBuffer(gl, width, height, GlRenderbufferStorage.Depth24Stencil8, GlFramebufferAttachment.DepthStencilAttachment))
        {
            gl.FramebufferTexture2D(bufferTarget, attachment, textureTarget, ((GlTextureHandle) texture).Handle, 0);

            Render(gl, materialScene, width, height, background);

            gl.FramebufferTexture2D(bufferTarget, attachment, textureTarget, 0, 0);
        }
    }

    public static void TakeDepthPicture(
        GlApi gl,
        IEnumerable<IGrouping<OpenGlMaterial, Instance>> materialScene,
        int width, int height,
        Color3 background,
        TextureHandle texture)
    {
        const GlFramebufferTarget bufferTarget = GlFramebufferTarget.Framebuffer;
        const GlFramebufferAttachment attachment = GlFramebufferAttachment.DepthAttachment;
        const GlTextureTarget textureTarget = GlTextureTarget.Texture2D;

        using (new GlFrameRenderBuffer(gl, width, height, GlRenderbufferStorage.DepthComponent, attachment))
        {
            gl.FramebufferTexture2D(bufferTarget, attachment, textureTarget, ((GlTextureHandle)texture).Handle, 0);

            Render(gl, materialScene, width, height, background);

            gl.FramebufferTexture2D(bufferTarget, attachment, textureTarget, 0, 0); // detach
        }
    }
}