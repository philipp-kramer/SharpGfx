using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.OpenGL.Materials;
using SharpGfx.Primitives;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpGfx.OpenGL;

public abstract class OpenGlDevice : Device
{
    public GlApi GL { get; }

    protected OpenGlDevice(GlApi gl)
    {
        GL = gl;
    }

    public override Material Emissive(Color4 color)
    {
        return new EmissiveMaterial(GL, color);
    }

    /// <summary>
    /// implicitly uses texture unit 0
    /// </summary>
    public override Material Emissive(TextureHandle color)
    {
        return new EmissiveTextureMaterial(GL, color, 0);
    }

    public override Material Lambert(Color4 color, Lighting lighting)
    {
        return new LambertMaterial(GL, color, lighting);
    }

    /// <summary>
    /// implicitly uses texture unit 0
    /// </summary>
    public override Material Lambert(TextureHandle color, Lighting lighting)
    {
        return new LambertColorTextureMaterial(GL, color, lighting, 0);
    }

    public override Material Phong(Reflectance<Color3> reflectance, Lighting lighting)
    {
        return new PhongMaterial(this, reflectance, lighting);
    }

    /// <summary>
    /// implicitly uses texture unit 0
    /// </summary>
    public override Material Phong(Reflectance<TextureHandle> reflectance, Lighting lighting)
    {
        return new PhongTextureMaterial(this, reflectance, lighting, 0);
    }

    public override Surface Surface(Material material, params SurfaceAttribute[] attributes)
    {
        return new GlSurface(GL, (OpenGlMaterial) material, attributes);
    }

    public override Surface Surface(Material material, ushort[] triangles, params SurfaceAttribute[] attributes)
    {
        return new GlIndexedSurface<ushort>(GL, (OpenGlMaterial) material, triangles, attributes);
    }

    public override Surface Surface(Material material, uint[] triangles, params SurfaceAttribute[] attributes)
    {
        return new GlIndexedSurface<uint>(GL, (OpenGlMaterial) material, triangles, attributes);
    }

    public override Instance Instance(Space space, string name, Surface surface)
    {
        return new GlInstance(GL, space, name, (GlSurface) surface);
    }

    public override TextureHandle Texture(Image<Rgba32> image)
    {
        return new GlTextureHandle(GL, GlTextures.CreateMipmapTexture(GL, image));
    }

    public override void Render(Window window, Rendering rendering)
    {
        var materialScene = GetAllInstancesByMaterial(rendering.Scene);
        GlRenderer.Render(GL, materialScene, window.Width, window.Height, rendering.Background);
        GL.ExecutePending();
    }

    public override void RenderWithCamera(Window window, CameraRendering rendering)
    {
        int width = window.Width;
        int height = window.Height;
        var materialScene = GetAllInstancesByMaterial(rendering.Scene);
        SetViewProjection(rendering, (float) width / height, materialScene);
        GlRenderer.Render(GL, materialScene, width, height, rendering.Background);
        GL.ExecutePending();
    }

    private void SetViewProjection(CameraRendering rendering, float aspect, List<IGrouping<OpenGlMaterial, GlInstance>> materialScene)
    {
        CheckSpaces(rendering.Scene);
        var materials = materialScene
            .Select(m => m.Key)
            .ToList();
        var camera = rendering.Camera;
        if (!camera.Position.Vector.In(World)) throw new ArgumentException("needs to be in world-space", nameof(camera));
        var view = GetViewMatrix(rendering.View);
        OpenGlMaterial.Set(materials, "cameraView", true, view);
        OpenGlMaterial.SetIfDefined(World, materials, "cameraPosition", camera.Position.Vector);
        var projection = GetProjection(aspect, camera);
        OpenGlMaterial.Set(materials, "projection", true, projection);
    }

    public abstract Matrix4 GetViewMatrix(CameraView cameraView);
    public abstract Matrix4 GetProjection(float aspect, Camera camera);

    protected abstract Matrix4 GetOffCenterProjection(float left, float right, float bottom, float top, float near, float far);
    public Matrix4 GetOffCenterProjection(IVector3 center, IVector3 positiveX, IVector3 positiveY, float ratio, float near, float far, Matrix4 view)
    {
        var left = (center - positiveX).Extend(View, 1);
        var right = (center + positiveX).Extend(View, 1);
        left *= ratio;
        right *= ratio;
        var l = (view * left).X;
        var r = (view * right).X;

        var bot = (center - positiveY).Extend(View, 1);
        var top = (center + positiveY).Extend(View, 1);
        bot *= ratio;
        top *= ratio;
        var b = (view * bot).Y;
        var t = (view * top).Y;

        return GetOffCenterProjection(l, r, b, t, near, far);
    }

    public TextureHandle RgbTexture(int width, int height)
    {
        return new GlTextureHandle(GL, GlTextures.CreateTexture(GL, width, height, GlPixelFormat.Rgb, GlPixelType.UnsignedByte));
    }

    public TextureHandle DepthTexture(int width, int height)
    {
        return new GlTextureHandle(GL, GlTextures.CreateTexture(GL, width, height, GlPixelFormat.DepthComponent, GlPixelType.Float));
    }

    public void TakeColorPicture(
        List<Instance> scene,
        Point3 cameraPosition,
        Matrix4 view,
        Matrix4 projection,
        int width, int height,
        Color3 background,
        TextureHandle texture)
    {
        var materialScene = GetAllInstancesByMaterial(scene);
        SetViewProjection(materialScene, cameraPosition, projection, view);
        GlRenderer.TakeColorPicture(GL, materialScene, width, height, background, texture);
    }

    public void TakeDepthPicture(
        List<Instance> scene, 
        Point3 cameraPosition,
        Matrix4 view,
        Matrix4 projection,
        int width, int height,
        Color3 background,
        TextureHandle texture)
    {
        var materialScene = GetAllInstancesByMaterial(scene);
        SetViewProjection(materialScene, cameraPosition, projection, view);
        GlRenderer.TakeDepthPicture(GL, materialScene, width, height, background, texture);
    }

    private void SetViewProjection(
        List<IGrouping<OpenGlMaterial, GlInstance>> materialScene,
        Point3 cameraPosition,
        Matrix4 projection, 
        Matrix4 view)
    {
        var materials = materialScene
            .Select(m => m.Key)
            .ToList();
        OpenGlMaterial.Set(materials, "cameraView", true, view);
        OpenGlMaterial.SetIfDefined(World, materials, "cameraPosition", cameraPosition.Vector);
        OpenGlMaterial.Set(materials, "projection", true, projection);
    }

    private static List<IGrouping<OpenGlMaterial, GlInstance>> GetAllInstancesByMaterial(List<Instance> scene)
    {
        return scene
            .SelectMany(instance => instance.All)
            .OfType<GlInstance>()
            .GroupBy(instance => (OpenGlMaterial) instance.Surface.Material)
            .ToList();
    }
}