using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenGL
{
    internal static class GL
    {
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "createGlfWindow", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void* CreateWindow(string title, int width, int height);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "isWindowCloseRequested", CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe bool IsWindowCloseRequested(void* glfWindow);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "swapBuffers", CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe void SwapBuffers(void* glfWindow);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getNewWidth", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint getNewWidth();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getNewHeight", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint getNewHeight();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getKey", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint getKey();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseButton", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint getMouseButton();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseAction", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint getMouseAction();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMousePosX", CallingConvention = CallingConvention.Cdecl)]
        internal static extern double getMousePosX();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMousePosY", CallingConvention = CallingConvention.Cdecl)]
        internal static extern double getMousePosY();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseScrollX", CallingConvention = CallingConvention.Cdecl)]
        internal static extern double getMouseScrollX();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseScrollY", CallingConvention = CallingConvention.Cdecl)]
        internal static extern double getMouseScrollY();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "terminateGlfw", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Terminate();


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "viewport", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Viewport(int x, int y, int width, int height);
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "clearColor", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ClearColor(float r, float g, float b, float a);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "clear", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Clear(GlBufferBit bufferBit);



        [DllImport(@"x64/OpenGL.dll", EntryPoint = "genVertexArray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint GenVertexArray();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "genBuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint GenBuffer();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bufferData", CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe void BufferData(GlBufferTarget target, long size, void* data);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getBufferParameter", CallingConvention = CallingConvention.Cdecl)]
        internal static extern long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "floatVertexAttribPointer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void FloatVertexAttribPointer(uint index, int size, int stride, int offset);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindVertexArray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BindVertexArray(uint array);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindBuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BindBuffer(GlBufferTarget target, uint buffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "enableVertexAttribArray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void EnableVertexAttribArray(uint array);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawTriangles", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DrawTriangles(int first, int count);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawIndexedTriangles", CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe void DrawIndexedTriangles(int count, GlType type, void* indices);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "enable", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Enable(GlCap cap);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "disable", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Disable(GlCap cap);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "blendFunc", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor);


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "genTexture", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint GenTexture();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindTexture", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BindTexture(GlTextureTarget target, uint texture);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "texImage2D", CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, void* pixels);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "generateMipmap", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void GenerateMipmap(GlTextureTarget target);
        
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "textureParameterI", CallingConvention = CallingConvention.Cdecl)]
        private static extern void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, int parameter);
        internal static void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, GlTextureParameter parameter) { TextureParameterI(target, name, (int) parameter);}

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "activeTexture", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ActiveTexture(GlTextureUnit glTexture);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteTexture", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DeleteTexture(uint texture);


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "genFramebuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint GenFramebuffer();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindFramebuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BindFramebuffer(GlFramebufferTarget target, uint framebuffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteFramebuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DeleteFramebuffer(uint framebuffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "framebufferTexture2D", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget textarget, uint texture, int level);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "genRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint GenRenderbuffer();

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "renderbufferStorage", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalformat, int width, int height);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "framebufferRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbuffertarget, uint renderbuffer);
        
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "checkFramebufferStatus", CallingConvention = CallingConvention.Cdecl)]
        internal static extern GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target);
        
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DeleteRenderbuffer(uint renderbuffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawBuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DrawBuffer(uint buf);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "readBuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReadBuffer(uint src);



        [DllImport(@"x64/OpenGL.dll", EntryPoint = "compile", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint Compile(string vertexShaderSource, string fragmentShaderSource);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "useProgram", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void UseProgram(uint shaderProgram);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteProgram", CallingConvention = CallingConvention.Cdecl)]
        public static extern void DeleteProgram(uint handle);


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteVertexArray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DeleteVertexArray(uint array);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteBuffer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void DeleteBuffer(uint buffer);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getAttributeLocation", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint GetAttributeLocation(uint buffer, string attributeParameter);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getUniformLocation", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint GetUniformLocation(uint buffer, string name);


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform1i", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint Uniform1(uint location, int value);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform1f", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint Uniform1(uint location, float value);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform2f", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint Uniform2(uint location, float v1, float v2);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform3f", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint Uniform3(uint location, float v1, float v2, float v3);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform4f", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint Uniform4(uint location, float v1, float v2, float v3, float v4);

        [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniformMatrix4f", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe uint UniformMatrix4(uint location, bool transpose, float* values);


        [DllImport(@"x64/OpenGL.dll", EntryPoint = "getTexImage", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, void* pixels);
    }
}
