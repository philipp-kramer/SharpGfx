using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkCompilation
    {
        internal static uint Compile(string vertexShader, string fragShader, string fragColorChannel, List<string> errors)
        {
            var pipeline = new List<int>();
            pipeline.Add(GetCompiledShader(vertexShader, ShaderType.VertexShader, errors));
            pipeline.Add(GetCompiledShader(fragShader, ShaderType.FragmentShader, errors));

            int program = GL.CreateProgram();
            GL.BindFragDataLocation(program, 0, fragColorChannel);
            LinkProgram(program, pipeline, errors);

            return (uint) program;
        }

        private static int GetCompiledShader(string source, ShaderType shaderType, List<string> errors)
        {
            var shader = GL.CreateShader(shaderType);
            GL.ShaderSource(shader, source);
            GL.CompileShader(shader);
            errors.Add(GL.GetShaderInfoLog(shader));
            return shader;
        }

        private static void LinkProgram(int handle, List<int> shaders, List<string> errors)
        {
            foreach (var shader in shaders)
            {
                GL.AttachShader(handle, shader);
            }
            GL.LinkProgram(handle);
            errors.Add(GL.GetProgramInfoLog(handle));

            foreach (var shader in shaders)
            {
                GL.DetachShader(handle, shader);
                GL.DeleteShader(shader);
            }
        }
    }
}