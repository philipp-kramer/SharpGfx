using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GlslParser;
using GlslParser.Tree;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Materials;

public abstract class OpenGlMaterial : Material, IDisposable
{
    private readonly Dictionary<string, Direction> _netChannels;
    private readonly Dictionary<string, Direction> _inputs;
    private readonly uint _handle;
    private readonly List<DeclarationNode> _declarations;

    protected GlApi GL { get; }
    protected bool CheckUndefinedChannels { get; set; } = true;
    public bool Transparent { get; set; }

    protected OpenGlMaterial(GlApi gl, string vertexShader, string fragShader)
    {
        GL = gl;
        _netChannels = new Dictionary<string, Direction>();
        _inputs = new Dictionary<string, Direction>();

        var errors = new List<string>();

        _declarations = GetDeclarations(vertexShader, errors);
        var fragmentDeclarations = GetDeclarations(fragShader, errors);
        CheckShader(_declarations);
        CheckShader(fragmentDeclarations);
        _declarations.AddRange(fragmentDeclarations.Where(fd => _declarations.All(vd => vd.Name != fd.Name)));

        var fragOutChannels = fragmentDeclarations
            .OfType<ChannelNode>()
            .Where(c => c.Direction == Direction.Out)
            .ToList();
        if (fragOutChannels.Count != 1)
        {
            throw new ArgumentException($"the fragment shader must have a single output defining the color, but it has {fragOutChannels.Count}.");
        }

        _handle = gl.Compile(vertexShader, fragShader, fragOutChannels[0].Name, errors);

        Report(errors);
    }

    protected override void DoInContext(Action action)
    {
        try
        {
            GL.UseProgram(_handle);
            action();
        }
        finally
        {
            GL.UseProgram(0);
        }
    }

    public override void Apply()
    {
        GL.UseProgram(_handle);
        if (Transparent)
        {
            GL.Enable(GlCap.Blend);
            GL.BlendFunc(GlBlendingFactor.SrcAlpha, GlBlendingFactor.OneMinusSrcAlpha);
        }
    }

    public override void UnApply()
    {
        if (Transparent)
        {
            GL.Disable(GlCap.Blend);
        }
        GL.UseProgram(0);
    }

    internal void SetVertexArrayAttributes(uint vertexArray, IVertexAttribute[] attributes, uint[] vertexBuffers)
    {
        DoInContext(() =>
        {
            GL.BindVertexArray(vertexArray);

            for (int i = 0; i < attributes.Length; i++)
            {
                string name;
                var attribute = attributes[i];
                name = GetName(attribute);
                _inputs[name] = Direction.In;

                GL.BindBuffer(GlBufferTarget.ArrayBuffer, vertexBuffers[i]);

                uint location = GL.GetAttributeLocation(_handle, name);
                GL.EnableVertexAttribArray(location);
                GL.FloatVertexAttribPointer(location, attribute.Rank, attribute.Rank, 0);

                GL.BindBuffer(GlBufferTarget.ArrayBuffer, 0);
            }

            GL.BindVertexArray(0);
        });
    }

    private static string GetName(IVertexAttribute attribute)
    {
        return attribute switch
        {
            GlVertexAttribute glva => glva.Name,
            PositionVa => "positionIn",
            NormalVa => "normalIn",
            TexPositionVa => "texCoordIn",
            _ => throw new ArgumentOutOfRangeException(nameof(attribute))
        };
    }

    private static List<DeclarationNode> GetDeclarations(string shader, List<string> errors)
    {
        var reader = new StringReader(shader);
        var parser = new Parser(new Lexer(reader, errors), errors);
        var program = parser.ParseProgram();
        var declarations = program.Declarations;
        return declarations;
    }

    protected void CheckShader(List<DeclarationNode> declarations)
    {
        var channels = declarations
            .OfType<ChannelNode>()
            .ToList();
        foreach (var channel in channels)
        {
            string channelName = channel.Name;
            var direction = channel.Direction;
            if (direction == Direction.In && _netChannels.TryGetValue(channelName, out var prev))
            {
                if (prev == Direction.Out)
                {
                    _netChannels.Remove(channelName);
                }
                else
                {
                    throw new ArgumentException("matching shader channels require 'out' in vertex and 'in' in fragment shader", channelName);
                }
            }
            else
            {
                var declaration = declarations.SingleOrDefault(d => d.Name == channel.Variable.Type.Name);
                if (declaration is StructNode @struct)
                {
                    foreach (var member in @struct.Members)
                    {
                        _netChannels.Add($"{channelName}.{member.Name}", direction);
                    }
                }
                else
                {
                    _netChannels.Add(channelName, direction);
                }
            }
        }
    }

    public static void SetIfDefined(Space space, IEnumerable<OpenGlMaterial> materials, string name, IVector3 vector)
    {
        if (!vector.In(space)) throw new ArgumentException($"needs to be in {space.Domain}-space", nameof(vector));

        foreach (var material in materials)
        {
            material.DoInContext(() =>
            {
                if (material._netChannels.ContainsKey(name))
                {
                    material.Set(name, vector);
                }
            });
        }
    }

    public static void Set(IEnumerable<OpenGlMaterial> materials, string name, bool transpose, Matrix4 matrix)
    {
        foreach (var material in materials)
        {
            material.DoInContext(() => material.Set(name, transpose, matrix));
        }
    }

    private void SetUniform(string channel)
    {
        if (CheckUndefinedChannels)
        {
            if (!_netChannels.ContainsKey(channel))
            {
                throw new ArgumentException($"shader channel {channel} not found");
            }
            if (_netChannels[channel] != Direction.Uniform)
            {
                Throw(channel, "shader channel is not uniform");
            }
        }

        _inputs[channel] = Direction.Uniform;
    }

    public void Set(string name, int value)
    {
        SetUniform(name);
        GL.Uniform1(GL.GetUniformLocation(_handle, name), value);
    }

    public void Set(string name, float value)
    {
        SetUniform(name);
        GL.Uniform1(GL.GetUniformLocation(_handle, name), value);
    }

    public void Set(string name, IVector2 value)
    {
        SetUniform(name);
        GL.Uniform2(GL.GetUniformLocation(_handle, name), value.X, value.Y);
    }

    public void Set(string name, IVector3 value)
    {
        SetUniform(name);
        GL.Uniform3(GL.GetUniformLocation(_handle, name), value.X, value.Y, value.Z);
    }

    public void Set(string name, IVector4 value)
    {
        SetUniform(name);
        GL.Uniform4(GL.GetUniformLocation(_handle, name), value.X, value.Y, value.Z, value.W);
    }

    public void Set(string name, bool transpose, Matrix4 value)
    {
        SetUniform(name);
        GL.UniformMatrix4(GL.GetUniformLocation(_handle, name), transpose, value);
    }

    public void Set(string name, float[] values)
    {
        SetUniform(name);
        GL.Uniform1(GL.GetUniformLocation(_handle, name), values);
    }

    public void Set(string name, IVector3[] values)
    {
        SetUniform(name);
        var unrolled = new float[3 * values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            unrolled[3*i] = values[i].X;
            unrolled[3*i + 1] = values[i].Y;
            unrolled[3*i + 2] = values[i].Z;
        }
        GL.Uniform3(GL.GetUniformLocation(_handle, name), unrolled);
    }

    public void Reset(string input)
    {
        if (CheckUndefinedChannels && !_netChannels.ContainsKey(input)) throw new ArgumentException("shader channel not found", nameof(input));
        _inputs.Remove(input);
    }

    public void CheckInputs()
    {
        foreach (var channel in _netChannels)
        {
            if (_inputs.TryGetValue(channel.Key, out var kind))
            {
                if (kind != channel.Value)
                {
                    Throw(channel.Key, $"shader input {kind} {channel.Key} should be {channel.Value}");
                }
            }
            else if (CheckUndefinedChannels && channel.Value != Direction.Out)
            {
                Throw(channel.Key, $"shader {kind} channel {channel.Key} has no input");
            }
        }
    }

    protected static void Report(List<string> errors)
    {
        foreach (var error in errors)
        {
            if (!string.IsNullOrEmpty(error)) Console.WriteLine(error);
        }
    }

    private void ReleaseUnmanagedResources()
    {
        GL.DeleteProgram(_handle);
    }

    protected virtual void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
    }

    public override void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(true);
    }

    private void Throw(string channel, string message)
    {
        var shader = _declarations.Any(d => d.Name == channel)
            ? "vertex"
            : "fragment";
        throw new ArgumentException($"{shader} {message}");
    }

    ~OpenGlMaterial()
    {
        GL.Add(() => Dispose(false));
    }
}