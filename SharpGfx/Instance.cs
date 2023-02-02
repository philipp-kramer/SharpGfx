using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.Primitives;

namespace SharpGfx;

public class Instance : IDisposable
{
    private Space _space;
    private readonly string _name;
    private readonly Instance[] _self;

    public Space Space
    {
        get => _space;
        set => _space = Space.Domain <= value.Domain
            ? value
            : throw new InvalidOperationException("the new space must not be upstream in the pipeline");
    }

    public List<Instance> Children { get; }
    public Matrix4 Transform { get; set; }
    public IEnumerable<Instance> All => Children
        .SelectMany(child => child.All)
        .Concat(_self);

    protected internal Instance(Space space, string name)
    {
        _space = space;
        _name = name;
        _self = new[] { this };
        Children = new List<Instance>();
        Transform = space.Identity4;
    }

    public void Add(Instance child)
    {
        Children.Add(child);
    }

    public virtual void Render() {}

    public virtual Instance Scale(float scale)
    {
        Transform *= Space.Scale4(scale);
        foreach (var child in Children) child.Scale(scale);
        return this;
    }

    public virtual Instance Scale(IVector3 scale)
    {
        Transform *= scale.Space.Scale4(scale);
        foreach (var child in Children) child.Scale(scale);
        return this;
    }

    public virtual Instance Translate(IVector3 delta)
    {
        if (Space.Domain > delta.Space.Domain) throw new ArgumentException("cross space operation");
        Space = delta.Space;
        Transform *= delta.Space.Translation4(delta);
        foreach (var child in Children) child.Translate(delta);
        return this;
    }

    public virtual Instance RotateX(float angle)
    {
        Transform *= Space.RotationX4(angle);
        foreach (var child in Children) child.RotateX(angle);
        return this;
    }

    public virtual Instance RotateY(float angle)
    {
        Transform *= Space.RotationY4(angle);
        foreach (var child in Children) child.RotateY(angle);
        return this;
    }

    public virtual Instance RotateZ(float angle)
    {
        Transform *= Space.RotationZ4(angle);
        foreach (var child in Children) child.RotateZ(angle);
        return this;
    }

    public override string ToString()
    {
        return _name;
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var child in Children) child.Dispose(true);
        }
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(true);
    }

    ~Instance()
    {
        Dispose(false);
    }
}