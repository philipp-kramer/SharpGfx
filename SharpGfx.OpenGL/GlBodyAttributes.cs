namespace SharpGfx.OpenGL;

public record GlSurfaceAttribute(string ParameterName, int Rank, float[] Values) : SurfaceAttribute(Rank, Values);
public record GlPositionSurfaceAttribute(float[] Values) : GlSurfaceAttribute("positionIn", 3, Values); 
public record GlNormalSurfaceAttribute(float[] Values) : GlSurfaceAttribute("normalIn", 3, Values);
public record GlTexCoordSurfaceAttribute(float[] Values) : GlSurfaceAttribute("texCoordIn", 2, Values);
