namespace SharpGfx.OptiX.Materials;

// TODO: interpolated materials
// TODO: standard material with metallness and roughness
// TODO: semi-transparent glass material: in hit shader go though body and determine exit ray
// TODO: denoising
public class MirrorMaterial : OptixMaterial
{
    public MirrorMaterial(Device device, string raysCu)
        : base(device, Resources.GetProgram("ray_mirror.cu"))
    {
    }
}