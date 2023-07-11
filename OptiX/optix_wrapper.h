#pragma once
#include "Geometry.h"
#include "Texture.h"
#include "Material.h"

extern "C"
{
    __declspec(dllexport) Context* create_context();
    __declspec(dllexport) Material* Material_Create(Context context, const char* rays_cu);
    __declspec(dllexport) Material* EmissiveMaterial_Create(Context context, const char* rays_cu, const float* color3);
    __declspec(dllexport) Material* EmissiveTextureMaterial_Create(Context context, const char* rays_cu, Texture* texture);
    
    __declspec(dllexport) Material* LambertMaterial_Create(
        Context context, 
        const char* rays_cu, 
        const float* material_color, 
        const float* ambient_color, 
        const float* light_positions, 
        const float* light_colors, 
        const int point_lights_count);

    __declspec(dllexport) Material* LambertTextureMaterial_Create(
        Context context, 
        const char* rays_cu, 
        Texture* texture, 
        const float* ambient_color, 
        const float* light_positions, 
        const float* light_colors, 
        const int point_lights_count);

    __declspec(dllexport) Material* PhongMaterial_Create(
        Context context,
        const char* rays_cu, 
        const float* material_color,
        const float* ambient_color,
        const float* lights_position,
        const float* lights_color,
        const int point_lights_count,
        const float* material_color_specular,
        const float material_shininess);

    __declspec(dllexport) void Material_Destroy(Material* material);

    __declspec(dllexport) int build(Context& context, Geometry** geometries, Material** materials, const int b_count, OptixInstance** instances, const int i_count, const float* bg_color);
    __declspec(dllexport) int update(Context& context, Geometry** geometries, const int b_count, OptixInstance** instances, const int i_count);
    __declspec(dllexport) int render(Window& window, Camera& camera, Context& context);
    __declspec(dllexport) Window* open_window(Context& context, const char* title, int width, int height);
    __declspec(dllexport) int close(Window& window, Camera& camera, Context& context);
}
