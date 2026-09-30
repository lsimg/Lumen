#version 330

uniform sampler2D base_texture;
uniform vec4 base_color;
uniform vec3 highlight_color;
uniform vec3 emissive_color;
uniform float metallic;
uniform float roughness;
uniform float highlight_level;
uniform float opacity;
uniform float model_brightness;
uniform float scan_strength;
uniform int region_count;
uniform vec3 region_min0;
uniform vec3 region_max0;
uniform vec3 region_min1;
uniform vec3 region_max1;
uniform vec3 region_min2;
uniform vec3 region_max2;
uniform vec3 region_min3;
uniform vec3 region_max3;
uniform float region_highlight_level;
uniform vec3 region_highlight_color;
in vec3 normal;
in vec3 world_position;
in vec3 scene_position;
in vec2 texcoord;
out vec4 fragment_color;

float inside_region(vec3 point, vec3 lower, vec3 upper) {
    vec3 above = step(lower, point);
    vec3 below = step(point, upper);
    return above.x * above.y * above.z * below.x * below.y * below.z;
}

void main() {
    vec4 authored = texture(base_texture, vec2(texcoord.x, 1.0 - texcoord.y)) * base_color;
    vec3 view_direction = normalize(vec3(0.0, 0.0, 3.2) - world_position);
    vec3 safe_normal = length(normal) > 0.0001 ? normalize(normal) : view_direction;
    float facing = max(dot(safe_normal, view_direction), 0.0);
    float fresnel = pow(1.0 - abs(dot(safe_normal, view_direction)), 2.0);
    float scanline = 0.5 + 0.5 * sin(world_position.y * 85.0);
    float light = 0.50 + 0.50 * facing;
    vec3 half_direction = normalize(view_direction + normalize(vec3(0.35, 0.70, 0.55)));
    float specular_power = mix(96.0, 7.0, roughness);
    float specular = pow(max(dot(safe_normal, half_direction), 0.0), specular_power);
    vec3 reflectance = mix(vec3(0.04), authored.rgb, metallic);
    vec3 original = authored.rgb * light * (1.0 - metallic * 0.72)
                  + reflectance * specular * mix(1.0, 0.25, roughness)
                  + emissive_color;
    original *= model_brightness;
    float region_mask = 0.0;
    if (region_count > 0) region_mask = max(region_mask, inside_region(scene_position, region_min0, region_max0));
    if (region_count > 1) region_mask = max(region_mask, inside_region(scene_position, region_min1, region_max1));
    if (region_count > 2) region_mask = max(region_mask, inside_region(scene_position, region_min2, region_max2));
    if (region_count > 3) region_mask = max(region_mask, inside_region(scene_position, region_min3, region_max3));
    float regional = region_mask * region_highlight_level;
    float active_level = max(highlight_level, regional);
    vec3 active_color = regional > highlight_level ? region_highlight_color : highlight_color;
    float rim = active_level * (0.25 + 0.75 * fresnel);
    vec3 color = mix(original, active_color, clamp(rim, 0.0, 0.72));
    color += active_color * rim * 0.20;
    color += active_color * scanline * scan_strength * max(region_mask, step(0.02, highlight_level));
    fragment_color = vec4(color, authored.a * opacity);
}
