#version 330

uniform sampler2D camera_texture;
uniform vec2 camera_texel_size;
uniform float camera_sharpen;
in vec2 uv;
out vec4 fragment_color;

void main() {
    vec3 center = texture(camera_texture, uv).bgr;
    vec3 neighbours = (
        texture(camera_texture, uv + vec2(camera_texel_size.x, 0.0)).bgr
        + texture(camera_texture, uv - vec2(camera_texel_size.x, 0.0)).bgr
        + texture(camera_texture, uv + vec2(0.0, camera_texel_size.y)).bgr
        + texture(camera_texture, uv - vec2(0.0, camera_texel_size.y)).bgr
    ) * 0.25;
    vec3 sharpened = clamp(center + (center - neighbours) * camera_sharpen, 0.0, 1.0);
    float focus = 1.0 - smoothstep(
        0.18,
        0.58,
        length((uv - vec2(0.5)) * vec2(0.86, 1.0))
    );
    fragment_color = vec4(sharpened * (1.0 - focus * 0.10), 1.0);
}
