#version 330

uniform sampler2D overlay_texture;
in vec2 uv;
out vec4 fragment_color;

void main() {
    vec4 bgra = texture(overlay_texture, uv);
    fragment_color = vec4(bgra.bgr, bgra.a);
}
