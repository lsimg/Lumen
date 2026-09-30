#version 330

uniform mat4 mvp;
uniform mat4 model_matrix;
uniform mat4 scene_matrix;
uniform float gripper_enabled;
uniform float gripper_open;
uniform vec4 gripper_params;
uniform float gripper_closed_angle;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
out vec3 normal;
out vec3 world_position;
out vec3 scene_position;
out vec2 texcoord;

void main() {
    vec3 position = in_position;
    if (gripper_enabled > 0.5) {
        float start_x = gripper_params.x;
        float end_x = gripper_params.y;
        float weight = clamp((start_x - position.x) / max(start_x - end_x, 0.00001), 0.0, 1.0);
        float side = position.y >= 0.0 ? 1.0 : -1.0;
        vec3 pivot = vec3(start_x, side * gripper_params.z, gripper_params.w);
        float angle = radians((1.0 - gripper_open) * gripper_closed_angle * side);
        float cosine = cos(angle);
        float sine = sin(angle);
        vec2 relative = position.xy - pivot.xy;
        vec3 hinged = position;
        hinged.xy = pivot.xy + vec2(cosine * relative.x - sine * relative.y, sine * relative.x + cosine * relative.y);
        position = mix(position, hinged, weight);
    }
    vec4 world = model_matrix * vec4(position, 1.0);
    world_position = world.xyz;
    scene_position = (scene_matrix * vec4(position, 1.0)).xyz;
    normal = normalize(mat3(model_matrix) * in_normal);
    texcoord = in_uv;
    gl_Position = mvp * vec4(position, 1.0);
}
