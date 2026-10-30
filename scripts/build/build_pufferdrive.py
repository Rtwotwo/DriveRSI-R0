"""Build only DriveRSI's native adapter against the pinned, unmodified PufferDrive header."""

import argparse
import ast
import hashlib
import json
import platform
import shlex
import subprocess
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Isolated MTGS Python 3.10 deployment.
    import tomli as tomllib


def lifecycle_header(header):
    """Bounded world population is independent of the32 policy/neighbor slots."""

    def replace(old, new, count=1):
        nonlocal header
        if header.count(old) != count:
            raise ValueError("Pinned lifecycle patch mismatch: " + old[:80])
        header = header.replace(old, new)

    replace("int static_agent_indices[MAX_AGENTS];", "int static_agent_indices[DRIVERSI_WORLD_CAPACITY];")
    replace("int expert_static_agent_indices[MAX_AGENTS];", "int expert_static_agent_indices[DRIVERSI_WORLD_CAPACITY];")
    replace(
        "i < env->num_objects && env->num_actors < MAX_AGENTS",
        "i < env->num_objects && env->num_actors < DRIVERSI_WORLD_CAPACITY",
    )
    replace(
        "if (entity->traj_valid[env->init_steps] != 1) {",
        "if (entity->traj_valid[env->init_steps] != 1 && !entity->mark_as_expert) {",
    )
    replace(
        "e->valid = e->traj_valid[env->init_steps];",
        "e->valid = e->traj_valid[env->init_steps];\n        if (!e->valid) { e->x = INVALID_POSITION; e->y = INVALID_POSITION; }",
    )
    replace(
        "if (env->entities[expert_idx].x == INVALID_POSITION)\n            continue;",
        "/* Replay membership can re-enter; move_expert checks this tick. */",
    )
    replace(
        "int t = env->timestep;\n    if (t < 0 || t >= agent->array_size)",
        "int t = env->timestep;\n    agent->valid = t >= 0 && t < agent->array_size && agent->traj_valid[t];\n    if (t < 0 || t >= agent->array_size)",
    )
    start = header.index("int collision_check(")
    end = header.index("int check_lane_aligned(", start)
    block = header[start:end].replace("i < MAX_AGENTS", "i < env->num_actors")
    block = block.replace(
        "if (entity->respawn_timestep != -1)",
        "if (entity->x == INVALID_POSITION || entity->respawn_timestep != -1)",
    )
    header = header[:start] + block + header[end:]
    start = header.index("void compute_observations(")
    end = header.index("int remaining_partner_obs", start)
    block = header[start:end]
    old = """        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors && env->static_agent_count > 0) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }"""
    new = """        int order[DRIVERSI_WORLD_CAPACITY];
        float distance[DRIVERSI_WORLD_CAPACITY];
        for (int j=0; j<env->num_actors; j++) {
            int k = j < env->active_agent_count ? env->active_agent_indices[j]
                : env->static_agent_indices[j-env->active_agent_count];
            order[j]=k;
            float dx=env->entities[k].x-ego_entity->x, dy=env->entities[k].y-ego_entity->y;
            distance[j]=env->entities[k].x == INVALID_POSITION ? INFINITY : dx*dx+dy*dy;
        }
        /* Preserve legacy slot order for <=32; larger worlds use nearest31. */
        if (env->num_actors>MAX_AGENTS) {
            for (int j=1; j<env->num_actors; j++) {
                int k=order[j], q=j; float d=distance[j];
                while(q>0 && distance[q-1]>d) {order[q]=order[q-1];distance[q]=distance[q-1];q--;}
                order[q]=k;distance[q]=d;
            }
        }
        for (int j = 0; j < env->num_actors && cars_seen < MAX_AGENTS-1; j++) {
            int index = order[j];
            if (env->entities[index].x == INVALID_POSITION) continue;"""
    if block.count(old) != 1:
        raise ValueError("Pinned neighbor patch mismatch")
    header = header[:start] + block.replace(old, new) + header[end:]
    return header


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / "clones/PufferDrive")
    parser.add_argument("--raylib", type=Path, default=root / "clones/raylib-5.5_linux_amd64")
    parser.add_argument("--fetch-raylib", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Host/interpreter-specific extension directory; defaults to the package directory",
    )
    args = parser.parse_args()
    config = tomllib.loads((root / "src/config/backends/pufferdrive.toml").read_text())
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("This initial build target is Linux x86_64")
    revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["source"]["revision"]:
        parser.error(f"Backend revision mismatch: {revision}")
    if subprocess.check_output(
        ["git", "-C", str(args.source), "diff", "HEAD", "--name-only"], text=True
    ).strip():
        parser.error("Build requires unmodified upstream source")
    if not (args.raylib / "lib/libraylib.a").exists():
        if not args.fetch_raylib:
            parser.error("Raylib 5.5 is missing; use --fetch-raylib or supply --raylib")
        args.raylib.parent.mkdir(parents=True, exist_ok=True)
        archive = args.raylib.parent / "raylib-5.5_linux_amd64.tar.gz"
        try:
            with urllib.request.urlopen(config["build"]["raylib_url"], timeout=60) as source:
                with archive.open("wb") as destination:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
            with tarfile.open(archive) as bundle:
                bundle.extractall(args.raylib.parent, filter="data")
        finally:
            archive.unlink(missing_ok=True)
    # One Python contract drives header arrays, C admission, and exported capacity.
    tree = ast.parse((root / "src/contracts/scene.py").read_text())
    capacity = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "MAX_WORLD_ACTORS"
                        for target in node.targets))
    if type(capacity) is not int or not 32 <= capacity <= 16384:
        parser.error("Unsupported MAX_WORLD_ACTORS build capacity")
    output_dir = (args.output_dir or root / "src/sim").resolve()
    output = output_dir / ("_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        *shlex.split(sysconfig.get_config_var("CC") or "gcc"),
        "-shared",
        "-fPIC",
        "-O2",
        "-std=gnu11",
        "-fvisibility=hidden",
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        "-DPLATFORM_DESKTOP",
        f"-DDRIVERSI_WORLD_CAPACITY={capacity}",
        f'-DPD_REVISION="{revision}"',
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{np.get_include()}",
        f"-I{args.source.resolve() / 'pufferlib/ocean/drive'}",
        f"-I{args.raylib.resolve() / 'include'}",
        str(root / "src/sim/native.c"),
        str(args.raylib.resolve() / "lib/libraylib.a"),
        "-lm",
        "-lpthread",
        "-ldl",
        "-lrt",
        "-l:libGL.so.1",
        "-l:libX11.so.6",
        "-o",
        str(output),
    ]
    # Pinned upstream grid bounds omit points exactly on its maximum boundary;
    # keep the checkout unchanged and apply fixes to a build copy.
    header = (args.source / "pufferlib/ocean/drive/drive.h").read_text()
    replacements = {
        # Upstream reset initializes goal counters only for controlled actors.
        # Snapshots include every actor and C struct padding, so allocator
        # residue must not become hidden episode state for replay actors.
        "Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));": "Entity *entities = (Entity *)calloc(env->num_entities, sizeof(Entity));",
        "env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);": "env->grid_map->grid_cols = 1 + ceil(grid_width / GRID_CELL_SIZE);",
        "env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);": "env->grid_map->grid_rows = 1 + ceil(grid_height / GRID_CELL_SIZE);",
        "signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);": "/* Preserve physical zero curvature; the straight-line branch handles it. */",
    }
    for old, new in replacements.items():
        if header.count(old) != 1:
            parser.error("Pinned header does not match a required adapter fix")
        header = header.replace(old, new)
    # The upstream centring mean includes the entire reference future. Even
    # when futures are never observed, they perturb float32 rounding now.
    start = header.index("void set_means(Drive *env) {")
    end = header.index("env->world_mean_x = mean_x;", start)
    mean_code = header[start:end]
    old = "for (int j = 0; j < env->entities[i].array_size; j++) {"
    if mean_code.count(old) != 2:
        parser.error("Pinned header does not match initial-state centring fix")
    mean_code = mean_code.replace(old, "for (int j = 0; j < 1; j++) {", 1)
    header = header[:start] + mean_code + header[end:]
    header = lifecycle_header(header)
    from native_boundary_patch import boundary_header

    header = boundary_header(header)
    # Explicit opt-in drive gear: braking cannot engage reverse gear.
    old = "int dynamics_model;"
    if header.count(old) != 1:
        raise ValueError("Pinned drive gear field mismatch")
    header = header.replace(old, old + "\n    int driversi_forward_only;")
    old = "        // Calculate new steering angle"
    if header.count(old) != 1:
        raise ValueError("Pinned forward braking insertion mismatch")
    header = header.replace(old, """        if (env->driversi_forward_only && v_new < 0.0f)
            v_new = 0.0f;
""" + old)
    # Native model executors supply a complete next state before the shared
    # collision/termination pass. No action quantization or post-collision teleport.
    header = header.replace("int driversi_forward_only;", """int driversi_forward_only;
    int driversi_external_valid[MAX_AGENTS];
    double driversi_external_state[MAX_AGENTS][10];""")
    hook = "    if (env->dynamics_model == CLASSIC) {\n        // Classic dynamics model"
    if header.count(hook) != 1:
        raise ValueError("Pinned native executor hook mismatch")
    header = header.replace(hook, """    if (env->driversi_external_valid[action_idx]) {
        double *v = env->driversi_external_state[action_idx];
        agent->x = (float)(v[0] - env->world_mean_x);
        agent->y = (float)(v[1] - env->world_mean_y);
        agent->z = v[2]; agent->heading = v[3];
        agent->heading_x = cosf(agent->heading); agent->heading_y = sinf(agent->heading);
        agent->vx = v[4]; agent->vy = v[5]; agent->vz = v[6];
        agent->a_long = v[7]; agent->a_lat = v[8]; agent->steering_angle = v[9];
        return;
    }
""" + hook)
    build_id = hashlib.sha256(
        header.encode() + (root / "src/sim/native.c").read_bytes() + repr(command).encode()
    ).hexdigest()
    command.insert(1, f'-DPD_BUILD_ID="{build_id}"')
    build_root = root / ".scratch/build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pufferdrive-", dir=build_root) as build:
        (Path(build) / "drive.h").write_text(header)
        command.insert(1, f"-I{build}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Compile beside the destination so a failed/interrupted build never
        # truncates a usable library and the completed install is atomic.
        with tempfile.NamedTemporaryFile(
            prefix=output.name + ".", suffix=".building", dir=output_dir, delete=False
        ) as temporary:
            temporary_output = Path(temporary.name)
        command[-1] = str(temporary_output)
        try:
            subprocess.run(command, check=True)
            temporary_output.replace(output)
        finally:
            temporary_output.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "extension": str(output),
                "pufferdrive_revision": revision,
                "build_id": build_id,
                "python_abi": sysconfig.get_config_var("SOABI"),
                "world_capacity": capacity,
                "host_libc": platform.libc_ver(),
            }
        )
    )


if __name__ == "__main__":
    main()
