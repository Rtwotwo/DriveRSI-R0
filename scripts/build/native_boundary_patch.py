"""Build-time boundary profile; upstream sources and official scores stay unchanged."""


def boundary_header(header):
    def replace(old, new):
        nonlocal header
        if header.count(old) != 1:
            raise ValueError("Pinned boundary hook mismatch: " + old[:60])
        header = header.replace(old, new)

    replace("int dynamics_model;", "int dynamics_model;\n    int driversi_corner_boundary;")
    replace("int collision_state;", "int collision_state;\n    int driversi_boundary_contact;")
    replace(
        "void compute_agent_metrics(Drive *env, int agent_idx) {",
        r"""
/* Even-odd coverage of a validated union boundary, including holes and islands.
 * No expansion/simplification/tolerance. Boundary points themselves are covered.
 */
static int driversi_point_in_drivable(Drive *env, double x, double y) {
    int inside = 0;
    for (int i=env->num_objects; i<env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type != ROAD_EDGE) continue;
        for (int j=0; j+1<e->array_size; j++) {
            double ax=e->traj_x[j], ay=e->traj_y[j];
            double bx=e->traj_x[j+1], by=e->traj_y[j+1];
            double cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax);
            if (cross==0 && x>=fmin(ax,bx) && x<=fmax(ax,bx)
                         && y>=fmin(ay,by) && y<=fmax(ay,by)) return 1;
            if ((ay>y)!=(by>y) && x<(bx-ax)*(y-ay)/(by-ay)+ax) inside=!inside;
        }
    }
    return inside;
}

void compute_agent_metrics(Drive *env, int agent_idx) {""",
    )
    replace(
        "    reset_agent_metrics(env, agent_idx);",
        "    reset_agent_metrics(env, agent_idx);\n    agent->driversi_boundary_contact = 0;",
    )
    replace(
        "                    collided = OFFROAD;\n                    break;",
        "                    agent->driversi_boundary_contact = 1;\n"
        "                    if (!env->driversi_corner_boundary) collided = OFFROAD;\n"
        "                    break;",
    )
    replace(
        "    // Check for vehicle collisions (skip for pedestrians)",
        """    if (env->driversi_corner_boundary && agent->type != PEDESTRIAN) {
        for (int k=0; k<4; k++)
            if (!driversi_point_in_drivable(env, corners[k][0], corners[k][1])) {
                collided = OFFROAD;
                break;
            }
    }

    // Check for vehicle collisions (skip for pedestrians)""",
    )
    return header
