/* DriveRSI CPython adapter. Simulation algorithms remain in unmodified PufferDrive
 * (Emerge-Lab/PufferDrive, MIT); see src/config/backends/pufferdrive.toml.
 * No rendering, process-global random sampling, or implicit episode reset is used.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <limits.h>
#ifndef DRIVERSI_WORLD_CAPACITY
#error "Build with the explicit platform world capacity"
#endif
#include "drive.h"

#define CAPSULE "driversi.pufferdrive.world.v1"
#define OBS_SIZE (EGO_FEATURES_JERK + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)

typedef struct {
    Drive env;
    int closed;
} Handle;

static Handle *get_handle(PyObject *capsule) {
    Handle *handle = PyCapsule_GetPointer(capsule, CAPSULE);
    if (handle && handle->closed) {
        PyErr_SetString(PyExc_RuntimeError, "World is closed");
        return NULL;
    }
    return handle;
}

static void release(Handle *handle) {
    if (!handle->closed) {
        free_allocated(&handle->env);
        free(handle->env.map_name);
        handle->closed = 1;
    }
}

static void destroy(PyObject *capsule) {
    Handle *handle = PyCapsule_GetPointer(capsule, CAPSULE);
    if (handle) { release(handle); free(handle); }
}

static void sync_replay_velocity(Drive *env) {
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        Entity *e = &env->entities[env->expert_static_agent_indices[i]];
        int t = env->timestep;
        if (t >= 0 && t < e->array_size && e->traj_valid[t]) {
            e->vx = e->traj_vx[t]; e->vy = e->traj_vy[t]; e->vz = e->traj_vz[t];
        } else { e->vx = e->vy = e->vz = 0.0f; }
    }
    compute_observations(env);
}

static PyObject *create(PyObject *self, PyObject *args) {
    const char *path; int forward_only=0, corner_boundary=0;
    if (!PyArg_ParseTuple(args, "s|pp", &path, &forward_only, &corner_boundary)) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
    char scenario_id[16]; int scene_counts[4];
    int valid_header = fread(scenario_id, 1, sizeof(scenario_id), file) == sizeof(scenario_id)
        && fread(scene_counts, sizeof(int), 4, file) == 4;
    fclose(file);
    if (!valid_header || scene_counts[2] < 1 || scene_counts[2] > DRIVERSI_WORLD_CAPACITY) {
        PyErr_SetString(PyExc_ValueError, "Scene object count exceeds native world capacity or invalid header");
        return NULL;
    }
    Handle *handle = calloc(1, sizeof(Handle));
    if (!handle) return PyErr_NoMemory();
    Drive *env = &handle->env;
    env->map_name = strdup(path);
    env->dt = 0.1f;
    env->episode_length = INT_MAX; /* Python owns truncation and final observation. */
    env->dynamics_model = JERK;
    env->driversi_forward_only = forward_only;
    env->driversi_corner_boundary = corner_boundary;
    env->action_type = 0;
    env->num_agents = MAX_AGENTS;
    env->max_controlled_agents = MAX_AGENTS;
    env->control_mode = CONTROL_VEHICLES;
    env->init_mode = INIT_ALL_VALID;
    env->goal_behavior = GOAL_STOP;
    env->collision_behavior = STOP_AGENT;
    env->offroad_behavior = STOP_AGENT;
    env->goal_radius = 2.0f;
    env->goal_speed = 100.0f;
    env->reward_goal = 1.0f;
    env->reward_vehicle_collision = -1.0f;
    env->reward_offroad_collision = -1.0f;
    allocate(env);
    if (env->active_agent_count < 1 || env->active_agent_count > MAX_AGENTS) {
        release(handle); free(handle);
        PyErr_SetString(PyExc_ValueError, "Scene has no eligible controlled vehicles");
        return NULL;
    }
    c_reset(env);
    sync_replay_velocity(env);
    return PyCapsule_New(handle, CAPSULE, destroy);
}

static PyObject *close_world(PyObject *self, PyObject *capsule) {
    Handle *handle = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!handle) return NULL;
    release(handle);
    Py_RETURN_NONE;
}

/* Optional next-state rows: [passive index,x,y,z,yaw,vx,vy,vz].
 * Validate the entire batch before mutation. Overrides exist only during c_step;
 * source trajectories remain immutable across snapshots, reset and branches. */
static PyObject *step_world(PyObject *self, PyObject *args) {
    PyObject *capsule, *input, *background = Py_None, *membership = Py_None, *policy = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OOO", &capsule, &input, &background, &membership, &policy)) return NULL;
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    Drive *env = &handle->env;
    PyArrayObject *actions = (PyArrayObject *)PyArray_FROM_OTF(input, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!actions) return NULL;
    PyArrayObject *targets = NULL, *members = NULL, *policy_rows = NULL;
    if (PyArray_NDIM(actions) != 1 || PyArray_SIZE(actions) != env->active_agent_count) {
        PyErr_SetString(PyExc_ValueError, "Provide exactly one action per controlled actor"); goto fail;
    }
    int *values = PyArray_DATA(actions);
    for (int i = 0; i < env->active_agent_count; i++) {
        if (values[i] < 0 || values[i] >= 12) {
            PyErr_SetString(PyExc_ValueError, "Jerk action must be in [0, 11]"); goto fail;
        }
    }
    int count=0, indices[DRIVERSI_WORLD_CAPACITY], tick=env->timestep+1;
    double *rows=NULL;
    int previous_valid[DRIVERSI_WORLD_CAPACITY];
    npy_bool *mask=NULL;
    if (membership != Py_None) {
        members=(PyArrayObject *)PyArray_FROM_OTF(membership,NPY_BOOL,NPY_ARRAY_IN_ARRAY);
        if (!members) goto fail;
        if (PyArray_NDIM(members)!=1 || PyArray_SIZE(members)!=env->num_objects) {
            PyErr_SetString(PyExc_ValueError,"Membership must cover all objects");goto fail;
        }
        mask=PyArray_DATA(members);
        for(int i=0;i<env->num_objects;i++) {
            Entity *e=&env->entities[i];
            if(tick<0 || tick>=e->array_size) {
                PyErr_SetString(PyExc_ValueError,"Membership outside world horizon");goto fail;
            }
        }
        for(int i=0;i<env->active_agent_count;i++)
            if(!mask[env->active_agent_indices[i]]) {
                PyErr_SetString(PyExc_ValueError,"Cannot remove a policy actor");goto fail;
            }
    }
    float saved[DRIVERSI_WORLD_CAPACITY][7], old_v[DRIVERSI_WORLD_CAPACITY][2];
    if (background != Py_None) {
        targets=(PyArrayObject *)PyArray_FROM_OTF(background,NPY_FLOAT64,NPY_ARRAY_IN_ARRAY);
        if (!targets) goto fail;
        if(PyArray_NDIM(targets)!=2 || PyArray_DIM(targets,1)!=8 || PyArray_DIM(targets,0)>DRIVERSI_WORLD_CAPACITY) {
            PyErr_SetString(PyExc_ValueError,"Background states exceed native world capacity or do not have8 columns");goto fail;
        }
        count=PyArray_DIM(targets,0);rows=PyArray_DATA(targets);
        for(int r=0;r<count;r++) {
            double *row=rows+r*8;
            for(int c=0;c<8;c++) if(!isfinite(row[c]) || fabs(row[c])>1e7) {
                PyErr_SetString(PyExc_ValueError,"Background state must be finite and bounded");goto fail;
            }
            int k=(int)row[0], passive=0;
            if(row[0]!=k || k<0 || k>=env->num_objects) {
                PyErr_SetString(PyExc_ValueError,"Invalid background actor index");goto fail;
            }
            for(int j=0;j<env->expert_static_agent_count;j++)
                if(env->expert_static_agent_indices[j]==k) passive=1;
            for(int j=0;j<r;j++) if(indices[j]==k) passive=0;
            Entity *e=&env->entities[k];
            if(!passive || tick<0 || tick>=e->array_size || !(mask ? mask[k] : e->traj_valid[tick])) {
                PyErr_SetString(PyExc_ValueError,"Background targets must be unique present passive actors");goto fail;
            }
            indices[r]=k;
        }
    }
    int policy_count=0, policy_slots[MAX_AGENTS]; double *policy_data=NULL;
    if(policy != Py_None) {
        policy_rows=(PyArrayObject *)PyArray_FROM_OTF(policy,NPY_FLOAT64,NPY_ARRAY_IN_ARRAY);
        if(!policy_rows) goto fail;
        if(PyArray_NDIM(policy_rows)!=2 || PyArray_DIM(policy_rows,1)!=11 || PyArray_DIM(policy_rows,0)>MAX_AGENTS) {
            PyErr_SetString(PyExc_ValueError,"Native policy states require [index,xyz,yaw,vxyz,a_long,a_lat,steer]");goto fail;
        }
        policy_count=PyArray_DIM(policy_rows,0);policy_data=PyArray_DATA(policy_rows);
        for(int r=0;r<policy_count;r++) {
            double *v=policy_data+r*11;int slot=-1;
            for(int c=0;c<11;c++) if(!isfinite(v[c]) || fabs(v[c])>1e7) {
                PyErr_SetString(PyExc_ValueError,"Nonfinite or unbounded native policy state");goto fail;
            }
            for(int j=0;j<env->active_agent_count;j++) if(v[0]==env->active_agent_indices[j]) slot=j;
            if(slot<0 || env->entities[env->active_agent_indices[slot]].stopped) {
                PyErr_SetString(PyExc_ValueError,"Native policy state must target a live controlled actor");goto fail;
            }
            for(int j=0;j<r;j++) if(policy_slots[j]==slot) {
                PyErr_SetString(PyExc_ValueError,"Duplicate native policy state");goto fail;
            }
            policy_slots[r]=slot;
        }
    }
    for(int r=0;r<policy_count;r++) {
        int slot=policy_slots[r];env->driversi_external_valid[slot]=1;
        memcpy(env->driversi_external_state[slot],policy_data+r*11+1,10*sizeof(double));
    }
    for(int i=0;i<env->active_agent_count;i++) ((int *)env->actions)[i]=values[i];
    for(int r=0;r<count;r++) {
        Entity *e=&env->entities[indices[r]];double *row=rows+r*8;
        float *arrays[]={e->traj_x,e->traj_y,e->traj_z,e->traj_heading,e->traj_vx,e->traj_vy,e->traj_vz};
        for(int c=0;c<7;c++) {saved[r][c]=arrays[c][tick];arrays[c][tick]=(float)row[c+1];}
        e->traj_x[tick]-=env->world_mean_x;e->traj_y[tick]-=env->world_mean_y;
        old_v[r][0]=e->vx;old_v[r][1]=e->vy;
    }
    if(mask) for(int i=0;i<env->num_objects;i++) {
        Entity *e=&env->entities[i];previous_valid[i]=e->traj_valid[tick];
        e->traj_valid[tick]=mask[i];
    }
    c_step(env);
    memset(env->driversi_external_valid,0,sizeof(env->driversi_external_valid));
    memset(env->driversi_external_state,0,sizeof(env->driversi_external_state));
    sync_replay_velocity(env);
    for(int r=0;r<count;r++) {
        Entity *e=&env->entities[indices[r]];
        float ax=(e->vx-old_v[r][0])/env->dt, ay=(e->vy-old_v[r][1])/env->dt;
        e->a_long=ax*e->heading_x+ay*e->heading_y;
        e->a_lat=-ax*e->heading_y+ay*e->heading_x;
        float *arrays[]={e->traj_x,e->traj_y,e->traj_z,e->traj_heading,e->traj_vx,e->traj_vy,e->traj_vz};
        for(int c=0;c<7;c++) arrays[c][tick]=saved[r][c];
    }
    if(mask) for(int i=0;i<env->num_objects;i++)
        env->entities[i].traj_valid[tick]=previous_valid[i];
    Py_DECREF(actions);Py_XDECREF(targets);Py_XDECREF(members);Py_XDECREF(policy_rows);Py_RETURN_NONE;
fail:
    Py_DECREF(actions);Py_XDECREF(targets);Py_XDECREF(members);Py_XDECREF(policy_rows);return NULL;
}

static PyObject *motion_world(PyObject *self, PyObject *capsule) {
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    Drive *env = &handle->env;
    npy_intp shape[] = {env->num_objects, 3};
    PyArrayObject *motion = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_FLOAT64);
    if (!motion) return NULL;
    double *out = PyArray_DATA(motion);
    for (int i = 0; i < env->num_objects; i++) {
        Entity *e = &env->entities[i];
        out[3*i] = e->a_long;
        out[3*i+1] = e->a_lat;
        out[3*i+2] = e->steering_angle;
    }
    return (PyObject *)motion;
}

static PyObject *read_world(PyObject *self, PyObject *capsule) {
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    Drive *env = &handle->env;
    npy_intp obs_shape[] = {env->active_agent_count, OBS_SIZE};
    npy_intp actor_shape[] = {env->num_objects, 13};
    npy_intp agent_shape[] = {env->active_agent_count};
    PyArrayObject *obs = (PyArrayObject *)PyArray_SimpleNew(2, obs_shape, NPY_FLOAT32);
    PyArrayObject *actors = (PyArrayObject *)PyArray_SimpleNew(2, actor_shape, NPY_FLOAT64);
    PyArrayObject *indices = (PyArrayObject *)PyArray_SimpleNew(1, agent_shape, NPY_INT32);
    PyArrayObject *rewards = (PyArrayObject *)PyArray_SimpleNew(1, agent_shape, NPY_FLOAT32);
    if (!obs || !actors || !indices || !rewards) {
        Py_XDECREF(obs); Py_XDECREF(actors); Py_XDECREF(indices); Py_XDECREF(rewards);
        return NULL;
    }
    memcpy(PyArray_DATA(obs), env->observations, PyArray_NBYTES(obs));
    memcpy(PyArray_DATA(indices), env->active_agent_indices, PyArray_NBYTES(indices));
    memcpy(PyArray_DATA(rewards), env->rewards, PyArray_NBYTES(rewards));
    double *out = PyArray_DATA(actors);
    for (int i = 0; i < env->num_objects; i++) {
        Entity *e = &env->entities[i];
        double values[] = {
            e->x + env->world_mean_x, e->y + env->world_mean_y, e->z, e->heading,
            e->vx, e->vy, e->vz,
            e->goal_position_x + env->world_mean_x, e->goal_position_y + env->world_mean_y,
            e->stopped, e->collision_state == VEHICLE_COLLISION,
            e->metrics_array[OFFROAD_IDX], e->metrics_array[REACHED_GOAL_IDX],
        };
        memcpy(out + i * 13, values, sizeof(values));
    }
    return Py_BuildValue("NNNNi", obs, actors, indices, rewards, env->timestep);
}

/* Snapshot only the mutable scalar tails and owned buffers, never C pointers.
 * Map, trajectory arrays, actor index tables, and grid caches are immutable in
 * this fixed-population / fixed-goal adapter. Python binds bytes to that scene.
 */
static size_t snapshot_size(Drive *env) {
    size_t scalar_size = sizeof(Entity) - offsetof(Entity, width);
    size_t n = env->active_agent_count;
    return 4 * sizeof(int) + env->num_objects * scalar_size + sizeof(Log)
        + n * (sizeof(Log) + (OBS_SIZE + 3) * sizeof(float) + 2);
}

static void transfer_snapshot(Drive *env, char *data, int restore) {
    size_t offset = 0;
    int header[] = {1, env->num_objects, env->active_agent_count, env->timestep};
    if (restore) { memcpy(header, data, sizeof(header)); env->timestep = header[3]; }
    else memcpy(data, header, sizeof(header));
    offset += sizeof(header);
    #define TRANSFER(ptr, size) do { \
        size_t bytes = (size); \
        if (restore) memcpy((ptr), data + offset, bytes); \
        else memcpy(data + offset, (ptr), bytes); \
        offset += bytes; \
    } while (0)
    for (int i = 0; i < env->num_objects; i++) {
        TRANSFER(&env->entities[i].width, sizeof(Entity) - offsetof(Entity, width));
    }
    size_t n = env->active_agent_count;
    TRANSFER(&env->log, sizeof(Log));
    TRANSFER(env->logs, n * sizeof(Log));
    TRANSFER(env->observations, n * OBS_SIZE * sizeof(float));
    TRANSFER(env->actions, n * 2 * sizeof(float));
    TRANSFER(env->rewards, n * sizeof(float));
    TRANSFER(env->terminals, n);
    TRANSFER(env->truncations, n);
    #undef TRANSFER
}

static PyObject *snapshot_world(PyObject *self, PyObject *capsule) {
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    PyObject *blob = PyBytes_FromStringAndSize(NULL, snapshot_size(&handle->env));
    if (!blob) return NULL;
    transfer_snapshot(&handle->env, PyBytes_AS_STRING(blob), 0);
    return blob;
}

static PyObject *restore_world(PyObject *self, PyObject *args) {
    PyObject *capsule; const char *data; Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "Oy#", &capsule, &data, &size)) return NULL;
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    Drive *env = &handle->env;
    int header[4];
    if ((size_t)size != snapshot_size(env)) {
        PyErr_SetString(PyExc_ValueError, "Snapshot byte length mismatch"); return NULL;
    }
    memcpy(header, data, sizeof(header));
    if (header[0] != 1 || header[1] != env->num_objects || header[2] != env->active_agent_count
        || header[3] < 0 || header[3] >= env->entities[env->active_agent_indices[0]].array_size) {
        PyErr_SetString(PyExc_ValueError, "Snapshot ABI or timestep mismatch"); return NULL;
    }
    transfer_snapshot(env, (char *)data, 1);
    Py_RETURN_NONE;
}

static PyObject *presence_world(PyObject *self, PyObject *capsule) {
    Handle *h=get_handle(capsule);if(!h)return NULL;
    npy_intp shape[]={h->env.num_objects};
    PyArrayObject *a=(PyArrayObject *)PyArray_SimpleNew(1,shape,NPY_BOOL);
    if(!a)return NULL;
    npy_bool *p=PyArray_DATA(a);
    for(int i=0;i<h->env.num_objects;i++) p[i]=h->env.entities[i].x!=INVALID_POSITION;
    return (PyObject *)a;
}

static PyObject *contacts_world(PyObject *self, PyObject *capsule) {
    Handle *handle = get_handle(capsule);
    if (!handle) return NULL;
    Drive *env=&handle->env;
    PyObject *pairs=PyList_New(0);
    if (!pairs) return NULL;
    for(int i=0;i<env->num_objects;i++) {
        Entity *a=&env->entities[i];
        if(a->x==INVALID_POSITION) continue;
        for(int j=i+1;j<env->num_objects;j++) {
            Entity *b=&env->entities[j];
            if(b->x==INVALID_POSITION) continue;
            if(check_aabb_collision(a,b)) {
                PyObject *pair=Py_BuildValue("ii",i,j);
                if(!pair || PyList_Append(pairs,pair)<0) {Py_XDECREF(pair);Py_DECREF(pairs);return NULL;}
                Py_DECREF(pair);
            }
        }
    }
    return pairs;
}

/* Model-native initial kinematics only; no position/lifecycle/collision override. */
static PyObject *initialize_motion(PyObject *self, PyObject *args) {
    PyObject *capsule,*input;
    if(!PyArg_ParseTuple(args,"OO",&capsule,&input)) return NULL;
    Handle *h=get_handle(capsule); if(!h) return NULL;
    Drive *env=&h->env;
    if(env->timestep!=0) {PyErr_SetString(PyExc_ValueError,"Initial motion only at tick zero");return NULL;}
    PyArrayObject *a=(PyArrayObject *)PyArray_FROM_OTF(input,NPY_FLOAT64,NPY_ARRAY_IN_ARRAY);
    if(!a) return NULL;
    if(PyArray_NDIM(a)!=2 || PyArray_DIM(a,0)!=env->active_agent_count || PyArray_DIM(a,1)!=3) {
        Py_DECREF(a);PyErr_SetString(PyExc_ValueError,"Initial motion must cover controlled actors");return NULL;
    }
    double *v=PyArray_DATA(a);
    for(int i=0;i<env->active_agent_count*3;i++) if(!isfinite(v[i]) || fabs(v[i])>1e3) {
        Py_DECREF(a);PyErr_SetString(PyExc_ValueError,"Invalid initial motion");return NULL;
    }
    for(int i=0;i<env->active_agent_count;i++) {
        Entity *e=&env->entities[env->active_agent_indices[i]];
        e->a_long=v[3*i];e->a_lat=v[3*i+1];e->steering_angle=v[3*i+2];
    }
    compute_observations(env);Py_DECREF(a);Py_RETURN_NONE;
}

static PyObject *boundary_contacts(PyObject *self, PyObject *capsule) {
    Handle *h=get_handle(capsule); if(!h) return NULL;
    npy_intp shape[]={h->env.num_objects};
    PyArrayObject *a=(PyArrayObject *)PyArray_SimpleNew(1,shape,NPY_BOOL);
    if(!a) return NULL;
    npy_bool *v=PyArray_DATA(a);
    for(int i=0;i<h->env.num_objects;i++)
        v[i]=h->env.entities[i].driversi_boundary_contact!=0;
    return (PyObject *)a;
}

static PyMethodDef methods[] = {
    {"boundary_contacts", boundary_contacts, METH_O, "Raw perimeter road-edge contact, separate from corner validity."},
    {"initialize_motion", initialize_motion, METH_VARARGS, "Initialize model-native kinematics at tick zero."},
    {"create", create, METH_VARARGS, "Create a world from a validated scene binary."},
    {"step", step_world, METH_VARARGS, "Apply a joint action and advance the world."},
    {"read", read_world, METH_O, "Copy current observations and actor state."},
    {"presence", presence_world, METH_O, "Actual native physical membership."},
    {"contacts", contacts_world, METH_O, "Current present-object box contacts, including replay objects."},
    {"motion", motion_world, METH_O, "Copy longitudinal/lateral acceleration and steering."},
    {"close", close_world, METH_O, "Close an owned world idempotently."},
    {"snapshot", snapshot_world, METH_O, "Export mutable state without pointers."},
    {"restore", restore_world, METH_VARARGS, "Restore same-scene mutable state."},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    PyObject *result = PyModule_Create(&module);
    if (result) {
        PyModule_AddStringConstant(result, "backend_revision", PD_REVISION);
        PyModule_AddStringConstant(result, "build_id", PD_BUILD_ID);
        PyModule_AddIntConstant(result, "observation_size", OBS_SIZE);
        PyModule_AddIntConstant(result, "background_state_version", 1);
        PyModule_AddIntConstant(result, "max_world_actors", DRIVERSI_WORLD_CAPACITY);
        PyModule_AddIntConstant(result, "variable_horizon", 1);
        PyModule_AddIntConstant(result, "continuity_version", 1);
        PyModule_AddIntConstant(result, "native_policy_state_version", 1);
        PyModule_AddIntConstant(result, "corner_boundary_version", 1);
    }
    return result;
}
