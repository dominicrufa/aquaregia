"""test aquaregia.nets"""
from aquaregia.nets import *
from aquaregia.utils import get_normal_test_graph

# constants
DEFAULT_MLP_SCALARS = {'update_xs': 1., 'velocity_t_fn': 1., 'velocity_log_s_fn': 1.}

def test_message_fn(seed = random.PRNGKey(214),
                    message_features=3,
                    num_nodes=5,
                    node_dim=3,
                    hs_features=2,
                    edge_features=2):
    """
    this is a simple test to assert that the message mlp has the appropriate shape

    TODO : assert that message is identical with consistent inputs from nodes, edges, xs
    """
    disp_fn, shift_fn = space.free()
    mlp_e = make_mlp([3,3,message_features])
    gn_message_fn = make_default_message_fn(mlp_e, disp_fn)
    class Dummy(nn.Module):
        @nn.compact
        def __call__(self, xs, hs, edges):
            return gn_message_fn(xs, hs, edges)

    xs_seed, hs_seed, edges_seed, init_seed = random.split(seed, num=4)
    xs = random.normal(xs_seed, shape=(num_nodes, node_dim))
    hs = random.normal(hs_seed, shape=(num_nodes, hs_features))
    edges = random.normal(edges_seed, shape=(num_nodes, num_nodes, edge_features))

    net = Dummy()
    params = net.init(init_seed, xs, hs, edges)
    messages = net.apply(params, xs, hs, edges)
    assert messages.shape == (num_nodes, num_nodes, message_features)

def test_default_node_update_fn(seed = random.PRNGKey(214),
                                message_features=3,
                                num_nodes=5,
                                node_dim=3,
                                hs_features=2,
                                edge_features=2):
    """
    assert the node update mlp has an appropriate shape

    TODO : assert unity with an appropriate input
    """
    disp_fn, shift_fn = space.free()
    mlp_e = make_mlp([3,3,message_features])
    mlp_h = make_mlp([3,3,hs_features])
    gn_message_fn = make_default_message_fn(mlp_e, disp_fn)
    gn_node_fn = make_default_update_h_fn(mlp_h)
    class Dummy(nn.Module):
        @nn.compact
        def __call__(self, xs, hs, edges):
            messages = gn_message_fn(xs, hs, edges)
            updated_hs = gn_node_fn(hs, messages)
            return updated_hs

    xs_seed, hs_seed, edges_seed, init_seed = random.split(seed, num=4)
    xs = random.normal(xs_seed, shape=(num_nodes, node_dim))
    hs = random.normal(hs_seed, shape=(num_nodes, hs_features))
    edges = random.normal(edges_seed, shape=(num_nodes, num_nodes, edge_features))

    net = Dummy()
    params = net.init(init_seed, xs, hs, edges)
    new_hs = net.apply(params, xs, hs, edges)
    assert new_hs.shape == (num_nodes, hs_features)


def xy_rotation(theta : float) -> Array:
    """make a rotation in the xy plane of an angle `theta`"""
    mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                     [jnp.sin(theta), jnp.cos(theta)]])
    return mat

def z_rotation(theta : float) -> Array:
    mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0.],
                     [jnp.sin(theta), jnp.cos(theta), 0.],
                     [0., 0., 1.]])
    return mat

def matrix_multiplication(A : Array,x : Array) -> Array:
    """perform a matrix-vector multiplication"""
    return A.dot(x)

"""vmap the matrix multiplication"""
vmatrix_multiplication = vmap(matrix_multiplication, in_axes=(None, 0))

def test_simple_equivariance(seed = random.PRNGKey(226),
                             dimensions = 3,
                             hs_features=5):
    """test that velocities are rotation equivariant and positions are rotation + translation equivariant"""
    seed, rotation_seed, init_seed, translation_seed = random.split(seed, num=4)

    displacement_fn, shift_fn = space.free() # must be free space

    g = get_normal_test_graph(dimension = dimensions, hs_features=hs_features) # grab a test graph

    rnvp, params = make_RNVP_module(graph = g,
                     message1_fn_features = [3,3],
                     message2_fn_features = [4,4],
                     h_fn_features = [hs_features, hs_features],
                     mlp_x_features = [3,2,1],
                     mlp_v_features = [3,2,1],
                     displacement_fn = displacement_fn,
                     shift_fn = shift_fn,
                     seed = init_seed,
                     C = 1.,
                     use_vs_make_scalars = True,
                     make_scalars_vals = DEFAULT_MLP_SCALARS,
                     num_VRV_repeats = 2, #try for 2

                     # optional activation fns
                     message1_fn_activation = nn.swish,
                     message2_fn_activation = nn.swish,
                     h_fn_activation = nn.swish,
                     mlp_x_activation = nn.swish,
                     mlp_v_activation = nn.swish,
                    )
    rotation_matrix = z_rotation(random.uniform(rotation_seed, minval=-jnp.pi, maxval=jnp.pi))

    translation_array = random.normal(translation_seed, shape=(dimensions,))

    #rotate xs
    rotated_translated_xs = vmatrix_multiplication(rotation_matrix, g.xs) + translation_array
    #out_ds = vmetric(rotated_translated_xs[::2], rotated_translated_xs[1::2])
    #assert jnp.allclose(ds, out_ds) # the distances between the particles should be invariant about rotation

    #rotate vs
    rotated_vs = vmatrix_multiplication(rotation_matrix, g.vs)

    # push rotated/translated graph through net
    out_rotated_xs, out_rotated_vs, logdetJs1 = rnvp.apply(params, rotated_translated_xs, rotated_vs)

    # push og graph through net
    out_xs, out_vs, logdetJs2 = rnvp.apply(params, g.xs, g.vs)

    # the logdetJs should be invariant to rotations of velocities, rotations + translations of positions
    assert jnp.isclose(logdetJs1, logdetJs2)

    # rotate the pushed through graph vs
    rotated_outvs = vmatrix_multiplication(rotation_matrix, out_vs)

    # rotate and translate the pushed through graph xs
    rotated_translated_outxs = vmatrix_multiplication(rotation_matrix, out_xs) + translation_array

    # assert equivariance of x about rotation/translation
    assert jnp.allclose(rotated_translated_outxs, out_rotated_xs)

    # assert equivariance of v about rotation
    assert jnp.allclose(rotated_outvs, out_rotated_vs)

def test_logdetJ(seed = random.PRNGKey(747),
                    num_nodes = 10,
                    dimension=3,
                    hs_features=5):
    """
    test that we are computing the logdetJ properly from the RNVP
    """
    from jax import jacfwd

    displacement_fn, shift_fn = space.periodic(jnp.array([0.1, 0.1, 0.1]))
    seed, rotation_seed, init_seed, translation_seed = random.split(seed, num=4)

    # displacement_fn, shift_fn = space.free() # must be free space

    g = get_normal_test_graph(dimension = dimension,
                              num_nodes=num_nodes,
                              hs_features=hs_features) # grab a test graph

    rnvp, params = make_RNVP_module(graph = g,
                     message1_fn_features = [3,3],
                     message2_fn_features = [4,4],
                     h_fn_features = [hs_features, hs_features],
                     mlp_x_features = [3,2,1],
                     mlp_v_features = [3,2,1],
                     displacement_fn = displacement_fn,
                     shift_fn = shift_fn,
                     seed = init_seed,
                     C = 1.,
                     use_vs_make_scalars = True,
                     make_scalars_vals = DEFAULT_MLP_SCALARS,
                     num_VRV_repeats = 2, #try for 2

                     # optional activation fns
                     message1_fn_activation = nn.swish,
                     message2_fn_activation = nn.swish,
                     h_fn_activation = nn.swish,
                     mlp_x_activation = nn.swish,
                     mlp_v_activation = nn.swish,
                    )


    out_xs, out_vs, logdetJ = rnvp.apply(params, g.xs, g.vs)

    def _wrapper(phase):
        in_xs, in_vs = phase[:dimension * num_nodes].reshape((num_nodes, dimension)), phase[dimension*num_nodes:].reshape((num_nodes, dimension))
        out_xs, out_vs, _ = rnvp.apply(params, in_xs, in_vs)
        return jnp.concatenate([out_xs.flatten(), out_vs.flatten()])

    J = jacfwd(_wrapper)(jnp.concatenate([g.xs.flatten(), g.vs.flatten()]))
    assert jnp.isclose(logdetJ, jnp.log(jnp.linalg.det(J)))
