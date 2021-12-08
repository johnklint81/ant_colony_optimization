import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

N = 40  # number of vertices
max_distance = 20
max_steps = 80
# ratio = 0.5  # total number of connections in the graph, max is N(N-1)
number_of_ants = 20
alpha = 0.8
beta = 1.0
rho = 0.5
max_pheromone_per_ant = 1.0
number_of_iterations = 20


def create_matrices(_N, _max_distance):
    # Location of vertice coordinates
    _vertice_array_x = np.random.rand(_N) * _max_distance
    _vertice_array_y = np.random.rand(_N) * _max_distance
    _vertices = np.array([_vertice_array_x, _vertice_array_y]).T

    # Connection matrix
    tri = Delaunay(_vertices)
    number_of_triangles = len(tri.simplices)
    _connection_matrix = np.zeros([_N, _N])
    for i in range(number_of_triangles):
        for j in range(3):
            for k in range(3):
                if j != k:
                    _vertex_1 = tri.simplices[i, j]
                    _vertex_2 = tri.simplices[i, k]
                    _connection_matrix[_vertex_1, _vertex_2] = 1
    _connection_matrix += _connection_matrix.T / 2
    _connection_matrix = _connection_matrix.astype(int)

    # Distance matrix
    _distance_matrix = np.zeros([_N, _N])

    for i in range(_N):
        for j in range(_N):
            _point1 = np.array([_vertice_array_x[i], _vertice_array_y[i]])
            _point2 = np.array([_vertice_array_x[j], _vertice_array_y[j]])
            _distance_matrix[i, j] = get_distance(_point1, _point2)
    _distance_matrix = np.where(_connection_matrix == 1, _distance_matrix, np.inf)
    np.fill_diagonal(_distance_matrix, np.inf)
    _distance_matrix = (_distance_matrix + _distance_matrix.T) / 2

    # Weight matrix
    _weight_matrix = 1 / _distance_matrix
    np.fill_diagonal(_weight_matrix, 0)

    # Pheromone matrix
    _pheromone_matrix = np.ones([N, N])
    _pheromone_matrix = np.where(_connection_matrix == 1, _pheromone_matrix, 0)
    return _connection_matrix, _vertice_array_x, _vertice_array_y, _distance_matrix, _weight_matrix, _pheromone_matrix


def get_distance(_point1, _point2):
    _distance = np.linalg.norm(_point2 - _point1)
    return _distance


def create_path(_connection_matrix, _start_location, _N):
    _path = np.zeros(_N + 1, dtype=int)
    _path[0] = _start_location
    for i in range(_N):
        _candidates = np.argwhere(_connection_matrix[:, _path[i]] == 1).flatten()
        _path[i + 1] = int(np.random.choice(_candidates))
    return _path


def path_length(_path, _distance_matrix):
    _length_of_path = 0
    for i in range(len(_path) - 1):
        _length_of_path += _distance_matrix[_path[i + 1], _path[i]]
    return _length_of_path


def simplify_path_once(_path):
    _duplicate_element_indices = get_duplicates(_path)
    _number_of_duplicate_elements = len(_duplicate_element_indices)
    # print(f"Duplicates on indices: {_duplicate_element_indices}")
    # print("------------")
    # print(f"path: {_path}")
    _path_candidates = []
    _duplicate_indices = []
    for i in range(_number_of_duplicate_elements):
        _duplicates = np.array(_duplicate_element_indices[i])
        _number_of_duplicates = len(_duplicates)
        _temp_path = np.copy(_path)
        _temp_path = np.delete(_temp_path, np.arange(_duplicates[0], _duplicates[-1], 1))
        _temp_duplicate_element_indices = get_duplicates(_temp_path)
        # print(f"Path to be corrected: {_path}")
        # print(f"Correcting duplicate indices: {_duplicates}")
        # print(f"Corrected path: {_temp_path}")
        # print(f"Updated duplicate element indices: {_temp_duplicate_element_indices}")
        # print("------------")
        _path_candidates.append(_temp_path)
        _duplicate_indices.append(_temp_duplicate_element_indices)
    return _path_candidates, _duplicate_indices


def simplify_entire_path(_path):
    _duplicates = True
    _simplified_paths = []
    _path_candidates = [_path]
    _path_candidates_list = [_path]
    while _duplicates:
        _number_of_temp_paths = len(_path_candidates_list)
        for i in range(_number_of_temp_paths):
            _temp_path_candidates, _duplicate_indices = simplify_path_once(_path_candidates_list[i])
            # _temp_path_candidates = [x for x in _temp_path_candidates if x != []]
            # _duplicate_indices = [x for x in _duplicate_indices if x != []]
            _temp_simplified_paths, _temp_path_candidates = store_candidates(_temp_path_candidates, _duplicate_indices)
            _number_of_simplified_paths = len(_temp_simplified_paths)
            # _simplified_paths = [_temp_simplified_paths[j] for j in range(_number_of_simplified_paths)]
            for j in range(_number_of_simplified_paths):
                _simplified_paths.append(_temp_simplified_paths[j])
            _number_of_path_candidates = len(_temp_path_candidates)
            for j in range(_number_of_path_candidates):
                _path_candidates.append(_temp_path_candidates[j])
            # _path_candidates = [_temp_path_candidates[j] for j in range(_number_of_path_candidates)]
            if len(_duplicate_indices) == 0:
                _duplicates = False
                _simplified_paths = _path_candidates
                return _simplified_paths
        _path_candidates_list = np.copy(_path_candidates)
        _seen = set()
        _path_candidates_list = [item for item in _path_candidates_list if
                                 not (tuple(item) in _seen or _seen.add(tuple(item)))]
        if len(_path_candidates_list) == 0:
            _duplicates = False
        _path_candidates = []
    return _simplified_paths


def simplify_and_return_shortest_path(_path):
    _duplicates = True
    _simplified_paths = []
    _path_candidates = [_path]
    _path_candidates_list = [_path]
    while _duplicates:
        _number_of_temp_paths = len(_path_candidates_list)
        for i in range(_number_of_temp_paths):
            _temp_path_candidates, _duplicate_indices = simplify_path_once(_path_candidates_list[i])
            _temp_simplified_paths, _temp_path_candidates = store_candidates(_temp_path_candidates, _duplicate_indices)
            _number_of_simplified_paths = len(_temp_simplified_paths)
            # _simplified_paths = [_temp_simplified_paths[j] for j in range(_number_of_simplified_paths)]
            for j in range(_number_of_simplified_paths):
                _simplified_paths.append(_temp_simplified_paths[j])
            _number_of_path_candidates = len(_temp_path_candidates)
            for j in range(_number_of_path_candidates):
                _path_candidates.append(_temp_path_candidates[j])
            # _path_candidates = [_temp_path_candidates[j] for j in range(_number_of_path_candidates)]
        _path_candidates_list = np.copy(_path_candidates)
        _seen = set()
        _path_candidates_list = [item for item in _path_candidates_list if
                                 not (tuple(item) in _seen or _seen.add(tuple(item)))]
        if len(_path_candidates_list) == 0:
            _duplicates = False
        _path_candidates = []

    if not _simplified_paths:
        _simplified_paths = _path
    return _simplified_paths


def store_candidates(_path_candidates, _duplicate_indices):
    _simplified_paths = []
    _path_candidates_list = []
    _number_of_duplicates = len(_duplicate_indices)

    if _number_of_duplicates == 0:
        _simplified_paths.append(_path_candidates)
        return _simplified_paths, _path_candidates_list
    else:
        for j in range(_number_of_duplicates):
            if not _duplicate_indices[j]:
                _simplified_paths.append(_path_candidates[j])
            else:
                _path_candidates_list.append(_path_candidates[j])
    return _simplified_paths, _path_candidates_list


def get_duplicates(_path):
    _idx_sort = np.argsort(_path)
    _sorted_paths = _path[_idx_sort]
    _values, _idx_start, count = np.unique(_sorted_paths, return_counts=True, return_index=True)
    _values = _values[count > 1]  # the values that have duplicates
    _indices_filter = np.split(_idx_sort, _idx_start[1:])  # indices of elements, length > 1 means duplicates
    _number_of_different_elements = len(_indices_filter)
    _duplicate_element_indices = []
    for i in range(_number_of_different_elements):
        if len(_indices_filter[i]) > 1:
            _duplicate_element_indices.append(np.sort(_indices_filter[i]))
    return _duplicate_element_indices


def ant_simulation(_number_of_ants, _start_location, _end_location, _number_of_steps, _alpha, _beta, _N,
                   _connection_matrix, _weight_matrix, _pheromone_matrix):
    _path_probability_matrix = path_probabilities(_weight_matrix, _pheromone_matrix, _alpha, _beta, _N)
    _path = np.zeros([_number_of_steps + 1, _number_of_ants], dtype=int)
    _destination_reached = []
    _path[0, :] = _start_location
    _searching = np.ones(_number_of_ants, dtype=bool)
    for step in range(_number_of_steps):
        for ant in range(_number_of_ants):
            if _searching[ant]:
                _selected_path = roulette_wheel_selection(_path_probability_matrix[:, _path[step, ant]])
                _path[step + 1, ant] = _selected_path
                if _path[step + 1, ant] == _end_location:
                    _destination_reached.append(_path[:step + 2, ant])
                    _searching[ant] = False
    return _destination_reached


def roulette_wheel_selection(_list_of_probabilities):
    _selected_path = np.random.choice(len(_list_of_probabilities), p=_list_of_probabilities)
    return _selected_path


def simplify_all_paths(_destination_reached):
    _number_of_successful_ants = len(_destination_reached)
    _shortest_length = np.ones(_number_of_successful_ants) * np.inf
    _shortest_paths = np.zeros(_number_of_successful_ants, dtype=object)
    for _successful_ant in range(_number_of_successful_ants):
        _simplified_unique_paths = simplify_entire_path(_destination_reached[_successful_ant])
        _number_of_simplified_paths = len(_simplified_unique_paths)
        for _path in range(_number_of_simplified_paths):
            _length_of_simplified_path = path_length(_simplified_unique_paths[_path], distance_matrix)
            if _length_of_simplified_path < _shortest_length[_successful_ant]:
                _shortest_length[_successful_ant] = _length_of_simplified_path
                _shortest_paths[_successful_ant] = _simplified_unique_paths[_path]
    return _shortest_paths, _shortest_length


def path_probabilities(_weight_matrix, _pheromone_matrix, _alpha, _beta, _N):
    _path_probability_matrix = np.zeros([_N, _N])
    for i in range(_N):
        for j in range(_N):
            _nominator = _pheromone_matrix[i, j] ** _alpha * _weight_matrix[i, j] ** _beta
            _denominator = np.sum(_pheromone_matrix[j, :] ** _alpha * _weight_matrix[j, :] ** _beta)
            _probability = _nominator / _denominator
            _path_probability_matrix[i, j] = _probability
    return _path_probability_matrix


def pheromone_change(_shortest_length, _shortest_path, _max_pheromone_per_ant, _N):
    _delta_pheromone = np.zeros([_N, _N])
    _number_of_shortest_paths = len(_shortest_path)
    for i in range(_number_of_shortest_paths):
        _number_of_steps = len(_shortest_path[i])
        _pheromone_change = _max_pheromone_per_ant / _shortest_length[i]
        _current_shortest_path = _shortest_path[i]
        for j in range(_number_of_steps - 1):
            _start_vertex = _current_shortest_path[j]
            _end_vertex = _current_shortest_path[j + 1]
            _delta_pheromone[_start_vertex, _end_vertex] += _pheromone_change
            _delta_pheromone[_end_vertex, _start_vertex] += _pheromone_change
    return _delta_pheromone


def pheromone_update(_pheromone_matrix, _N, _shortest_lengths, _shortest_paths, _alpha, _beta, _rho,
                     _max_pheromone_per_ant):
    _pheromone_matrix *= (1 - _rho)
    _delta_pheromone = pheromone_change(_shortest_lengths, _shortest_paths, _max_pheromone_per_ant, _N)
    _pheromone_matrix += _delta_pheromone
    return _pheromone_matrix


connection_matrix, vertice_array_x, vertice_array_y, distance_matrix, weight_matrix, pheromone_matrix = \
    create_matrices(N, max_distance)

index_range = np.arange(0, N, 1)
start_location = np.random.choice(index_range)  # index of start location vertex
index_range = np.delete(index_range, start_location)
end_location = np.random.choice(index_range)

first_pheromone_matrix = np.copy(pheromone_matrix)
shortest_length = np.inf

for i in range(number_of_iterations):
    print(f"Current iteration: {i + 1}/{number_of_iterations}")
    destination_reached = ant_simulation(number_of_ants, start_location, end_location, max_steps, alpha, beta, N,
                                         connection_matrix, weight_matrix, pheromone_matrix)
    shortest_paths, shortest_lengths = simplify_all_paths(destination_reached)
    pheromone_matrix = pheromone_update(pheromone_matrix, N, shortest_lengths, shortest_paths, alpha, beta, rho,
                                        max_pheromone_per_ant)
    shortest_single_length = np.min(shortest_lengths)
    if shortest_single_length < shortest_length:
        shortest_length = shortest_single_length
        first_min_pheromone_matrix = np.copy(pheromone_matrix)
        first_min_paths = shortest_paths

# Plotting
# ---------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
annotations = np.arange(1, N + 1, 1)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
pos1 = ax[0].pcolormesh(connection_matrix, cmap='gray_r', edgecolors='darkgreen', linewidth='0.01')
for axis in [ax[0].xaxis, ax[0].yaxis]:
    axis.set(ticks=np.arange(0.5, len(annotations)), ticklabels=annotations)
pos2 = ax[1].pcolormesh(distance_matrix, cmap='gray_r', edgecolors='darkred', linewidth='0.01')
for axis in [ax[1].xaxis, ax[1].yaxis]:
    axis.set(ticks=np.arange(0.5, len(annotations)), ticklabels=annotations)
pos3 = ax[2].plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
                  markerfacecolor='white', alpha=1, zorder=2)
annotations = np.arange(1, N + 1, 1)

for i, value in enumerate(annotations):
    ax[2].annotate(value, (vertice_array_x[i], vertice_array_y[i]), fontsize=10, color='black',
                   horizontalalignment='center', verticalalignment='center')
for i in range(N):
    for j in range(N):
        if connection_matrix[i, j] == 1:
            x_plot = np.array([vertice_array_x[i], vertice_array_x[j]])
            y_plot = np.array([vertice_array_y[i], vertice_array_y[j]])
            ax[2].plot(x_plot, y_plot, 'k-', linewidth=0.8, zorder=1)

# pos = ax.imshow(distance_matrix, cmap='gray')
ax[0].set_title("Connection matrix")
ax[1].set_title("Distance matrix")
ax[2].set_title("Graph")
ax[2].grid()
ax[2].set_xlim((-1, max_distance + 1))
ax[2].set_ylim((-1, max_distance + 1))
pos1.set_clim(0, 1)
pos2.set_clim(0, max_distance)
cb1 = fig.colorbar(pos1, ax=ax[0], shrink=0.778)
cb2 = fig.colorbar(pos2, ax=ax[1], shrink=0.778)

fig1, ax1 = plt.subplots(1, figsize=(12, 12))
ax1.plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='white', alpha=1, zorder=3)

ax1.plot(vertice_array_x[start_location], vertice_array_y[start_location], 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='green', alpha=1, zorder=4)
ax1.plot(vertice_array_x[end_location], vertice_array_y[end_location], 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='red', alpha=1, zorder=4)

for i, value in enumerate(annotations):
    ax1.annotate(value, (vertice_array_x[i], vertice_array_y[i]), fontsize=10, color='black',
                 horizontalalignment='center', verticalalignment='center')

value = annotations[start_location]
ax1.annotate(value, (vertice_array_x[start_location], vertice_array_y[start_location]), fontsize=10, color='black',
             horizontalalignment='center', verticalalignment='center', zorder=4)
value = annotations[end_location]
ax1.annotate(value, (vertice_array_x[end_location], vertice_array_y[end_location]), fontsize=10, color='black',
             horizontalalignment='center', verticalalignment='center', zorder=4)

linewidth1 = 1
linewidth2 = 2
for i in range(N):
    for j in range(N):
        if connection_matrix[i, j] == 1:
            x_plot = np.array([vertice_array_x[i], vertice_array_x[j]])
            y_plot = np.array([vertice_array_y[i], vertice_array_y[j]])
            ax1.plot(x_plot, y_plot, 'k-', linewidth=linewidth1, zorder=1)

length_of_shortest_path_array = len(shortest_paths)
for i in range(length_of_shortest_path_array):
    length_of_path = len(shortest_paths[i])
    current_shortest_path = shortest_paths[i]
    for j in range(length_of_path - 1):
        x_path = np.array([vertice_array_x[current_shortest_path[j]], vertice_array_x[current_shortest_path[j + 1]]])
        y_path = np.array([vertice_array_y[current_shortest_path[j]], vertice_array_y[current_shortest_path[j + 1]]])
        ax1.plot(x_path, y_path, 'r--', alpha=0.5, linewidth=linewidth2, zorder=2)

ax1.set_title("Ants that reached the end location")
ax1.set_aspect('equal')

fig2, ax2 = plt.subplots(1, figsize=(12, 12))
ax2.plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='white', alpha=1, zorder=3)

ax2.plot(vertice_array_x[start_location], vertice_array_y[start_location], 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='green', alpha=1, zorder=4)
ax2.plot(vertice_array_x[end_location], vertice_array_y[end_location], 'wo', markersize=12, markeredgecolor='black',
         markerfacecolor='red', alpha=1, zorder=4)

for i, value in enumerate(annotations):
    ax2.annotate(value, (vertice_array_x[i], vertice_array_y[i]), fontsize=10, color='black',
                 horizontalalignment='center', verticalalignment='center')

value = annotations[start_location]
ax2.annotate(value, (vertice_array_x[start_location], vertice_array_y[start_location]), fontsize=10, color='black',
             horizontalalignment='center', verticalalignment='center', zorder=4)
value = annotations[end_location]
ax2.annotate(value, (vertice_array_x[end_location], vertice_array_y[end_location]), fontsize=10, color='black',
             horizontalalignment='center', verticalalignment='center', zorder=4)

linewidth1 = 1
linewidth2 = 2
shortest_length_index = np.argmin(shortest_lengths)
shortest_single_path = shortest_paths[shortest_length_index]

for i in range(N):
    for j in range(N):
        if connection_matrix[i, j] == 1:
            x_plot = np.array([vertice_array_x[i], vertice_array_x[j]])
            y_plot = np.array([vertice_array_y[i], vertice_array_y[j]])
            ax2.plot(x_plot, y_plot, 'k-', linewidth=linewidth1, zorder=1)

length_of_path = len(shortest_single_path)
current_shortest_path = shortest_single_path
print(f"The shortest path is: {current_shortest_path}")
print(f"The shortest length is: {shortest_lengths[shortest_length_index]}")
for j in range(length_of_path - 1):
    x_path = np.array([vertice_array_x[current_shortest_path[j]], vertice_array_x[current_shortest_path[j + 1]]])
    y_path = np.array([vertice_array_y[current_shortest_path[j]], vertice_array_y[current_shortest_path[j + 1]]])
    ax2.plot(x_path, y_path, 'r--', alpha=1, linewidth=linewidth2, zorder=2)

ax2.set_aspect('equal')
ax2.set_title("The shortest path")

fig3, ax3 = plt.subplots(1, 3, figsize=(36, 12))
ax3[0].plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
            markerfacecolor='white', alpha=1, zorder=3)
ax3[1].plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
            markerfacecolor='white', alpha=1, zorder=3)
ax3[2].plot(vertice_array_x, vertice_array_y, 'wo', markersize=12, markeredgecolor='black',
            markerfacecolor='white', alpha=1, zorder=3)
for i, value in enumerate(annotations):
    for j in range(3):
        ax3[j].annotate(value, (vertice_array_x[i], vertice_array_y[i]), fontsize=10, color='black',
                        horizontalalignment='center', verticalalignment='center')

for i in range(N):
    for j in range(N):
        if connection_matrix[i, j] == 1:
            x_plot = np.array([vertice_array_x[i], vertice_array_x[j]])
            y_plot = np.array([vertice_array_y[i], vertice_array_y[j]])
            for k in range(3):
                ax3[k].plot(x_plot, y_plot, 'k-', linewidth=linewidth1, zorder=1)
                ax3[k].set_aspect('equal')
            ax3[0].plot(x_plot, y_plot, 'orange', linewidth=(linewidth2 * first_pheromone_matrix[i, j]), zorder=2,
                        alpha=1)
            ax3[1].plot(x_plot, y_plot, 'orange', linewidth=(linewidth2 * first_min_pheromone_matrix[i, j]), zorder=2,
                        alpha=1)
            ax3[2].plot(x_plot, y_plot, 'orange', linewidth=(linewidth2 * pheromone_matrix[i, j]), zorder=2, alpha=1)
            ax3[0].set_title("Initial pheromone matrix")
            ax3[1].set_title("Pheromone matrix, first minimum")
            ax3[2].set_title("Pheromone matrix, final step")

plt.tight_layout()
plt.show()
