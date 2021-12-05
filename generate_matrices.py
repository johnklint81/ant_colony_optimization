import numpy as np
import matplotlib.pyplot as plt

N = 12  # number of vertices
max_number_of_connections = 4
max_distance = 10
# ratio = 0.5  # total number of connections in the graph, max is N(N-1)
start_location = int(np.random.randint(0, N))


def create_matrices(_N):
    # Connection matrix
    # _connection_matrix = np.random.choice([0, 1], size=(_N, _N), p=[1 - _ratio, _ratio])
    # _connection_matrix = np.floor((_connection_matrix + _connection_matrix.T) / 2).astype(int)
    # np.fill_diagonal(_connection_matrix, 0)
    # Can produce vertices with no connections unfortunately

    # Connection matrix
    _connection_matrix = np.zeros([_N, _N])
    for i in range(_N):
        _ratio = np.abs(np.random.randint(low=0, high=_N - 2)) / _N
        _random_index = np.random.choice(np.arange(_N), replace=False, size=int(max(2, _N * _ratio)))
        _connection_matrix[i, _random_index] = 1
    _connection_matrix = np.ceil((_connection_matrix + _connection_matrix.T) / 2).astype(int)
    np.fill_diagonal(_connection_matrix, 0)

    # Distance matrix
    _distance_matrix = np.random.uniform(low=1, high=max_distance, size=(_N, _N))
    _distance_matrix = np.where(_connection_matrix == 1, _distance_matrix, np.inf)
    np.fill_diagonal(_distance_matrix, np.inf)
    _distance_matrix = (_distance_matrix + _distance_matrix.T) / 2

    # Weight matrix
    _weight_matrix = 1 / _distance_matrix
    np.fill_diagonal(_weight_matrix, 0)

    # Pheromone matrix
    _pheromone_matrix = np.ones([N, N])
    _pheromone_matrix = np.where(_connection_matrix == 1, _pheromone_matrix, 0)
    return _connection_matrix, _distance_matrix, _weight_matrix, _pheromone_matrix


def create_path(_connection_matrix, _start_location, _N):
    _path = np.zeros(_N + 1, dtype=int)
    _path[0] = _start_location
    for i in range(_N):
        _candidates = np.argwhere(_connection_matrix[:, _path[i]] == 1).flatten()
        _path[i + 1] = int(np.random.choice(_candidates))
    return _path


def path_length(_path, _distance_matrix):
    _length_of_path = 0
    _path = _path.astype(int)
    for i in range(len(_path) - 1):
        _length_of_path += _distance_matrix[_path[i + 1], _path[i]]
    return _length_of_path


def simplify_path_once(_path):
    _duplicate_element_indices = get_duplicates(_path)
    _number_of_duplicate_elements = len(_duplicate_element_indices)
    print(f"Duplicates on indices: {_duplicate_element_indices}")
    print("------------")
    print(f"path: {_path}")
    _path_candidates = []
    _duplicate_indices = []
    for i in range(_number_of_duplicate_elements):
        _duplicates = np.array(_duplicate_element_indices[i])
        # print(f"Correcting duplicate indices: {_duplicates}")
        _number_of_duplicates = len(_duplicates)
        _temp_path = np.copy(_path)
        _temp_path = np.delete(_temp_path, np.arange(_duplicates[0], _duplicates[-1], 1))
        # print(f"Corrected path: {_temp_path}")
        _temp_duplicate_element_indices = get_duplicates(_temp_path)
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

            _temp_simplified_paths, _temp_path_candidates = store_candidates(_temp_path_candidates, _duplicate_indices)
            _number_of_simplified_paths = len(_temp_simplified_paths)
            for j in range(_number_of_simplified_paths):
                _simplified_paths.append(_temp_simplified_paths[j])
            _number_of_path_candidates = len(_temp_path_candidates)
            for j in range(_number_of_path_candidates):
                print(_temp_path_candidates[j])
                _path_candidates.append(_temp_path_candidates[j])
        _path_candidates_list = np.copy(_path_candidates)
        if len(_path_candidates_list) == 0:
            _duplicates = False
        _path_candidates = []
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
                print(_path_candidates[j])
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
            _duplicate_element_indices.append(_indices_filter[i])
    return _duplicate_element_indices


connection_matrix, distance_matrix, weight_matrix, pheromone_matrix = create_matrices(N)
# print(connection_matrix)
# print('--------')
# print(distance_matrix)
# print('--------')
# print(weight_matrix)
# print('--------')
# print(pheromone_matrix)

path = create_path(connection_matrix, start_location, N)
print(f"Original path: {path}")

path_candidates = simplify_entire_path(path)
print(f"Path candidates: {path_candidates}")
seen = set()
path_candidates_unique = [item for item in path_candidates if not(tuple(item) in seen or seen.add(tuple(item)))]
print(f"Path candidates (no duplicates): {path_candidates_unique}")
print("-----------------------------------")
original_length_of_path = path_length(path, distance_matrix)
print(f"Length of original path: {original_length_of_path:.3f}")

number_of_paths = len(path_candidates_unique)
shortest_length = np.inf
for i in range(number_of_paths):
    length_of_simplified_path = path_length(path_candidates_unique[i], distance_matrix)
    print(f"Length of simplified path number {i}: {length_of_simplified_path:.3f}")
    if length_of_simplified_path < shortest_length:
        shortest_length = length_of_simplified_path
print(f"Length of shortest path: {shortest_length:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
pos1 = ax[0].pcolormesh(connection_matrix, cmap='gray_r', edgecolors='darkgreen', linewidth='0.01')
pos2 = ax[1].pcolormesh(distance_matrix, cmap='gray_r', edgecolors='darkred', linewidth='0.01')
# pos = ax.imshow(distance_matrix, cmap='gray')
ax[0].set_title("Connection matrix")
ax[1].set_title("Distance matrix")
pos1.set_clim(0, 1)
pos2.set_clim(0, max_distance)
cb1 = fig.colorbar(pos1, ax=ax[0], shrink=0.755)
cb2 = fig.colorbar(pos2, ax=ax[1], shrink=0.755)

plt.tight_layout()
plt.show()
