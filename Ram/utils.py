"""
Utils: Contains code for POD basis formation, saving results, and Partition-of-Unity
"""

import time
import jax
import json
import jax.numpy as jnp
from collections import OrderedDict
from scipy.spatial import KDTree
from collections import defaultdict
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Point
from torch.utils.data import DataLoader
from jax.tree_util import tree_map
from torch.utils.data import Dataset
from torch.utils.data import default_collate
import inspect
import re
import numpy as np


def split_linear_string(input_string):
    # Define a regular expression pattern to match the desired format
    pattern = re.compile(r'([a-zA-Z]+)_([0-9]+)')

    # Use the pattern to search for matches in the input string
    match = pattern.match(input_string)

    if match:
        # Extract the matched groups (word and number)
        word = match.group(1)
        number = int(match.group(2))

        # Return the extracted values
        return word, number
    else:
        # Return None if no match is found
        return None


def split_conv_string(input_string):
    # Define a regular expression pattern to match the desired format
    pattern = re.compile(r'([a-zA-Z]+)_([0-9]+)_([0-9]+)_?([0-9]*)_([a-zA-Z]+)')

    # Use the pattern to search for matches in the input string
    match = pattern.match(input_string)

    if match:
        # Extract the matched groups (words and numbers)
        words = match.group(1)
        number1 = int(match.group(2))
        number2 = int(match.group(3))
        number3 = int(match.group(4)) if match.group(4) else None
        operation = match.group(5).upper()

        # Return the extracted values
        return words, number1, number2, number3, operation
    else:
        # Return None if no match is found
        return None


def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


class fstr:
    def __init__(self, payload):
        self.payload = payload

    def __str__(self):
        vars = inspect.currentframe().f_back.f_globals.copy()
        vars.update(inspect.currentframe().f_back.f_locals)
        return self.payload.format(**vars)

    def __add__(self, another):
        if isinstance(another, fstr):
            self.payload += another.payload
        else:
            self.payload += another


def wendland(r, k=1):
    """
    Compactly supported Wendland kernel with C2 and C4 options (parameter k to the function)
    """
    # making r's >= 1 to be exactly 1
    r = r.at[r >= 1.0].set(1.0)

    r_squared = jnp.power(r, 2)
    try:
        if k == 1:
            phi = (jnp.power(1-r, 4)) * ((4*r) + 1)

        elif k == 2:
            phi = (jnp.power(1-r, 6)) * (35*r_squared + 18*r + 3)
    except:
        raise Exception(f"ERROR: k= {k} not implemented manually.\n")

    return phi


class PU:
    """
    General PU class for partitioning an arbitrary domain in d-dimensions using d-dimensional spheres
    """

    def __init__(self, config={}):
        assert config != {} or config is not None, "Config cannot be empty"
        self.num_partitions = config["num_partitions"]
        self.dim = config["dim"]
        if config.get("overlap", None) is not None:
            self.overlap = config["overlap"]

    def partition_domain(self, points, centers=None, radius=None, plot_bool=False):
        """
        assuming points is either numpy or jax array, places d-dimensional centers in a d-cube domain and makes the centers KDTree
        Optionally, the patch centers and variable radii can be given
        """
        points = points
        points_tree = KDTree(points)
        self.dim = points.shape[1]
        dim_ranges = []
        mins = points_tree.mins
        maxes = points_tree.maxes

        if centers is None:
            self.H = jnp.abs(mins-maxes).max().item()
            # self.centers_tensor = jnp.linspace(mins, maxes, self.num_partitions)
            # trying something better than linspace, using meshgrid for a more diverse set of centers
            centers_tensor_not = jnp.linspace(mins, maxes, self.num_partitions)
            temp_list = []
            for i in range(self.dim):
                temp_list.append(centers_tensor_not[:, i])
            e_list = np.meshgrid(*temp_list)
            e_list = [e.flatten() for e in e_list]
            self.centers_tensor = np.column_stack([*e_list]).reshape(-1, self.dim)

            # to check whether centers are in the interior region of point cloud
            concave_hull = alphashape.alphashape(points, 6.0)
            valid_idx = []
            for i in range(len(self.centers_tensor)):
                if concave_hull.contains(Point(self.centers_tensor[i])):
                    valid_idx.append(i)

            self.centers_tensor = self.centers_tensor[valid_idx]
            self.centers_tensor = self.centers_tensor[1::(self.centers_tensor.shape[0]//self.num_partitions)-1][:self.num_partitions]
        else:
            self.centers_tensor = jnp.asarray(centers).astype(jnp.float32)

        if radius is None:
            self.radius_given = False
            self.radius = (1 + self.overlap) * (jnp.sqrt(self.dim) * self.H / 2.)
            assert self.radius > 0, "Patch radius is negative or zero"
            self.center_tree = KDTree(self.centers_tensor)
        else:
            self.radius_given = True
            self.radius = jnp.asarray(radius).astype(jnp.float32)
            center_dict = OrderedDict()  # making a list of centers based on the unique radii
            for i in range(len(self.radius)):
                assert self.radius[i] > 0, f"{i} Patch radius is negative or zero"
                if str(self.radius[i]) not in center_dict.keys():
                    center_dict[str(self.radius[i])] = [self.centers_tensor[i]]
                else:
                    center_dict[str(self.radius[i])].append(self.centers_tensor[i])

            self.unique_radius_list = [float(k) for k in center_dict.keys()]
            self.center_list = list(center_dict.values())  # contains list of list of centers
            self.center_tree_list = []  # KDTree for different sized centers
            for center_collection in self.center_list:
                self.center_tree_list.append(KDTree(jnp.asarray(center_collection)))
            self.num_center_trees = len(self.center_tree_list)

        if plot_bool:
            fig, ax = plt.subplots()
            plt.scatter(points[:, 0], points[:, 1], s=3, color="red")
            plt.scatter(self.centers_tensor[:, 0], self.centers_tensor[:, 1], s=5, color="red")
            for idx, center in enumerate(self.centers_tensor):
                circle = plt.Circle(center, self.radius[idx], edgecolor='blue', facecolor='none')
                ax.plot(center[0], center[1], marker='o', markersize=5, color='green')
                ax.add_patch(circle)
            plt.show()

    def get_radius(self):
        if self.radius == 0.0:
            raise Exception("ERROR: radius is not computed yet\n")
        return self.radius

    def check_partioning(self, points, change_M=False, min_partitions_per_point=2):
        """
        Gets the maximum number of patches any point belongs to, checks to make sure that every point belongs to min_partitions_per_point patches
        Checks all patch centers, removes partitions that don't contain any points (if any such exist)
        """
        start = time.perf_counter()
        partition_sizes = []

        # checking all partitions
        M = self.centers_tensor.shape[0]
        points_tree = KDTree(points)
        for i in range(M):
            cur_radius = self.radius[i] if self.radius_given else self.radius  # whether radius list is given or a single radius is computed
            temp_indices = jnp.asarray(points_tree.query_ball_point(self.centers_tensor[i], cur_radius))
            if len(temp_indices) == 0:
                M -= 1
                self.centers_tensor = jnp.delete(self.centers_tensor, i, 0) if change_M else self.centers_tensor
                continue
            partition_sizes.append(len(temp_indices))

        if change_M:
            self.num_partitions = M
            self.center_tree = KDTree(self.centers_tensor)

        # checking all points
        K = 0
        for i in range(points.shape[0]):
            if not self.radius_given:
                temp_indices = self.center_tree.query_ball_point(points[i], self.radius)
            else:
                temp_indices = []
                for j in range(self.num_center_trees):
                    cur_tree_indices = self.center_tree_list[j].query_ball_point(points[i], self.unique_radius_list[j])
                    # need to convert each tree's local index into the global index of all partition centers
                    cur_tree_indices = [np.where((self.centers_tensor == self.center_list[j][p]).all(axis=1))[0][0] for p in cur_tree_indices]
                    temp_indices += cur_tree_indices

            if len(temp_indices) < min_partitions_per_point:
                print(points[i])
                raise Exception(f"ERROR: {min_partitions_per_point} partition not satisfied for point {i}\n")
            K = max(K, len(temp_indices))

        self.min_partitions_per_point = min_partitions_per_point
        time_taken = time.perf_counter() - start
        print(f"{time_taken}s taken")

        return K, M, partition_sizes

    def form_points_per_group(self, points, safety=True):
        """
        A grouping strategy where the points in the domain are divided into groups where the points in each group belong the same number of patches (even if the patches are different). Allows for Jax optimizations in implementing the PU inference logic
        """
        start = time.perf_counter()
        group_indices = OrderedDict()
        group_points = OrderedDict()
        participation_idx = OrderedDict()
        radius_arrs = OrderedDict()

        participation_numbers = []

        if not self.radius_given:
            idx_list = self.center_tree.query_ball_point(points, self.radius)
        else:
            idx_list = []
            for i in range(points.shape[0]):
                temp_indices = []
                for j in range(self.num_center_trees):
                    cur_tree_indices = self.center_tree_list[j].query_ball_point(points[i], self.unique_radius_list[j])
                    # need to convert each tree's local index into the global index of all partition centers
                    cur_tree_indices = [np.where((self.centers_tensor == self.center_list[j][p]).all(axis=1))[0][0] for p in cur_tree_indices]
                    temp_indices += cur_tree_indices

                idx_list.append(temp_indices)

        for i, idx in enumerate(idx_list):
            cur_num_patches = len(idx)
            participation_numbers.append(cur_num_patches)

            if cur_num_patches < self.min_partitions_per_point and safety:
                raise Exception(f"ERROR: {min_partitions_per_point} partition not satisfied for point {i}\n")

            if str(cur_num_patches) not in group_indices.keys():
                group_indices[str(cur_num_patches)] = [i]
                group_points[str(cur_num_patches)] = [points[i]]
                participation_idx[str(cur_num_patches)] = [idx]
                if self.radius_given:
                    radius_arrs[str(cur_num_patches)] = [self.radius[jnp.asarray(idx)]]
            else:
                group_indices[str(cur_num_patches)].append(i)
                group_points[str(cur_num_patches)].append(points[i])
                participation_idx[str(cur_num_patches)].append(idx)
                if self.radius_given:
                    radius_arrs[str(cur_num_patches)].append(self.radius[jnp.asarray(idx)])

        group_points = [jnp.asarray(val) for val in group_points.values()]
        group_indices = [jnp.asarray(val).astype(jnp.int32) for val in group_indices.values()]
        participation_idx = [jnp.asarray(val).astype(jnp.int32) for val in participation_idx.values()]
        if self.radius_given:
            radius_arrs = [jnp.asarray(val).astype(jnp.float32) for val in radius_arrs.values()]

        time_taken = time.perf_counter() - start
        print(f"{time_taken}s taken")
        return len(group_points), participation_idx, group_points, group_indices, sorted(list(set(participation_numbers))), radius_arrs

    def form_weights_per_group(self, points=None, points_per_group=None, participation_idx=None, radius_arrs=None, weight_choice="wendland"):
        """
        Forms the wendland functions for all points in the given groups
        """
        start = time.perf_counter()
        if points is None and points_per_group is None:
            raise Exception("Both points and points_per_group cannot be None")

        if weight_choice == "wendland":
            weight_func = wendland

        if points_per_group is None:
            _, participation_idx, points_per_group, indices_per_group = form_points_per_group(points)

        def l2_distance(a, b):
            return jnp.linalg.norm(a-b, axis=-1)

        l2_distance_vmap = jax.jit(jax.vmap(l2_distance, in_axes=(0, 0), out_axes=0))

        centers_ = self.centers_tensor.copy()
        weights_per_group = []
        for i in range(len(points_per_group)):
            if self.radius_given:
                scaled_distances = l2_distance_vmap(points_per_group[i], centers_[participation_idx[i]]) / radius_arrs[i]
            else:
                scaled_distances = l2_distance_vmap(points_per_group[i], centers_[participation_idx[i]]) / self.radius
            weights = weight_func(scaled_distances)
            cur_group_weights = weights * (1 / jnp.expand_dims(weights.sum(1), 1))  # making the weights sum to 1

            weights_per_group.append(cur_group_weights)

        time_taken = time.perf_counter() - start
        print(f"{time_taken}s taken")
        return weights_per_group

