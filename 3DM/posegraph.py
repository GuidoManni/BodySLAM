import open3d as o3d
import numpy as np
# TODO: implement pypose posegraph

class PoseGraph:
    def __init__(self, max_correspondence_distance=0.005, edge_prune_threshold=0.05, preference_loop_closure=0.01,
                 reference_node=0):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=reference_node)
        self.convergence_criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()

    def add_node(self, extrinsic_matrix):
        extrinsic_matrix = self._check_type(extrinsic_matrix)
        node = o3d.pipelines.registration.PoseGraphNode(extrinsic_matrix)
        self.pose_graph.nodes.append(node)

    def add_edge(self, motion_matrix, source_id, target_id, uncertain, info=np.eye(6)):
        motion_matrix = self._check_type(motion_matrix)
        info = self._check_type(info)
        edge = o3d.pipelines.registration.PoseGraphEdge(source_node_id=source_id, target_node_id=target_id,
                                                        transformation=motion_matrix, information=info,
                                                        uncertain=uncertain)
        self.pose_graph.edges.append(edge)

    def optimize(self):
        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            self.option)

    def _check_type(self, matrix):
        # we want to save only numpy array so we need to check if they are of the right type
        # if not we convert it
        if not isinstance(matrix, np.ndarray):
            return matrix.cpu().numpy()
        else:
            return matrix

