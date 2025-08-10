import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import threading

from slam_interface import SLAM


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable
class ReconstructionWindow:
    """Main GUI window for BodySLAM 3D reconstruction interface."""
    
    def __init__(self, font_id: int, rgb_path: str, depth_path: str, model_path: str):
        """Initialize the reconstruction window.
        
        Args:
            font_id: Font identifier for GUI text
            rgb_path: Path to RGB images directory
            depth_path: Path to depth maps directory  
            model_path: Path to visual odometry model
        """
        self.window = gui.Application.instance.create_window('BodySLAM', 1280, 800)
        
        # Initialize SLAM system
        self.slam = self._initialize_slam(rgb_path, depth_path, model_path)
        
        # Initialize GUI components
        self._setup_gui(font_id)
    
    def _initialize_slam(self, rgb_path: str, depth_path: str, model_path: str) -> SLAM:
        """Initialize the SLAM system with provided paths."""
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB path does not exist: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth path does not exist: {depth_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        rgb_list = sorted([os.path.join(rgb_path, f) for f in os.listdir(rgb_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        depth_list = sorted([os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(rgb_list) != len(depth_list):
            raise ValueError(f"Mismatch in number of RGB ({len(rgb_list)}) and depth ({len(depth_list)}) images")
        if len(rgb_list) == 0:
            raise ValueError("No valid images found in the provided directories")
            
        return SLAM(rgb_list, depth_list, model_path)
    
    def _setup_gui(self, font_id: int) -> None:
        """Setup the GUI components."""
        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)


        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        '''
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)
        '''
        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        #w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = True
        self.is_running = True
        self.is_surface_updated = False

        self.idx = 0
        self.poses = []

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        #fps_panel_width = 7 * em
        #fps_panel_height = 2 * em
        #self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
        #                                rect.y, fps_panel_width,
        #                                fps_panel_height)

    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        max_points = 100000

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)
        self.is_started = True



    def _on_close(self):
        self.is_done = True

        if self.is_started:
            '''
            print('Saving model to {}...'.format(config.path_npz))
            self.model.voxel_grid.save(config.path_npz)
            print('Finished.')

            mesh_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.ply'
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, config,
                                        mesh_fname)
            
            print('Finished.')

            log_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.log'
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')
            '''
            print('Finished.')

        return True

    def init_render(self, depth_ref = None, color_ref = None):
        self.input_depth_image.update_image(depth_ref)
        self.input_color_image.update_image(color_ref)

        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth = None, input_color = None, pcd=None, frustum = None):
        self.input_depth_image.update_image(input_depth)
        self.input_color_image.update_image(input_color)
        #self.widget3d.scene.update_geometry(
        #    'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
        #                   rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.remove_geometry("points")

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)
        self.widget3d.scene.add_geometry("points", pcd, mat)


    def update_output_info(self, num_frames, frame_id, transformation):
        info = 'Frame {}/{}\n\n'.format(num_frames, frame_id)
        info += 'Transformation:\n{}\n'.format(
            np.array2string(transformation),
            precision = 3, max_line_width=40, suppress_small=True)
        self.output_info.txt = info

    def update_main(self):
        while self.idx < self.slam.n_frames:
            #curr_rgbd, prev_rgbd, pcd, curr_global_pose = self.slam.main_loop_gui(self.idx)
            #print(curr_global_pose)

            if self.idx == 0:
                curr_rgbd, pcd, curr_global_pose = self.slam._first_loop()
                pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.init_render(curr_rgbd.colored_depth, curr_rgbd.o3d_color))
            if self.idx > 0:
                curr_rgbd, prev_rgbd, pcd, curr_global_pose = self.slam._sequential_loop(self.idx)
                pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)

                frustum = o3d.geometry.LineSet.create_camera_visualization(
                    curr_rgbd.cv2_color.shape[1], curr_rgbd.cv2_color.shape[0], self.slam.o3d_intrinsic.intrinsic_matrix,
                    curr_global_pose, 0.2)
                frustum.paint_uniform_color([0.961, 0.475, 0.000])

                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.update_render(curr_rgbd.colored_depth, curr_rgbd.o3d_color, pcd=pcd, frustum=frustum))

            self.idx += 1



def main():
    """Main entry point for the GUI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BodySLAM 3D Reconstruction GUI")
    parser.add_argument("--rgb_path", type=str, required=True, 
                       help="Path to RGB images directory")
    parser.add_argument("--depth_path", type=str, required=True,
                       help="Path to depth maps directory")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to visual odometry model")
    
    args = parser.parse_args()
    
    # Initialize GUI application
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    
    try:
        w = ReconstructionWindow(mono, args.rgb_path, args.depth_path, args.model_path)
        app.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing SLAM: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
