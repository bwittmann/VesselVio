"""
The terminal-based access to the analysis pipeline. Headless, and doesn't come with the full functionality of the GUI. Best for testing or modifying the pipeline. Updates made here will need to be reflected on the QtThreading page.
"""

__author__ = "Jacob Bumgarner <jrbumgarner@mix.wvu.edu>"
__license__ = "GPLv3 - GNU General Pulic License v3 (see LICENSE)"
__copyright__ = "Copyright 2022 by Jacob Bumgarner"
__webpage__ = "https://jacobbumgarner.github.io/VesselVio/"
__download__ = "https://jacobbumgarner.github.io/VesselVio/Downloads"


import os
import time
from pathlib import Path
import sys

import igraph as ig

from library import (
    feature_extraction as FeatExt,
    graph_io as GIO,
    graph_processing as GProc,
    helpers,
    image_processing as ImProc,
    input_classes as IC,
    results_export as ResExp,
    volume_processing as VolProc,
    volume_visualization as VolVis,
)
from library.annotation import (
    labeling,
    segmentation,
    segmentation_prep,
    tree_processing,
)

repo_root = Path(__file__).parents[1] / 'data_generation'
sys.path.insert(0, str(repo_root))
from label_propagation import propagate_lg


def printc(text, color="green", bold=False):
    """To highlight key steps."""
    colors = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }

    color_code = colors.get(color.lower(), "37")
    style = "1" if bold else "0"

    print(f"\033[{style};{color_code}m{text}\033[0m")


#######################
### Input Functions ###
#######################
# Process graphs that are loaded into the dataset.
def process_graph(file_path, gen_options, graph_options, vis_options, verbose):
    if type(file_path) == dict:
        filename = ImProc.get_filename(file_path["Vertices"])
    else:
        filename = ImProc.get_filename(file_path)
    if verbose:
        tic = time.perf_counter()
        print("Analyzing file:", filename)
        print("Loading graph...")

    # Make sure our resolution is in the proper format
    resolution = ImProc.prep_resolution(gen_options.resolution)

    # Import graph
    graph = GIO.graph_loading_dock(
        file_path, graph_options, resolution, Visualize=vis_options.visualize
    )

    # Remove
    if graph_options.graph_type == "Centerlines" and graph_options.filter_cliques:
        if verbose:
            print("Filtering branch point cliques...", end="\r")

        GProc.clique_filter_input(graph, verbose=verbose)

    # Prune segments if needed
    if gen_options.prune_length:
        if verbose:
            print("Pruning endpoint segments...", end="\r")
        GProc.prune_input(
            graph,
            gen_options.prune_length,
            resolution,
            centerline_smoothing=graph_options.smooth_centerlines,
            graph_type=graph_options.graph_type,
            verbose=verbose,
        )

    # Filter segments
    if gen_options.filter_length:
        if verbose:
            print("Filtering isolated segments...", end="\r")
        GProc.filter_input(
            graph,
            gen_options.filter_length,
            resolution,
            centerline_smoothing=graph_options.smooth_centerlines,
            graph_type=graph_options.graph_type,
            verbose=verbose,
        )

    if verbose:
        print("Analyzing graph...", end="\r")
    result, seg_result = FeatExt.feature_input(
        graph,
        filename,
        graph_type=graph_options.graph_type,
        centerline_smoothing=graph_options.smooth_centerlines,
        save_seg_results=gen_options.save_seg_results,
        reduce_graph=vis_options.visualize,
        verbose=verbose,
    )

    if vis_options.visualize:
        if verbose:
            print("Visualizing graph...", end="\r")
        if "hex" not in graph.es.attributes():
            graph.es["hex"] = ["FFFFFF"]
        VolVis.mesh_construction(
            graph, vis_options, graph_type=graph_options.graph_type, verbose=verbose
        )

    else:
        if verbose:
            print("Writing results...", end="\r")
        ResExp.cache_result(result)
        ResExp.write_results(gen_options.results_folder)
        if gen_options.save_seg_results:
            ResExp.write_seg_results(
                seg_result, gen_options.results_folder, filename, roi_Name="None"
            )
        if verbose:
            print(f"Analysis complete in {time.perf_counter() - tic:0.2f} seconds.")


# Process raw segmented volumes
def process_volume(
    path_to_file, gen_options, ann_options, vis_options, iteration, verbose
):
    filename = path_to_file.name
    volume_file = [path for path in list(path_to_file.iterdir()) if 'part_vessels_ss.nii' in path.name][0]

    if verbose:
        tic = time.perf_counter()
        print("Processing dataset:", filename)


    if ann_options.annotation_type == "None":
        annotation_data = {None: None}
    else:
        annotation_data = tree_processing.convert_annotation_data(
            ann_options.annotation_regions, ann_options.annotation_atlas
        )
        roi_array = segmentation_prep.build_roi_array(
            annotation_data, annotation_type=ann_options.annotation_type
        )

    g_main = ig.Graph()

    for i, roi_name in enumerate(annotation_data.keys()):
        if verbose and roi_name:
            if roi_name:
                print(f"Analyzing {filename}: {roi_name}.")
            else:
                print(f"Analyzing {filename}.")

        ## Image and volume processing.
        # region
        volume, image_shape, spacing, volume_raw, anatomy = ImProc.load_volume(path_to_file, raw_file=True, anatomy=True, verbose=verbose)
        resolution = ImProc.prep_resolution(gen_options.resolution) # make sure the resolution is in the proper format
        printc(f'Loaded volume of shape {image_shape} and spacing {spacing}.')  # TODO check difference between spacing and resolution

        if volume is None:  # make sure the image was loaded.
            if verbose:
                print("Error loading volume.")
            break
        elif not ImProc.binary_check(volume):
            if verbose:
                print("Error: Non-binary image loaded.")
            break

        # If there as an ROI, segment the ROI from the volume.
        if roi_name:
            roi_id = i % 255
            if roi_id == 0:
                if not helpers.check_storage(volume_file):
                    file_size = helpers.get_file_size(volume_file, GB=True)
                    if verbose:
                        print(
                            f"Not enough disk space! Need at least {file_size:.1f}GB of free space for the volume annotation."
                        )
                    return

                # We have to relabel every 255 elements because the volume.dtype == uint8.
                roi_sub_array = roi_array[i : i + 255]
                roi_volumes, minima, maxima = labeling.volume_labeling_input(
                    volume,
                    ann_options.annotation_file,
                    roi_sub_array,
                    ann_options.annotation_type,
                    verbose=verbose,
                )
                if roi_volumes is None:
                    break
            roi_volume = roi_volumes[roi_id]
            if roi_volume > 0:
                point_minima, point_maxima = minima[roi_id], maxima[roi_id]
                volume = segmentation.segmentation_input(
                    point_minima, point_maxima, roi_id + 1, verbose=verbose
                )

            # Make sure the ROI is in the volume.
            if not roi_volume or not ImProc.segmentation_check(volume):
                ResExp.cache_result(
                    [filename, roi_name, "ROI not in dataset."]
                )  # Cache results
                if verbose:
                    print("ROI Not in dataset.")
                continue
        else:
            volume, point_minima, volume_raw, anatomy = VolProc.volume_prep(volume, volume_raw, anatomy)  # crops volume to bounding 
            roi_name = "None"
            roi_volume = "NA"

        # Pad the volume for skeletonizatino
        volume = VolProc.pad_volume(volume) # pads by one on each side
        volume_raw = VolProc.pad_volume(volume_raw)
        anatomy = VolProc.pad_volume(anatomy)
        
        # Skeletonize, then find radii of skeleton points
        points = VolProc.skeletonize(volume, verbose=verbose)   # (n, 3); centerline coords
        printc(f'Generated skeleton with {points.shape[0]} elements.')

        # Calculate radii
        skeleton_radii, vis_radii = VolProc.radii_calc_input(   # (n, 1); radii for centerline coords
            volume,
            points,
            resolution,
            gen_vis_radii=vis_options.visualize or gen_options.save_graph,
            verbose=verbose,
        )

        # Calculate intensity
        skeleton_int, skeleton_int_shpere_mean, skeleton_int_shpere_min, \
        skeleton_int_shpere_max, skeleton_int_shpere_sd = VolProc.int_calc_input(   # (n, 1); intensity in HU for centerline
            volume,
            volume_raw,
            skeleton_radii,
            points,
            verbose=verbose,
        )

        # Add anatomy
        skeleton_anatomy = VolProc.ana_match_input(   # (n, 1); anatomy label
            points,
            anatomy,
            verbose=verbose,
        )

        # Now, we can treat 2D arrays as 3D arrays for compatibility
        # with the rest of our pipeline.
        if volume.ndim == 2:
            points, volume, volume_shape = ImProc.reshape_2D(
                points, volume, verbose=verbose
            )
        else:
            volume_shape = volume.shape

        # At this point, delete the volume
        del volume
        # endregion

        ## Graph construction.
        # region
        # Send information to graph network creation.
        graph = GProc.create_graph(
            volume_shape,
            skeleton_radii,
            vis_radii,
            points,
            point_minima,
            skeleton_int,
            skeleton_int_shpere_mean,
            skeleton_int_shpere_min,
            skeleton_int_shpere_max,
            skeleton_int_shpere_sd,
            skeleton_anatomy,
            verbose=verbose,
        )
        printc(f'Generated skeleton graph with {graph.vcount()} nodes and {graph.ecount()} edges.')

        if gen_options.prune_length > 0:
            # Prune connected endpoint segments based on a user-defined length
            GProc.prune_input(
                graph, gen_options.prune_length, resolution, verbose=verbose
            )
            printc(f'Pruned skeleton graph resulting in {graph.vcount()} nodes and {graph.ecount()} edges.')

        # Filter isolated segments that are shorter than defined length
        # If visualizing the dataset, filter these from the volume as well.
        GProc.filter_input(
            graph, gen_options.filter_length, resolution, verbose=verbose
        )
        printc(f'Filtered skeleton graph resulting in {graph.vcount()} nodes and {graph.ecount()} edges.')

        # endregion
        ## Analysis.
        result, seg_results = FeatExt.feature_input(
            graph,
            resolution,
            filename,
            volume_file,
            image_dim=gen_options.image_dimensions,
            image_shape=image_shape,
            roi_name=roi_name,
            roi_volume=roi_volume,
            save_seg_results=gen_options.save_seg_results,
            # Reduce graph if saving or visualizing
            reduce_graph=vis_options.visualize or gen_options.save_graph,
            verbose=verbose,
            centerline_smoothing=True
        )
        ResExp.cache_result(result)  # Cache results

        # if gen_options.save_seg_results:
        #     ResExp.write_seg_results(seg_results, results_folder, filename, roi_name)

        # if gen_options.save_graph and not vis_options.visualize:
        #     GIO.save_graph(graph, filename, results_folder, verbose=verbose)

        VolVis.vana_graph_vis(path_to_file)
        printc(f"Finished processing {filename}.")
        sys.exit()


        if roi_name != "None":
            graph.es["hex"] = [annotation_data[roi_name]["colors"][0]]
            graph.es["roi_ID"] = i
        else:
            graph.es["hex"] = ["FFFFFF"]
        g_main += graph
        del graph

    if verbose:
        print(
            f"Dataset analysis completed in a total "
            f"of {time.perf_counter() - tic:0.2f} seconds."
        )

    ## Visualization
    if vis_options.visualize:
        if (
            not vis_options.visualize
            or not vis_options.load_smoothed
            and not vis_options.load_original
        ):
            volume = None
        else:
            volume, *_ = ImProc.load_volume(volume_file)
            volume = ImProc.prep_numba_compatability(volume)
            # Don't bound for visualization, as points will be true, not relative
            volume = VolProc.pad_volume(volume)
            if volume.ndim == 2:
                _, volume, _ = ImProc.reshape_2D(points, volume, verbose=verbose)

        VolVis.mesh_construction(
            g_main, vis_options, volume, file_name=volume_file, iteration=iteration, verbose=verbose
        )

    # ResExp.write_results(results_folder, gen_options.image_dimensions, verbose=True)

    # Make sure we delete the labeled_cache_volume if it exists
    ImProc.clear_labeled_cache()
    return


if __name__ == "__main__":
    compiler_file = os.path.join(
        helpers.get_cwd(), "VesselVio/library/volumes/JIT_volume.nii"
    )  # DON'T DELETE

    #####################
    ### Graph Options ###
    #####################
    # region
    graph_file_format = "csv"  # 'csv', 'graphml', 'gml', 'edgelist', etc.
    delimiter = ";"  # If the file is a csv, what is the delimiter?
    vertex_representation = "Branches"  # 'Centerlines' or 'Branches' See documentation

    attribute_key = IC.AttributeKey(
        X="pos_x",
        Y="pos_x",
        Z="pos_x",
        vertex_radius="radius",
        edge_radius="avgRadiusAvg",
        length="length",
        volume="volume",
        surface_area="",
        tortuosity="curveness",
        edge_source="node1id",
        edge_target="node2id",
        edge_hex="",
    )

    centerline_smoothing = True  # Smooth centerlines in vertex-based graphs?
    clique_corrections = True  # Eliminate cliques from vertex-based graphs?

    graph_options = IC.GraphOptions(
        graph_file_format,
        vertex_representation,
        clique_corrections,
        centerline_smoothing,
        attribute_key,
        delimiter,
    )
    # endregion

    ######################
    ### Volume Options ###
    ######################
    # region
    # Filepath to the annotation. RGB series folder OR .nii Allen brain atlas file
    annotation_file = "ANNOTATION_VOLUME_FILE.nii"

    atlas = "VesselVio/libraryannotations/annotation_trees/p56 Mouse Brain.json"
    annotation_type = "ID"  # 'RGB' or 'ID'

    annotation_regions = ["Dentate gyrus, molecular layer"]

    annotation_options = IC.AnnotationOptions(
        annotation_file, atlas, annotation_type, annotation_regions
    )

    # endregion

    #############################
    ### Visualization Options ###
    #############################
    # region
    visualize = True  # Visualize the dataset?
    simplified_visualization = False  # Faster but less detailed visualization.
    # Network
    load_network = False
    # Scaled
    load_scaled = True
    # General
    show_branch_points = False
    show_end_points = False
    scalars = "Radius"  # 'Radius', 'Length', 'Tortuosity', 'Surface Area', 'Volume', 'Original_RGB', 'Shifted_RGB', 'Rainbow_RGB':
    render_annotation_colors = False  # True to visualize annotation colors
    color_map_theme = "viridis"  # See PyVista color map themes

    # Volumes
    load_original_volume = False
    load_smoothed_volume = False
    # Movie options
    create_movie = False  # Generate orbital movie?
    movie_title = "Synth Demo4"
    viewup = [-0.56, -0.44, 0.69]

    vis_options = IC.VisualizationOptions(
        visualize,
        simplified_visualization,
        load_scaled,
        load_network,
        load_original_volume,
        load_smoothed_volume,
        scalars,
        color_map_theme,
        show_branch_points,
        show_end_points,
        create_movie,
        movie_title,
        viewup,
        render_annotation_colors,
    )

    # endregion

    #######################
    ### General Options ###
    #######################
    # region
    # General features
    resolution = 1  # Single number or [X, Y, Z] format
    prune_length = 5  # Prune end point segments shorter than prune_length
    filter_length = 10  # Filter isolated segments shorter than filter_length
    image_dimensions = 3  # 2 or 3. Affects features extraction. 2D datasets can be treated as if they were 3D.

    # Results/graph export
    save_segment_results = True  # Save individual segment features to csv file
    results_folder = "/home/bastian/git/vana/data/vis/vesselvio"
    save_graph = True  # Save reduced graph export?
    verbose = True

    gen_options = IC.AnalysisOptions(
        results_folder,
        resolution,
        prune_length,
        filter_length,
        150,
        save_segment_results,
        save_graph,
        image_dimensions,
    )
    # endregion

    #####################
    ### RUN THIS FILE ###
    #####################
    # Use this key in place of 'ann_options' if you aren't analyzing annotated datasets.
    no_annotation = IC.AnnotationOptions(None, None, "None", None)
    # process_volume(compiler_file, gen_options, no_annotation, vis_options, 0, verbose)

    ######################
    ### Run files here ###
    ######################
    path_to_sample = Path("/home/bastian/git/vana/data/vis/vesselvio/liver_87")
    # path_to_sample = Path("/mnt/hdd0/bastian/MSD3_cads_iso/liver_48")


    iteration = 0

    # Use "no_anno" in place of "anno_options" if there are no annotations
    process_volume(path_to_sample, gen_options, no_annotation, vis_options, iteration, verbose)
    
    ### Graph files
    # Follow the format below to load csv-based graphs.
    vertices = "vertices.csv"
    edges = "edges.csv"
    graph0 = {"Vertices": vertices, "Edges": edges}

    # iGraph compatible format
    graph1 = "example.graphml"

    # process_graph(graph0, gen_options, graph_options, vis_options, verbose)
