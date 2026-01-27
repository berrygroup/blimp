from typing import Union, Optional, List
from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import mahotas as mh

from blimp.utils import get_channel_names

logger = logging.getLogger(__name__)


def segment_nuclei_cellpose(
    intensity_image: AICSImage,
    nuclei_channel: int = 0,
    pretrained_model: Union[str, Path, None] = None,
    diameter: Optional[int] = None,
    threshold: float = 0,
    flow_threshold: float = 0.4,
    normalize: Union[bool, dict] = True,
    gpu: bool = False,
) -> AICSImage:
    """Segment nuclei in 2D images across all timepoints using cellpose 4.

    Parameters
    ----------
    intensity_image
        intensity image in 5D format "TCZYX" where Z=1
    nuclei_channel
        channel number corresponding to nuclear stain
    pretrained_model
        path to custom pretrained model, if None uses default "cpsam" model
    diameter
        estimated diameter of nuclei in pixels, if None cellpose estimates
    threshold
        cellprob_threshold, float between [-6,+6] after which objects are discarded
    flow_threshold
        flow error threshold for filtering masks
    normalize
        normalization settings, can be bool or dict of parameters
    gpu
        whether to use GPU acceleration, by default False

    Returns
    -------
    AICSImage
        label image with segmented nuclei for all timepoints
        
    Raises
    ------
    ValueError
        If input image has Z dimension > 1 (3D images not supported)
    """
    from cellpose import models

    # Check that input is 2D only
    if intensity_image.dims.Z > 1:
        raise ValueError(
            f"segment_nuclei_cellpose only supports 2D images (Z=1). "
            f"Input image has Z={intensity_image.dims.Z}. "
            f"For 3D segmentation, use cellpose with do_3D=True directly."
        )

    # Initialize model once for all timepoints
    if pretrained_model is None:
        logger.debug("Initializing cellpose with default cpsam model")
        cellpose_model = models.CellposeModel(gpu=gpu)
    else:
        logger.debug(f"Initializing cellpose with pretrained model {str(pretrained_model)}")
        cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model=str(pretrained_model))

    # Segment all timepoints
    all_masks = []
    n_timepoints = intensity_image.dims.T
    
    for t in range(n_timepoints):
        logger.debug(f"Segmenting nuclei at timepoint {t}/{n_timepoints-1}")
        
        # Extract single 2D image
        nuclei_image = intensity_image.get_image_data("YX", C=nuclei_channel, T=t, Z=0)
        
        # Add channel dimension (YX -> CYX)
        # Cellpose's convert_image will automatically convert to 3 channels
        nuclei_image_with_channel = nuclei_image[np.newaxis, :, :]
        
        # Run segmentation
        # Note: cellpose 4 cpsam model returns 3 values (masks, flows, styles)
        results = cellpose_model.eval(
            nuclei_image_with_channel,
            channel_axis=0,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=threshold,
            normalize=normalize,
            do_3D=False,
        )
        
        # Extract masks (first element of returned tuple)
        masks = results[0]
        all_masks.append(masks)

    # Stack all timepoints and convert to AICSImage format (add C and Z dimensions)
    masks_stack = np.stack(all_masks)[:, np.newaxis, np.newaxis, :, :]
    segmentation = AICSImage(
        masks_stack,
        channel_names=["Nuclei"],
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    return segmentation


def expand_objects_watershed(
    seeds_image: np.ndarray, background_image: np.ndarray, intensity_image: np.ndarray
) -> np.ndarray:
    """Expand objects.

    Expands objects in `seeds_image` using a watershed transform
    on `intensity_image`.

    Parameters
    ----------
    seeds_image:
        objects that should be expanded
    background_image:
        regions in the image that should be considered background and should
        not be part of an object after expansion
    intensity_image:
        grayscale image; pixel intensities determine how far individual
        objects are expanded

    Returns
    -------
    numpy.ndarray
        expanded objects
    """
    # We compute the watershed transform using the seeds of the primary
    # objects and the additional seeds for the background regions. The
    # background regions will compete with the foreground regions and
    # thereby work as a stop criterion for expansion of primary objects.

    labels = np.where(seeds_image != 0, seeds_image, background_image)
    regions = mh.cwatershed(np.invert(intensity_image), labels)

    # Remove background regions
    n_objects = len(np.unique(seeds_image[seeds_image > 0]))
    regions[regions > n_objects] = 0

    # Ensure objects are separated
    lines = mh.labeled.borders(regions)
    regions[lines] = 0

    # Close holes in objects.
    foreground_mask = regions > 0
    holes = np.logical_xor(mh.close_holes(foreground_mask), foreground_mask)
    holes = mh.morph.dilate(holes)
    holes_labeled, n_holes = mh.label(holes)
    for i in range(1, n_holes + 1):
        fill_value = np.unique(regions[holes_labeled == i])[-1]
        fill_value = fill_value[fill_value > 0][0]
        regions[holes_labeled == i] = fill_value

    # Remove objects that are obviously too small, i.e. smaller than any of
    # the seeds (this could happen when we remove certain parts of objects
    # after the watershed region growing)
    primary_sizes = mh.labeled.labeled_size(seeds_image)
    if len(primary_sizes) > 1:
        min_size = np.min(primary_sizes[1:]) + 1
        regions = mh.labeled.filter_labeled(regions, min_size=min_size)[0]

    # Remove regions that don't overlap with seed objects and assign
    # correct labels to the other regions, i.e. those of the corresponding seeds.
    new_label_image, n_new_labels = mh.labeled.relabel(regions)
    lut = np.zeros(np.max(new_label_image) + 1, new_label_image.dtype)
    for i in range(1, n_new_labels + 1):
        orig_labels = seeds_image[new_label_image == i]
        orig_labels = orig_labels[orig_labels > 0]
        orig_count = np.bincount(orig_labels)
        orig_unique = np.where(orig_count)[0]
        if orig_unique.size == 1:
            lut[i] = orig_unique[0]
        elif orig_unique.size > 1:
            logger.warning("objects overlap after expansion: %s", ", ".join(map(str, orig_unique)))
            lut[i] = np.where(orig_count == np.max(orig_count))[0][0]
    expanded_image = lut[new_label_image]

    # Ensure that seed objects are fully contained within expanded objects
    index = (seeds_image - expanded_image) > 0
    expanded_image[index] = seeds_image[index]

    return expanded_image


def segment_secondary(
    primary_label_image: np.ndarray,
    intensity_image: np.ndarray,
    contrast_threshold: float,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
) -> np.ndarray:
    """Segment Secondary.

    Detects secondary objects in an image by expanding the primary objects
    encoded in `primary_label_image`. The outlines of secondary objects are
    determined based on the watershed transform of `intensity_image` using the
    primary objects in `primary_label_image` as seeds.

    Parameters
    ----------
    primary_label_image: numpy.ndarray[numpy.int32]
        2D labeled array encoding primary objects, which serve as seeds for
        watershed transform
    intensity_image: numpy.ndarray[numpy.uint8 or numpy.uint16]
        2D grayscale array that serves as gradient for watershed transform;
        optimally this image is enhanced with a low-pass filter
    contrast_threshold: int
        contrast threshold for automatic separation of forground from background
        based on locally adaptive thresholding (when ``0`` threshold defaults
        to `min_threshold` manual thresholding)
    min_threshold: int, optional
        minimal foreground value; pixels below `min_threshold` are considered
        background
    max_threshold: int, optional
        maximal foreground value; pixels above `max_threshold` are considered
        foreground
    plot: bool, optional
        whether a plot should be generated

    Returns
    -------
    numpy.ndarray
        secondary_label_image

    Note
    ----
    Setting `min_threshold` and `max_threshold` to the same value reduces
    to manual thresholding.
    """
    if np.any(primary_label_image == 0):
        has_background = True
    else:
        has_background = False

    if not has_background:
        secondary_label_image = primary_label_image
    else:
        # We use adaptive thresholding to determine background regions,
        # i.e. regions in the intensity_image that should not be covered by
        # secondary objects.
        n_objects = len(np.unique(primary_label_image))
        logger.info(f"primary label image has {n_objects -1} objects")

        if np.max(primary_label_image) != n_objects - 1:
            raise ValueError(f"Objects are not consecutively labeled, please relabel before secondary segmentation.")
        # SB: Added a catch for images with no primary objects
        # note that background is an 'object'
        if n_objects > 1:
            background_mask = mh.thresholding.bernsen(intensity_image, 5, contrast_threshold)
            if min_threshold is not None:
                logger.info(f"set lower threshold level to {min_threshold}")
                background_mask[intensity_image < min_threshold] = True

            if max_threshold is not None:
                logger.info(f"set upper threshold level to {max_threshold}")
                background_mask[intensity_image > max_threshold] = False
            background_label_image = (mh.label(background_mask)[0] > 0).astype(np.int32)
            if n_objects >= 2147483646:
                raise ValueError(f"Number of objects ({n_objects}) exceeds 32-bit datatype.")
            background_label_image[background_mask] += n_objects

            logger.info("detect secondary objects via watershed transform")
            secondary_label_image = expand_objects_watershed(
                primary_label_image, background_label_image, intensity_image
            )
        else:
            logger.info("skipping secondary segmentation")
            secondary_label_image = np.zeros(primary_label_image.shape, dtype=np.int32)

    n_objects = len(np.unique(secondary_label_image)[1:])
    logger.info("identified {n_objects} objects")

    return secondary_label_image



def resolve_multi_parent_objects(
    label_image: AICSImage, 
    measure_object: Optional[Union[int, str, List[Union[int, str]]]] = None, 
    parent_object: Union[int, str] = 0, 
    timepoint: int = 0, 
    in_place: bool = True
) -> AICSImage | None:
    """
    Resolve child objects that span multiple parent objects by removing pixels
    to ensure each child object is fully contained within a single parent.
    
    When a child object overlaps with multiple parent objects, pixels are assigned
    to the parent with which the child object has the largest overlap.
    
    Parameters
    ----------
    label_image
        The labeled image containing objects in separate channels.
    measure_object
        The child object(s) to be resolved. Can be channel index, channel name, or list of indices/names.
        If None (default), resolve conflicts for all channels except parent_object.
    parent_object
        The parent object channel, can be index or channel name, by default 0.
    timepoint
        Timepoint at which to resolve objects, by default 0.
    in_place
        If True, modify the input label_image in place. If False, return a new
        AICSImage with resolved objects, by default True.
        
    Returns
    -------
    AICSImage | None
        If in_place=False, returns a new AICSImage with resolved child objects.
        If in_place=True, returns None and modifies the input label_image.
    """

    # Convert parent_object to channel name and index
    parent_object_name = get_channel_names(label_image, parent_object)[0]
    parent_object_index = label_image.channel_names.index(parent_object_name)

    # Determine which channels to process
    if measure_object is None:
        # Process all channels except parent_object_index
        measure_indices = [i for i in range(label_image.dims.C) if i != parent_object_index]
        measure_names = [label_image.channel_names[i] for i in measure_indices]
        logger.info(f"Resolving multi-parent conflicts for all objects except parent object {parent_object_name} (index {parent_object_index}): {measure_names}")
    else:
        # Convert measure_object to list of names and indices
        measure_names = get_channel_names(label_image, measure_object)
        measure_indices = [label_image.channel_names.index(name) for name in measure_names]
        
        # Check if any measure object is the same as parent object
        if parent_object_index in measure_indices:
            logger.warning(f"Parent object '{parent_object_name}' is also in measure_objects. Skipping it.")
            measure_indices = [i for i in measure_indices if i != parent_object_index]
            measure_names = [label_image.channel_names[i] for i in measure_indices]
        
        if not measure_indices:
            logger.warning("No valid measure objects to process after removing parent object.")
            return None
    
    # If not in_place, create a copy of the data
    if not in_place:
        new_label_stack = label_image.data.copy()
    
    # Process each measure channel
    for current_measure_index in measure_indices:
        _resolve_single_measure_object(
            label_image, current_measure_index, parent_object_index, timepoint, in_place, new_label_stack if not in_place else None
        )
    
    # Return new AICSImage if not in_place, otherwise return None
    if not in_place:
        resolved_label_image = AICSImage(
            new_label_stack,
            channel_names=label_image.channel_names,
            physical_pixel_sizes=label_image.physical_pixel_sizes
        )
        return resolved_label_image
    
    return None


def _resolve_single_measure_object(
    label_image: AICSImage, measure_object_index: int, parent_object_index: int, timepoint: int, in_place: bool, new_label_stack: Optional[np.ndarray] = None
) -> None:
    """
    Helper function to resolve multi-parent conflicts for a single measure object channel.
    
    Parameters
    ----------
    label_image
        The labeled image containing objects in separate channels.
    measure_object_index
        Index of the channel containing child objects to be resolved.
    parent_object_index
        Index of the channel containing parent objects.
    timepoint
        Timepoint at which to resolve objects.
    in_place
        If True, modify the input label_image in place.
    new_label_stack
        If not in_place, the copied data array to modify.
    """

    # Get the appropriate arrays based on dimensionality
    if label_image.dims.Z == 1:
        logger.debug(f"Processing channel {measure_object_index} ({label_image.channel_names[measure_object_index]}) in 2D.")
        label_array = label_image.get_image_data("YX", C=measure_object_index, T=timepoint, Z=0).copy()
        parent_label_array = label_image.get_image_data("YX", C=parent_object_index, T=timepoint, Z=0)
        is_2d = True
    elif label_image.dims.Z > 1:
        logger.debug(f"Processing channel {measure_object_index} ({label_image.channel_names[measure_object_index]}) in 3D ({label_image.dims.Z} Z-planes).")
        label_array = label_image.get_image_data("ZYX", C=measure_object_index, T=timepoint).copy()
        parent_label_array = label_image.get_image_data("ZYX", C=parent_object_index, T=timepoint)
        is_2d = False
    
    # Find all unique child object labels
    child_labels = np.unique(label_array[label_array > 0])
    conflicts_resolved = 0
    
    # Skip child objects that are only a single pixel
    child_sizes = np.bincount(label_array.ravel())[child_labels]
    valid_child_labels = child_labels[child_sizes > 1]

    # Process each valid child object
    for child_label in valid_child_labels:
        # Get mask for current child object
        child_mask = label_array == child_label
        
        # Find all parent labels that overlap with this child
        overlapping_parents = np.unique(parent_label_array[child_mask])
        overlapping_parents = overlapping_parents[overlapping_parents > 0]  # Remove background
        
        if len(overlapping_parents) > 1:
            # Child spans multiple parents - need to resolve
            conflicts_resolved += 1
            logger.debug(f"Resolving child object {child_label} spanning {len(overlapping_parents)} parents: {overlapping_parents}")
            
            # Calculate overlap counts
            overlap_counts = np.zeros(len(overlapping_parents), dtype=np.int64)
            for i, parent_label in enumerate(overlapping_parents):
                overlap_counts[i] = np.sum(child_mask & (parent_label_array == parent_label))
            
            # Find parent with largest overlap
            best_parent_idx = np.argmax(overlap_counts)
            best_parent = overlapping_parents[best_parent_idx]
            overlap_count = overlap_counts[best_parent_idx]
            
            logger.debug(f"Assigning to parent {best_parent} (overlap: {overlap_count} pixels)")
            
            # Remove child pixels that don't belong to the best parent
            remove_mask = child_mask & (parent_label_array != best_parent)
            label_array[remove_mask] = 0
    
    logger.info(f"Resolved {conflicts_resolved} multi-parent conflicts for {label_image.channel_names[measure_object_index]} objects")
    
    # Update the label image data
    if is_2d:
        # For 2D, update the specific slice
        if in_place:
            label_image.data[timepoint, measure_object_index, 0, :, :] = label_array
        else:
            new_label_stack[timepoint, measure_object_index, 0, :, :] = label_array
    else:
        # For 3D, update the entire volume
        if in_place:
            label_image.data[timepoint, measure_object_index, :, :, :] = label_array
        else:
            new_label_stack[timepoint, measure_object_index, :, :, :] = label_array


def mask_child_objects_by_parent(
    label_image: AICSImage, 
    measure_object: Optional[Union[int, str, List[Union[int, str]]]] = None, 
    parent_object: Union[int, str] = 0, 
    timepoint: int = 0, 
    in_place: bool = True
) -> AICSImage | None:
    """
    Mask child objects by parent objects, removing any pixels that extend 
    beyond parent boundaries.
    
    This function masks (sets to zero) any parts of child objects that extend 
    outside their parent objects, ensuring all child objects are fully contained 
    within parent boundaries. This is useful for enforcing parent-child 
    relationships in multi-channel segmentation data.
    
    Parameters
    ----------
    label_image
        The labeled image containing objects in separate channels.
    measure_object
        The child object(s) to be masked. Can be channel index, channel name, 
        or list of indices/names. If None (default), mask all channels except parent_object.
    parent_object
        The parent object channel used as a mask, can be index or channel name, by default 0.
    timepoint
        Timepoint at which to mask objects, by default 0.
    in_place
        If True, modify the input label_image in place. If False, return a new
        AICSImage with masked objects, by default True.
        
    Returns
    -------
    AICSImage | None
        If in_place=False, returns a new AICSImage with masked child objects.
        If in_place=True, returns None and modifies the input label_image.
        
    Examples
    --------
    >>> # Mask all objects to be within cell boundaries
    >>> masked_labels = mask_child_objects_by_parent(
    ...     label_image, 
    ...     parent_object='Cell',
    ...     in_place=False
    ... )
    
    >>> # Mask specific organelles to be within nuclei
    >>> mask_child_objects_by_parent(
    ...     label_image,
    ...     measure_object=['Organelle1', 'Organelle2'], 
    ...     parent_object='Nucleus'
    ... )
    """

    # Convert parent_object to channel name and index
    parent_object_name = get_channel_names(label_image, parent_object)[0]
    parent_object_index = label_image.channel_names.index(parent_object_name)

    # Determine which channels to process
    if measure_object is None:
        # Process all channels except parent_object_index
        measure_indices = [i for i in range(label_image.dims.C) if i != parent_object_index]
        measure_names = [label_image.channel_names[i] for i in measure_indices]
        logger.info(f"Masking child objects by parent for all objects except parent object {parent_object_name} (index {parent_object_index}): {measure_names}")
    else:
        # Convert measure_object to list of names and indices
        measure_names = get_channel_names(label_image, measure_object)
        measure_indices = [label_image.channel_names.index(name) for name in measure_names]
        
        # Check if any measure object is the same as parent object
        if parent_object_index in measure_indices:
            logger.warning(f"Parent object '{parent_object_name}' is also in measure_objects. Skipping it.")
            measure_indices = [i for i in measure_indices if i != parent_object_index]
            measure_names = [label_image.channel_names[i] for i in measure_indices]
        
        if not measure_indices:
            logger.warning("No valid measure objects to process after removing parent object.")
            return None
    
    # If not in_place, create a copy of the data
    if not in_place:
        new_label_stack = label_image.data.copy()
    
    # Process each measure channel
    for current_measure_index in measure_indices:
        _mask_single_measure_object_by_parent(
            label_image, current_measure_index, parent_object_index, timepoint, in_place, new_label_stack if not in_place else None
        )
    
    # Return new AICSImage if not in_place, otherwise return None
    if not in_place:
        masked_label_image = AICSImage(
            new_label_stack,
            channel_names=label_image.channel_names,
            physical_pixel_sizes=label_image.physical_pixel_sizes
        )
        return masked_label_image
    
    return None


def _mask_single_measure_object_by_parent(
    label_image: AICSImage, measure_object_index: int, parent_object_index: int, timepoint: int, in_place: bool, new_label_stack: Optional[np.ndarray] = None
) -> None:
    """
    Helper function to mask a single child object channel by parent objects.
    
    Removes pixels from child objects that extend outside their parent boundaries,
    ensuring child objects are fully contained within parent objects.
    
    Parameters
    ----------
    label_image
        The labeled image containing objects in separate channels.
    measure_object_index
        Index of the channel containing child objects to be masked.
    parent_object_index
        Index of the channel containing parent objects used as masks.
    timepoint
        Timepoint at which to mask objects.
    in_place
        If True, modify the input label_image in place.
    new_label_stack
        If not in_place, the copied data array to modify.
    """

    # Get the appropriate arrays based on dimensionality
    if label_image.dims.Z == 1:
        logger.debug(f"Masking object channel {measure_object_index} ({label_image.channel_names[measure_object_index]}) by parent in 2D.")
        label_array = label_image.get_image_data("YX", C=measure_object_index, T=timepoint, Z=0).copy()
        parent_label_array = label_image.get_image_data("YX", C=parent_object_index, T=timepoint, Z=0)
        is_2d = True
    elif label_image.dims.Z > 1:
        logger.debug(f"Masking object channel {measure_object_index} ({label_image.channel_names[measure_object_index]}) by parent in 3D ({label_image.dims.Z} Z-planes).")
        label_array = label_image.get_image_data("ZYX", C=measure_object_index, T=timepoint).copy()
        parent_label_array = label_image.get_image_data("ZYX", C=parent_object_index, T=timepoint)
        is_2d = False
    
    # Count objects before masking
    initial_objects = len(np.unique(label_array[label_array > 0]))
    initial_pixels = np.sum(label_array > 0)
    
    # Set all child object pixels outside parent objects to zero (mask them)
    outside_parent_mask = (label_array > 0) & (parent_label_array == 0)
    label_array[outside_parent_mask] = 0
    
    # Count objects and pixels after masking
    final_objects = len(np.unique(label_array[label_array > 0]))
    final_pixels = np.sum(label_array > 0)
    removed_pixels = initial_pixels - final_pixels
    
    logger.info(f"Masked {label_image.channel_names[measure_object_index]} objects by parent: "
                f"{initial_objects} -> {final_objects} objects, removed {removed_pixels} pixels outside parent")
    
    # Update the label image data
    if is_2d:
        # For 2D, update the specific slice
        if in_place:
            label_image.data[timepoint, measure_object_index, 0, :, :] = label_array
        else:
            new_label_stack[timepoint, measure_object_index, 0, :, :] = label_array
    else:
        # For 3D, update the entire volume
        if in_place:
            label_image.data[timepoint, measure_object_index, :, :, :] = label_array
        else:
            new_label_stack[timepoint, measure_object_index, :, :, :] = label_array