from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import pandas as pd
import pytest

from blimp.constants import blimp_config
import blimp.processing.quantify

from .helpers import _ensure_test_data  # noqa: F401, I252
from .helpers import _load_test_data

logger = logging.getLogger(__name__)


def test_measure_parent_object_label(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image, label_image = _load_test_data("synthetic_2D")

    with pytest.raises(ValueError):
        # check for ValueError when assignment of measure objects to parent is ambiguous
        blimp.processing.quantify._measure_parent_object_label(
            label_image=label_image, measure_object_index=2, parent_object_index=0, timepoint=0
        )

    current_res = blimp.processing.quantify._measure_parent_object_label(
        label_image=label_image, measure_object_index=1, parent_object_index=0, timepoint=0
    )

    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    current_res.to_csv(Path(testdata_config.RESULTS_DIR) / "measure_parent_label_2D_results_t_0.csv")

    true_res_path = Path(testdata_config.RESOURCES_DIR) / "measure_parent_label_2D_results_t_0.csv"
    true_res = pd.read_csv(true_res_path, index_col=0)

    pd.testing.assert_frame_equal(current_res, true_res, check_dtype=False)


def test_quantify_single_timepoint_2D_no_parent(_ensure_test_data):
    blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # no parent object
    res_obj1 = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D, label_image=label_image_2D, measure_object=0, timepoint=0
    )

    assert res_obj1.Object1_intensity_max_Channel1.to_list() == [1000.0, 2000.0, 3000.0, 4000.0]
    assert res_obj1.Object1_area.to_list() == [100.0, 81.0, 900.0, 961.0]

    res_obj2 = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D, label_image=label_image_2D, measure_object=1, timepoint=0
    )

    assert res_obj2.count().label == 100
    assert list(np.unique(res_obj2.Object2_area)) == [4.0]

    with pytest.raises(AttributeError):
        # check no parent label when parent_label is none
        res_obj2.parent_label


def test_quantify_single_timepoint_2D_with_parent(_ensure_test_data):
    blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # parent object
    res_obj1_parent1 = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D, label_image=label_image_2D, measure_object=0, parent_object=0, timepoint=0
    )

    # check the parent object labels are correct when parent = measure
    assert res_obj1_parent1.parent_label.to_list() == [1, 2, 3, 4]

    with pytest.raises(ValueError):
        # check for ValueError when assignment of measure objects to parent is ambiguous
        res_obj3_parent1 = blimp.processing.quantify._quantify_single_timepoint_2D(
            intensity_image=intensity_image_2D,
            label_image=label_image_2D,
            measure_object=2,
            parent_object=0,
            timepoint=0,
        )

    # quantify object channel 1 relative to parent object channel 0
    res_obj2_parent1 = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D, label_image=label_image_2D, measure_object=1, parent_object=0, timepoint=0
    )

    assert list(np.unique(res_obj2_parent1.parent_label_name)) == ["Object1"]

    # check the number of objects in each parent object is correct
    assert res_obj2_parent1.query("parent_label != 0").groupby("parent_label").size().to_list() == [5, 4, 21, 20]


def test_quantify_single_timepoint_2D_intensity_channels_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj2_intensity1_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=1,
        timepoint=0,
        intensity_channels="Channel1",
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_str_input.Object2_intensity_min_Channel2

    assert list(np.unique(res_obj2_intensity1_str_input.Object2_intensity_min_Channel1)) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]


def test_quantify_single_timepoint_2D_intensity_channels_list1_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj2_intensity1_list1_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=1,
        timepoint=0,
        intensity_channels=["Channel1"],
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_list1_str_input.Object2_intensity_min_Channel2

    assert list(np.unique(res_obj2_intensity1_list1_str_input.Object2_intensity_min_Channel1)) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]


def test_quantify_single_timepoint_2D_intensity_channels_list2_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj2_intensity1_list2_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=1,
        timepoint=0,
        intensity_channels=["Channel1", "Channel2"],
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_list2_str_input.Object2_intensity_min_Channel3

    assert list(np.unique(res_obj2_intensity1_list2_str_input.Object2_intensity_min_Channel2)) == [0, 5000.0]


def test_quantify_single_timepoint_2D_intensity_channels_list_int_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj2_intensity1_list_int_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=1,
        timepoint=0,
        intensity_channels=[0, 1],
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_list_int_input.Object2_intensity_min_Channel3

    assert list(np.unique(res_obj2_intensity1_list_int_input.Object2_intensity_min_Channel1)) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]
    assert list(np.unique(res_obj2_intensity1_list_int_input.Object2_intensity_min_Channel2)) == [0, 5000.0]


def test_quantify_single_timepoint_2D_texture_channels_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj1_texture1_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels="Channel1",
    )

    assert res_obj1_texture1_str_input["Object1_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_str_input["Object1_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_2D_texture_channels_list1_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj1_texture1_list1_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=["Channel1"],
    )

    assert res_obj1_texture1_list1_str_input["Object1_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list1_str_input["Object1_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_2D_texture_channels_list2_str_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj1_texture1_list2_str_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=["Channel1", "Channel2"],
    )

    assert res_obj1_texture1_list2_str_input["Object1_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert res_obj1_texture1_list2_str_input["Object1_Channel2_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list2_str_input["Object1_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_2D_texture_channels_list_int_input(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_obj1_texture1_list_int_input = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=[0, 1],
    )

    assert res_obj1_texture1_list_int_input["Object1_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert res_obj1_texture1_list_int_input["Object1_Channel2_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list_int_input["Object1_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_2D_no_border_objects(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_no_border = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_object=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert not any(res_no_border.Object1_is_border)


def test_quantify_single_timepoint_2D_one_border_object(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_border = blimp.processing.quantify._quantify_single_timepoint_2D(
        intensity_image=AICSImage(
            intensity_image_2D.data[:, :, :, 52:, 52:], channel_names=intensity_image_2D.channel_names
        ),
        label_image=AICSImage(label_image_2D.data[:, :, :, 52:, 52:], channel_names=label_image_2D.channel_names),
        measure_object=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res_border.query("label==1").Object1_is_border.iloc[0] == True


def test_quantify_single_timepoint_3D_no_parent(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1 = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D, label_image=label_image_3D, measure_object=0, timepoint=0
    )

    assert res_obj1["Object1_3D_MIP_area"].to_list() == [10.0**2, 9.0**2, 30.0**2, 46.0**2]
    assert res_obj1["Object1_3D_number_of_voxels"].to_list() == [10.0**3, 9.0**3, 30.0**3, 46.0**3]
    assert res_obj1["Object1_3D_Middle_perimeter"].to_list() == [10.0 * 4 - 4, 9.0 * 4 - 4, 30.0 * 4 - 4, 46.0 * 4 - 4]

    res_obj2 = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D, label_image=label_image_3D, measure_object=1, timepoint=0
    )

    assert res_obj2.count().label == 100
    assert list(np.unique(res_obj2["Object2_3D_number_of_voxels"].to_list())) == [2.0**3]

    with pytest.raises(AttributeError):
        # check no parent label when parent_label is none
        res_obj2.parent_label


def test_quantify_single_timepoint_3D_with_parent(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1_parent1 = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D, label_image=label_image_3D, measure_object=0, parent_object=0, timepoint=0
    )

    # check the parent object labels are correct when parent = measure
    assert res_obj1_parent1.parent_label.to_list() == [1, 2, 3, 4]

    with pytest.raises(ValueError):
        # check for ValueError when assignment of measure objects to parent is ambiguous
        res_obj3_parent1 = blimp.processing.quantify._quantify_single_timepoint_3D(
            intensity_image=intensity_image_3D,
            label_image=label_image_3D,
            measure_object=2,
            parent_object=0,
            timepoint=0,
        )

    res_obj2_parent1 = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D, label_image=label_image_3D, measure_object=1, parent_object=0, timepoint=0
    )

    assert list(np.unique(res_obj2_parent1.parent_label_name)) == ["Object1"]

    # check the number of objects in each parent object is correct
    assert res_obj2_parent1.query("parent_label != 0").groupby("parent_label").size().to_list() == [7, 5, 15, 13]


def test_quantify_single_timepoint_3D_intensity_channels_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj2_intensity1_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=1,
        timepoint=0,
        intensity_channels="Channel1",
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_str_input.Object2_intensity_min_Channel2

    assert list(np.unique(res_obj2_intensity1_str_input["Object2_3D_intensity_min_Channel1"].to_list())) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]


def test_quantify_single_timepoint_3D_intensity_channels_list1_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj2_intensity1_list1_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=1,
        timepoint=0,
        intensity_channels=["Channel1"],
    )

    with pytest.raises(AttributeError):
        res_obj2_intensity1_list1_str_input.Object2_intensity_min_Channel2

    assert list(np.unique(res_obj2_intensity1_list1_str_input["Object2_3D_intensity_min_Channel1"].to_list())) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]


def test_quantify_single_timepoint_3D_intensity_channels_list2_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj2_intensity1_list2_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=1,
        timepoint=0,
        intensity_channels=["Channel1", "Channel2"],
    )

    with pytest.raises(KeyError):
        res_obj2_intensity1_list2_str_input["Object2_3D_intensity_min_Channel3"].to_list()

    assert list(np.unique(res_obj2_intensity1_list2_str_input["Object2_3D_intensity_min_Channel2"].to_list())) == [
        5000.0,
        6000.0,
    ]


def test_quantify_single_timepoint_3D_intensity_channels_list_int_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj2_intensity1_list_int_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=1,
        timepoint=0,
        intensity_channels=[0, 1],
    )

    with pytest.raises(KeyError):
        res_obj2_intensity1_list_int_input["Object2_3D_intensity_min_Channel3"].to_list()

    assert list(np.unique(res_obj2_intensity1_list_int_input["Object2_3D_intensity_min_Channel1"].to_list())) == [
        0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]
    assert list(np.unique(res_obj2_intensity1_list_int_input["Object2_3D_intensity_min_Channel2"].to_list())) == [
        5000.0,
        6000.0,
    ]


def test_quantify_single_timepoint_3D_texture_channels_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1_texture1_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels="Channel1",
    )

    assert res_obj1_texture1_str_input["Object1_3D_MIP_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_str_input["Object1_3D_MIP_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_3D_texture_channels_list1_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1_texture1_list1_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=["Channel1"],
    )

    assert res_obj1_texture1_list1_str_input["Object1_3D_MIP_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list1_str_input["Object1_3D_MIP_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_3D_texture_channels_list2_str_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1_texture1_list2_str_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=["Channel1", "Channel2"],
    )

    assert res_obj1_texture1_list2_str_input["Object1_3D_MIP_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    # assert res_obj1_texture1_list2_str_input[
    #     "Object1_3D_Middle_Channel2_Haralick-angular-second-moment-1"
    # ].to_list() == [
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    # ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list2_str_input["Object1_3D_Middle_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_3D_texture_channels_list_int_input(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_obj1_texture1_list_int_input = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=0,
        timepoint=0,
        calculate_textures=True,
        texture_channels=[0, 1],
    )

    assert res_obj1_texture1_list_int_input["Object1_3D_MIP_Channel1_Haralick-angular-second-moment-1"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    # assert res_obj1_texture1_list_int_input[
    #     "Object1_3D_Middle_Channel2_Haralick-angular-second-moment-1"
    # ].to_list() == [
    #     1.0,
    #     1.0,
    #     1.0,
    #     1.0,
    # ]
    with pytest.raises(KeyError):
        res_obj1_texture1_list_int_input["Object1_3D_Middle_Channel3_Haralick-angular-second-moment-1"]


def test_quantify_single_timepoint_3D_no_border_objects(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res_no_border = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_object=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert not any(res_no_border["Object1_3D_is_border"].to_list())


def test_quantify_single_timepoint_3D_one_border_object(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    # test one border object (crop image to generate a border object)
    res_border_all = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=AICSImage(
            intensity_image_3D.data[:, :, :, 8:, 8:],
            channel_names=intensity_image_3D.channel_names,
            physical_pixel_sizes=intensity_image_3D.physical_pixel_sizes,
        ),
        label_image=AICSImage(
            label_image_3D.data[:, :, :, 8:, 8:],
            channel_names=label_image_3D.channel_names,
            physical_pixel_sizes=label_image_3D.physical_pixel_sizes,
        ),
        measure_object=0,
        timepoint=0,
        intensity_channels=0,
    )

    assert res_border_all.query("label==1")["Object1_3D_is_border"].any()
    assert res_border_all.query("label==1")["Object1_3D_is_border_XY"].any()

    res_border_Z_only = blimp.processing.quantify._quantify_single_timepoint_3D(
        intensity_image=AICSImage(
            intensity_image_3D.data[:, :, 8:, :, :],
            channel_names=intensity_image_3D.channel_names,
            physical_pixel_sizes=intensity_image_3D.physical_pixel_sizes,
        ),
        label_image=AICSImage(
            label_image_3D.data[:, :, 8:, :, :],
            channel_names=label_image_3D.channel_names,
            physical_pixel_sizes=label_image_3D.physical_pixel_sizes,
        ),
        measure_object=0,
        timepoint=0,
        intensity_channels=0,
    )

    assert res_border_Z_only.query("label==1")["Object1_3D_is_border"].any()
    assert not res_border_Z_only.query("label==1")["Object1_3D_is_border_XY"].any()


def test_quantify_no_parent(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res[0].Object1_intensity_max_Channel1.to_list() == [1000.0, 2000.0, 3000.0, 4000.0]
    assert res[0].Object1_area.to_list() == [100.0, 81.0, 900.0, 961.0]


def test_quantify_with_parent(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=0,
        parent_object=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res[0].parent_label.to_list() == [1, 2, 3, 4]


def test_quantify_multiple_objects(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=[0, 1],
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res[0].Object1_intensity_max_Channel1.to_list() == [1000.0, 2000.0, 3000.0, 4000.0]
    assert len(res[1].Object2_intensity_max_Channel1.to_list()) == 100


def test_quantify_all_objects_all_times_no_parent(_ensure_test_data):
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
    )

    assert len(res) == 3
    with pytest.raises(AttributeError):
        res[0].parent_label

    assert res[0].query("TimepointID==3").Object1_intensity_max_Channel1.to_list() == [1000.0, 2000.0, 3000.0, 4000.0]
    assert res[0].query("TimepointID==4").Object1_intensity_max_Channel1.to_list() == []

    true_res_path = Path(testdata_config.RESOURCES_DIR) / "quantify_2D_results_object2_t_all.csv"
    true_res = pd.read_csv(true_res_path, index_col=0)

    pd.testing.assert_frame_equal(res[1], true_res, check_dtype=False)


def test_quantify_aggregate(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res_agg = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=[0, 1],
        parent_object=0,
        aggregate=True,
        timepoint=0,
    )

    # check manual parent object labelling agrees with aggregate counts
    res_manual = blimp.processing.quantify._measure_parent_object_label(
        label_image=label_image_2D, measure_object_index=1, parent_object_index=0, timepoint=0
    )

    pd.testing.assert_series_equal(
        res_manual.groupby("parent_label").size()[1:].reset_index(drop=True).rename(None),
        res_agg["Object2_count"].reset_index(drop=True).rename(None),
    )

    assert res_agg.Object2_intensity_max_Channel1_mean.to_list() == [1000.0, 2000.0, 3000.0, 4000.0]
    assert res_agg.Object2_intensity_min_Channel2_max.to_list() == [0.0, 5000.0, 5000.0, 0.0]


def test_quantify_texture_features(_ensure_test_data):
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=0,
        timepoint=0,
        intensity_channels="Channel1",
        texture_channels="Channel1",
    )

    assert res[0]["Object1_Channel1_Haralick-angular-second-moment-1"].to_list() == [1.0, 1.0, 1.0, 1.0]


def test_quantify_3D_no_parent(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res[0]["Object1_3D_MIP_area"].to_list() == [10.0**2, 9.0**2, 30.0**2, 46.0**2]
    assert res[0]["Object1_3D_number_of_voxels"].to_list() == [10.0**3, 9.0**3, 30.0**3, 46.0**3]


def test_quantify_3D_with_parent(_ensure_test_data):
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=0,
        parent_object=0,
        timepoint=0,
        intensity_channels="Channel1",
    )

    assert res[0].parent_label.to_list() == [1, 2, 3, 4]
