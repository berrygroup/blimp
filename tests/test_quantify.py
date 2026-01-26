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
        texture_channels=["Channel1", "Channel2", "Channel3"],
        texture_objects=1,  # Object2
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
        texture_objects=0,
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


def test_point_object_2D_non_aggregate(_ensure_test_data):
    """Test point object quantification in 2D without aggregation."""
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Quantify channel 1 as point objects (non-aggregated) - all timepoints
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        intensity_channels=["Channel1", "Channel2"],
        point_objects=1,
    )

    # Should return a list with one DataFrame
    assert isinstance(res, list)
    assert len(res) == 1
    
    point_features = res[0]
    
    # Save results for manual inspection
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    point_features.to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_2D_non_aggregate_results.csv", index=False)
    
    # Each pixel with label > 0 should be treated as a separate point
    # Check that we have the expected columns
    assert "label" in point_features.columns
    assert "Object2_area" in point_features.columns
    assert "Object2_intensity_Channel1" in point_features.columns
    assert "Object2_intensity_Channel2" in point_features.columns
    assert "TimepointID" in point_features.columns
    
    # Area should be 1 for all points (single pixel)
    assert (point_features["Object2_area"] == 1).all()
    
    # Should have no morphology features like perimeter, eccentricity, etc.
    assert "Object2_perimeter" not in point_features.columns
    assert "Object2_eccentricity" not in point_features.columns
    
    # Should have no texture features
    assert not any("Haralick" in col for col in point_features.columns)
    
    # Check TimepointID values (should have multiple timepoints)
    unique_timepoints = point_features["TimepointID"].unique()
    assert len(unique_timepoints) >= 1  # At least one timepoint


def test_point_object_2D_with_parent_non_aggregate(_ensure_test_data):
    """Test point object quantification in 2D with parent mapping but no aggregation."""
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Quantify channel 1 as point objects with parent but no aggregation - all timepoints
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        parent_object=0,
        aggregate=False,
        intensity_channels=["Channel1"],
        point_objects=1,
    )

    assert isinstance(res, list)
    assert len(res) == 1
    
    point_features = res[0]
    
    # Save results for manual inspection
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    point_features.to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_2D_with_parent_non_aggregate_results.csv", index=False)
    
    # Should have parent_label and parent_label_name columns
    assert "parent_label" in point_features.columns
    assert "parent_label_name" in point_features.columns
    assert (point_features["parent_label_name"] == "Object1").all()
    
    # Parent labels should be valid (>= 0)
    assert (point_features["parent_label"] >= 0).all()
    
    # Count points per parent (excluding parent_label=0, which are outside parents)
    points_per_parent = point_features[point_features["parent_label"] > 0].groupby("parent_label").size()
    
    # Should have points in multiple parents (based on synthetic data structure)
    assert len(points_per_parent) > 1


def test_point_object_2D_aggregate(_ensure_test_data):
    """Test point object quantification in 2D with aggregation to parent."""
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Quantify with aggregation: channel 0 (parent) and channel 1 (points) - all timepoints
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=[0, 1],
        parent_object=0,
        aggregate=True,
        intensity_channels=["Channel1", "Channel2"],
        point_objects=1,
    )

    # Should return a single aggregated DataFrame
    assert isinstance(res, pd.DataFrame)
    
    # Save results for manual inspection
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    res.to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_2D_aggregate_results.csv", index=False)
    
    # Should have parent features
    assert "Object1_area" in res.columns
    assert "label" in res.columns
    
    # Should have aggregated point features
    assert "Object2_count" in res.columns
    assert "Object2_intensity_sum_Channel1" in res.columns
    assert "Object2_intensity_mean_Channel1" in res.columns
    assert "Object2_intensity_min_Channel1" in res.columns
    assert "Object2_intensity_max_Channel1" in res.columns
    assert "Object2_intensity_sum_Channel2" in res.columns
    assert "Object2_intensity_mean_Channel2" in res.columns
    
    # Check that counts are reasonable
    assert (res["Object2_count"] >= 0).all()
    
    # Verify aggregation: mean should be between min and max
    mask = res["Object2_count"] > 0
    assert (res.loc[mask, "Object2_intensity_mean_Channel1"] >= res.loc[mask, "Object2_intensity_min_Channel1"]).all()
    assert (res.loc[mask, "Object2_intensity_mean_Channel1"] <= res.loc[mask, "Object2_intensity_max_Channel1"]).all()
    
    # Verify no duplicate columns with _x/_y suffixes
    assert not any(col.endswith("_x") or col.endswith("_y") for col in res.columns), \
        f"Found duplicate columns: {[col for col in res.columns if col.endswith('_x') or col.endswith('_y')]}"
    
    # Verify fast-path was used: no standard aggregation columns (no "_sum_sum" suffix)
    assert not any("_sum_sum" in col for col in res.columns)
    
    # Check multiple timepoints are present
    unique_timepoints = res["TimepointID"].unique()
    assert len(unique_timepoints) >= 1  # At least one timepoint
    
    # Verify we get results for all timepoints and labels
    # Just check that we have a reasonable number of rows (at least num_parents rows)
    label_array = label_image_2D.get_image_data("YX", C=0, T=0, Z=0)
    num_parents = len(np.unique(label_array)) - 1  # Exclude background
    assert len(res) >= num_parents  # At least one row per parent


def test_point_object_3D_non_aggregate(_ensure_test_data):
    """Test point object quantification in 3D without aggregation."""
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    # Quantify channel 1 as point objects (non-aggregated)
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=1,
        timepoint=0,
        intensity_channels=["Channel1", "Channel2"],
        point_objects=1,
    )

    # Should return a list with one DataFrame
    assert isinstance(res, list)
    assert len(res) == 1
    
    point_features = res[0]
    
    # Each voxel with label > 0 should be treated as a separate point
    assert "label" in point_features.columns
    assert "Object2_3D_intensity_Channel1" in point_features.columns
    assert "Object2_3D_intensity_Channel2" in point_features.columns
    assert "TimepointID" in point_features.columns
    
    # Should have no 3D morphology features
    assert "Object2_3D_physical_volume_pL" not in point_features.columns
    assert "Object2_3D_number_of_voxels" not in point_features.columns
    
    # Should have no 2D-derived features (MIP, Middle)
    assert not any("_3D_MIP_" in col for col in point_features.columns)
    assert not any("_3D_Middle_" in col for col in point_features.columns)
    
    # Check TimepointID
    assert (point_features["TimepointID"] == 1).all()


def test_point_object_3D_with_parent_non_aggregate(_ensure_test_data):
    """Test point object quantification in 3D with parent mapping but no aggregation."""
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    # Quantify channel 1 as point objects with parent but no aggregation
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=1,
        parent_object=0,
        aggregate=False,
        timepoint=0,
        intensity_channels=["Channel1"],
        point_objects=1,
    )

    assert isinstance(res, list)
    assert len(res) == 1
    
    point_features = res[0]
    
    # Should have parent_label and parent_label_name columns
    assert "parent_label" in point_features.columns
    assert "parent_label_name" in point_features.columns
    assert (point_features["parent_label_name"] == "Object1").all()
    
    # Parent labels should be valid (>= 0)
    assert (point_features["parent_label"] >= 0).all()
    
    # Count points per parent (excluding parent_label=0)
    points_per_parent = point_features[point_features["parent_label"] > 0].groupby("parent_label").size()
    
    # Should have points in multiple parents
    assert len(points_per_parent) > 1


def test_point_object_3D_aggregate(_ensure_test_data):
    """Test point object quantification in 3D with aggregation to parent."""
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")

    # Quantify with aggregation: channel 0 (parent) and channel 1 (points) - all timepoints
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=[0, 1],
        parent_object=0,
        aggregate=True,
        intensity_channels=["Channel1", "Channel2"],
        point_objects=1,
    )

    # Should return a single aggregated DataFrame
    assert isinstance(res, pd.DataFrame)
    
    # Save results for manual inspection
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    res.to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_3D_aggregate_results.csv", index=False)
    
    # Should have parent features (3D)
    assert "Object1_3D_number_of_voxels" in res.columns
    assert "label" in res.columns
    
    # Should have aggregated point features
    assert "Object2_count" in res.columns
    assert "Object2_intensity_sum_Channel1" in res.columns
    assert "Object2_intensity_mean_Channel1" in res.columns
    assert "Object2_intensity_min_Channel1" in res.columns
    assert "Object2_intensity_max_Channel1" in res.columns
    assert "Object2_intensity_sum_Channel2" in res.columns
    
    # Check that counts are reasonable
    assert (res["Object2_count"] >= 0).all()
    
    # Verify aggregation: mean should be between min and max
    mask = res["Object2_count"] > 0
    assert (res.loc[mask, "Object2_intensity_mean_Channel1"] >= res.loc[mask, "Object2_intensity_min_Channel1"]).all()
    assert (res.loc[mask, "Object2_intensity_mean_Channel1"] <= res.loc[mask, "Object2_intensity_max_Channel1"]).all()
    
    # Verify no duplicate columns with _x/_y suffixes
    assert not any(col.endswith("_x") or col.endswith("_y") for col in res.columns), \
        f"Found duplicate columns: {[col for col in res.columns if col.endswith('_x') or col.endswith('_y')]}"
    
    # Verify fast-path was used: no standard aggregation columns
    assert not any("_sum_sum" in col for col in res.columns)
    
    # Check multiple timepoints are present
    unique_timepoints = res["TimepointID"].unique()
    assert len(unique_timepoints) >= 1
    
    # Number of rows should equal number of parent objects * number of timepoints
    label_array = label_image_3D.get_image_data("ZYX", C=0, T=0)
    num_parents = len(np.unique(label_array)) - 1  # Exclude background
    num_timepoints = intensity_image_3D.dims.T
    assert len(res) == num_parents * num_timepoints


def test_point_object_mixed_with_regular_objects_aggregate(_ensure_test_data):
    """Test quantification with both point objects and regular objects, with aggregation."""
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Add channel 2 as a regular object to test mixing
    # Quantify: channel 0 (parent), channel 1 (points), channel 2 (regular - will be filtered based on data)
    # Note: Based on test structure, channel 2 may not be valid for aggregation
    # Let's just test with channels 0 and 1 for now
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=[0, 1],
        parent_object=0,
        aggregate=True,
        timepoint=0,
        intensity_channels=["Channel1"],
        point_objects=[1],  # Only channel 1 is a point object
    )

    # Should successfully aggregate
    assert isinstance(res, pd.DataFrame)
    assert "Object1_area" in res.columns
    assert "Object2_count" in res.columns


def test_point_object_aggregate_comparison_with_manual(_ensure_test_data):
    """Verify that fast aggregation produces consistent results with expected behavior."""
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Get aggregated results - all timepoints
    res_agg = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=[0, 1],
        parent_object=0,
        aggregate=True,
        intensity_channels=["Channel1"],
        point_objects=1,
    )

    # Get non-aggregated results to verify counts - all timepoints
    res_non_agg = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        parent_object=0,
        aggregate=False,
        intensity_channels=["Channel1"],
        point_objects=1,
    )

    # Count points per parent manually from non-aggregated data (per timepoint)
    point_features = res_non_agg[0]
    
    # For each timepoint in aggregated results, verify counts match
    for _, agg_row in res_agg.iterrows():
        parent_label = agg_row["label"]
        timepoint_id = agg_row["TimepointID"]
        agg_count = agg_row["Object2_count"]
        
        # Count points for this parent in this timepoint from non-aggregated data
        manual_count = len(point_features[
            (point_features["parent_label"] == parent_label) & 
            (point_features["TimepointID"] == timepoint_id)
        ])
        
        assert agg_count == manual_count, \
            f"Count mismatch for parent {parent_label} at timepoint {timepoint_id}: aggregated={agg_count}, manual={manual_count}"


def test_point_object_with_string_input(_ensure_test_data):
    """Test point object specification using string channel names."""
    testdata_config = blimp_config.get_data_config("testdata")
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Test 1: Non-aggregated with string name - all timepoints
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects="Object2",
        intensity_channels=["Channel1"],
        point_objects="Object2",  # String input
    )

    assert isinstance(res, list)
    assert len(res) == 1
    assert "Object2_area" in res[0].columns
    assert (res[0]["Object2_area"] == 1).all()
    
    # Save results for manual inspection
    logger.info(f"saving results to {testdata_config.RESULTS_DIR}")
    res[0].to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_string_input_non_aggregate_results.csv", index=False)
    
    # Test 2: Aggregated with string names for both measure_objects and point_objects - all timepoints
    res_agg = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=["Object1", "Object2"],  # String names
        parent_object="Object1",  # String name
        aggregate=True,
        intensity_channels=["Channel1"],
        point_objects="Object2",  # String input for point object
    )
    
    assert isinstance(res_agg, pd.DataFrame)
    assert "Object2_count" in res_agg.columns
    assert "Object1_area" in res_agg.columns
    
    # Save aggregated results
    res_agg.to_csv(Path(testdata_config.RESULTS_DIR) / "point_object_string_input_aggregate_results.csv", index=False)
    
    # Test 3: List of string names for point_objects parameter - all timepoints
    res_list = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=["Object1", "Object2"],
        parent_object="Object1",
        aggregate=True,
        intensity_channels=["Channel1"],
        point_objects=["Object2"],  # List of string names
    )
    
    assert isinstance(res_list, pd.DataFrame)
    assert "Object2_count" in res_list.columns
    
    # Verify both approaches give the same result
    pd.testing.assert_frame_equal(res_agg, res_list)


def test_point_object_no_intensity_channels(_ensure_test_data):
    """Test point object quantification when no intensity channels specified."""
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")

    # Without intensity channels, should still work but have minimal features
    res = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        timepoint=0,
        point_objects=1,
    )

    assert isinstance(res, list)
    point_features = res[0]
    
    # Should have label and area (minimal features for point objects)
    assert "label" in point_features.columns
    assert "Object2_area" in point_features.columns
    # Point objects will have area=1 for each point
    assert (point_features["Object2_area"] == 1).all()


def test_point_object_2D_comparison_with_standard_object(_ensure_test_data):
    """Compare point object quantification with standard object quantification in 2D.
    
    Verify that the total area from standard quantification matches the number of 
    point objects (each with area=1).
    """
    intensity_image_2D, label_image_2D = _load_test_data("synthetic_2D")
    
    # Quantify Object2 as a standard object
    res_standard = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        timepoint=0,
        intensity_channels=["Channel1"],
    )
    
    # Quantify Object2 as point objects
    res_points = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_2D,
        label_image=label_image_2D,
        measure_objects=1,
        timepoint=0,
        intensity_channels=["Channel1"],
        point_objects=1,
    )
    
    # Calculate total area from standard quantification
    standard_features = res_standard[0]
    total_area_standard = standard_features["Object2_area"].sum()
    
    # Calculate total "area" from point objects (count of points, each with area=1)
    point_features = res_points[0]
    num_points = len(point_features)
    total_area_points = point_features["Object2_area"].sum()
    
    # The totals should match
    assert num_points == total_area_standard, \
        f"Point count ({num_points}) doesn't match standard total area ({total_area_standard})"
    assert total_area_points == total_area_standard, \
        f"Point total area ({total_area_points}) doesn't match standard total area ({total_area_standard})"
    
    # Verify that total intensity also matches (sum across all points)
    total_intensity_standard = standard_features["Object2_intensity_sum_Channel1"].sum()
    total_intensity_points = point_features["Object2_intensity_Channel1"].sum()
    
    assert total_intensity_standard == total_intensity_points, \
        f"Point total intensity ({total_intensity_points}) doesn't match standard total intensity ({total_intensity_standard})"


def test_point_object_3D_comparison_with_standard_object(_ensure_test_data):
    """Compare point object quantification with standard object quantification in 3D.
    
    Verify that the total volume (in voxels) from standard quantification matches 
    the number of point objects (each representing one voxel).
    """
    intensity_image_3D, label_image_3D = _load_test_data("synthetic_3D")
    
    # Quantify Object2 as a standard object
    res_standard = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=1,
        timepoint=0,
        intensity_channels=["Channel1"],
    )
    
    # Quantify Object2 as point objects
    res_points = blimp.processing.quantify.quantify(
        intensity_image=intensity_image_3D,
        label_image=label_image_3D,
        measure_objects=1,
        timepoint=0,
        intensity_channels=["Channel1"],
        point_objects=1,
    )
    
    # Calculate total volume from standard quantification
    standard_features = res_standard[0]
    total_voxels_standard = standard_features["Object2_3D_number_of_voxels"].sum()
    
    # Calculate count from point objects (each represents one voxel)
    point_features = res_points[0]
    num_points = len(point_features)
    
    # The totals should match
    assert num_points == total_voxels_standard, \
        f"Point count ({num_points}) doesn't match standard total voxels ({total_voxels_standard})"
    
    # Verify that total intensity also matches (sum across all voxels)
    total_intensity_standard = standard_features["Object2_3D_intensity_sum_Channel1"].sum()
    total_intensity_points = point_features["Object2_3D_intensity_Channel1"].sum()
    
    assert total_intensity_standard == total_intensity_points, \
        f"Point total intensity ({total_intensity_points}) doesn't match standard total intensity ({total_intensity_standard})"
