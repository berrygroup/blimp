"""
Copyright 2022 (C) University of New South Wales
Original author:
Scott Berry <scott.berry@unsw.edu.au>
"""
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from pathlib import Path
from blimp.preprocessing.operetta_parse_metadata import get_images_metadata, get_plate_metadata


# combine z and channel info into OME TIFFs
# sort by increasing z position and and select a single channel/site
def read_images_single_site(in_path,image_metadata,Row,Col,StandardFieldID,ChannelID,TimepointID):
    filename_list = image_metadata.sort_values(
        by=["TimepointID","ChannelID","Row","Col","StandardFieldID","AbsPositionZ"]
    ).query(
        "Row=='" + Row + "' & Col=='" + Col + "' & StandardFieldID=='" + StandardFieldID + "' & ChannelID=='" + ChannelID + "' & TimepointID=='" + TimepointID + "'"
    ).URL.tolist()
    filepath_list = [Path(in_path) / filename for filename in filename_list]
    images = [AICSImage(filepath) for filepath in filepath_list]
    return(images)


def operetta_to_ome_tiff(in_path, out_path, metadata_file, save_metadata_files=False):
    """
    Reads a metadata file and loads individual TIFF files exported from 
    the Perkin-Elmer Operetta. These are combined into multi-channel,
    multi-z-plane stacks and saved as OME-TIFFs

    Parameters
    ----------
    in_path : str or Path
        path to the "Images" directory (including "Images")
    out_path : str or Path
        path where the OME-TIFFs should be saved
    metadata_file : str
        name of the xml metadata file inside "Images"
    save_metadata_files : bool, optional
        whether the metadata files should be saved after XML parsing

    """
    
    if (save_metadata_files):
        plate_metadata_file = Path(in_path) / "plate_metadata.csv"
        image_metadata_file = Path(in_path) / "image_metadata.csv"
    else:
        plate_metadata_file = ""
        image_metadata_file = ""

    metadata_path = Path(in_path) / metadata_file

    plate_metadata = get_plate_metadata(metadata_path,plate_metadata_file)
    image_metadata = get_images_metadata(metadata_path,image_metadata_file)

    # get unique sites
    timepoints = image_metadata[["TimepointID"]].drop_duplicates()
    sites = image_metadata[["Row","Col","StandardFieldID"]].drop_duplicates()
    channels = image_metadata[["ChannelID"]].drop_duplicates()

    for site_index, site in sites.iterrows():
        multi_timepoint_img=[]
        for timepoint_index, timepoint in timepoints.iterrows():
            multi_channel_img=[]
            for channel_index, channel in channels.iterrows():
                imgs = read_images_single_site(
                    in_path,image_metadata,
                    site.Row,site.Col,site.StandardFieldID,channel.ChannelID,timepoint.TimepointID
                )
                # combine z-planes (axis 2)
                multi_channel_img.append(np.concatenate([img.data for img in imgs],axis=2))
            # combine channels (axis 1)
            multi_timepoint_img.append(np.concatenate(multi_channel_img,axis=1))
        # combine timepoints (axis 0)
        multi_timepoint_img = np.concatenate(multi_timepoint_img,axis=0)

        # write OME-TIFF
        


# image = np.random.rand(10, 3, 1024, 2048)
# ome = OmeTiffWriter.build_ome(data_shapes = ,
# data_types = [np.uint16,np.uint8,np.uint16,np.uint16,np.uint16],
# dimension_order = ["T","C","Z","Y","X"],
# channel_names = ,
# image_name = ,
# physical_pixel_sizes = [],
# channel_colors = [],
# is_rgb = ,)
# OmeTiffWriter.save(image, "file.ome.tif", ome_xml = ome)




if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="operetta_to_ome_tiff")

    parser.add_argument(
        "-i",
        "--in_path",
        help="directory containing the input files"
    )

    parser.add_argument(
        "-o",
        "--out_path",
        help="directory to write the output files"
    )

    parser.add_argument(
        "-m", "--mip",
        default=False,
        action="store_true"
        help="whether to save maximum intensity projections"
    )

    parser.add_argument(
        "-f",
        "--metadata_file",
		default="Index.idx.xml"
        help="name of the metadata file",
    )

    parser.add_argument(
        "-s",
        "--save_metadata_files",
        default=False,
        action="store_true"
        help="flag to indicate that metadata files should be saved",
    )

    args = parser.parse_args()

    operetta_to_ome_tiff(
        in_path=args.in_path,
        out_path=args.out_path,
        metadata_file=args.metadata_file,
        mip=args.mip,
        save_metadata_files=args.save_metadata_files
    )

