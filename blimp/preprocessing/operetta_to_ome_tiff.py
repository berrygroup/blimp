"""
Copyright 2022 (C) University of New South Wales
Original author:
Scott Berry <scott.berry@unsw.edu.au>
"""
import os
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Pattern
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
from blimp.utils import init_logging
from blimp.preprocessing.operetta_parse_metadata import get_image_metadata, get_plate_metadata

OPERETTA_REGEX = 'r(?P<Row>\d+)c(?P<Col>\d+)f(?P<FieldID>\d+)p(?P<PlaneID>\d+)' + \
                 '-ch(?P<ChannelID>\d+)sk(?P<TimepointID>\d+)(?P<End>fk\d+fl\d+)'
operetta_regex_filename_pattern = re.compile(OPERETTA_REGEX)

# combine z and channel info into OME TIFFs
# sort by increasing z position and and select a single channel/site
def _read_images_single_site(
    in_path : Union[str,Path],
    image_metadata : pd.DataFrame,
    Row : int,
    Col : int,
    StandardFieldID : int,
    ChannelID : int,
    TimepointID : int) -> List[AICSImage]:
    """
    Reads a set of images corresponding to a single imaging 
    site (field of view) from inividual files on disk.

    Parameters
    ----------
    in_path
        Full path to the folder containing images
    image_metadata
        Metadata dataframe such as that generated by 
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    Row
        Plate row
    Col
        Plate column
    StandardFieldID
        Field identifier (top-left to bottom-right, 
        column-wise increasing first)
    ChannelID
        Channel identifier
    TimepointID
        Timepoint identifier
    
    Returns
    -------
    PhysicalPixelSizes
        AICSImage object for containing pixel dimensions
    """
    
    query = "Row==" + str(Row) + \
            " & Col==" + str(Col) + \
            " & StandardFieldID==" + str(StandardFieldID) + \
            " & ChannelID==" + str(ChannelID) + \
            " & TimepointID==" + str(TimepointID)
    
    filename_list = image_metadata.sort_values(
        by=["TimepointID","ChannelID","Row","Col","StandardFieldID","AbsPositionZ"]
    ).query(query).URL.tolist()
    filepath_list = [Path(in_path) / filename for filename in filename_list]
    images = [AICSImage(filepath) for filepath in filepath_list]
    return(images)


def _get_zyx_resolution(
    image_metadata : pd.DataFrame,
    Row : int,
    Col : int,
    StandardFieldID : int,
    ChannelID : int,
    TimepointID : int) -> PhysicalPixelSizes:
    """
    Determines the z,y,x resolution from the metadata

    Parameters
    ----------
    image_metadata
        Metadata dataframe such as that generated by 
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    Row
        Plate row
    Col
        Plate column
    StandardFieldID
        Field identifier (top-left to bottom-right, 
        column-wise increasing first)
    ChannelID
        Channel identifier
    TimepointID
        Timepoint identifier
    
    Returns
    -------
    PhysicalPixelSizes
        AICSImage object for containing pixel dimensions
    """
    
    # z resolution is not directly provided in the image metadata,
    # derive this from the full z-range divided by number of planes
    query = "Row==" + str(Row) + \
            " & Col==" + str(Col) + \
            " & StandardFieldID==" + str(StandardFieldID) + \
            " & ChannelID==" + str(ChannelID) + \
            " & TimepointID==" + str(TimepointID)
        
    xyz = image_metadata.query(query)[["ImageResolutionX","ImageResolutionY","AbsPositionZ"]].drop_duplicates()
    
    return(PhysicalPixelSizes(Z=(xyz.AbsPositionZ.max() - xyz.AbsPositionZ.min()) / (xyz.shape[0]-1),
                              Y=xyz.ImageResolutionY.iloc[0],
                              X=xyz.ImageResolutionX.iloc[0]))


def _remove_TCZ_filename(pattern : Pattern,
                         filename : str,
                         mip : bool=False):
    """
    Restructures a operetta tiff filename string to 
    remove reference to Time, Channel, Z

    Parameters
    ----------
    pattern
        Compiled operetta regular expression
    filename
        filename to change
    mip
        Whether or not the metadata represents maximum-intensity
        projections
    
    Returns
    -------
    str
        Filename with TCZ information removed

    """
    m = pattern.match(filename)
    new_filename = 'r' + m.group('Row') + \
                   'c' + m.group('Col') + \
                   'f' + m.group('FieldID') + \
                   '-' + m.group('End') 
    if mip: 
        new_filename += '-mip.ome.tiff'
    else:
        new_filename += '.ome.tiff'
    return(new_filename)


def _aggregate_TCZ_metadata(
    image_metadata : pd.DataFrame,
    mip : bool=False) -> pd.DataFrame:
    """
    Removes Time, Channel, Z-plane information from a dataframe. 
    Absolute imaging time information is averaged across all 
    acquisitions for a given time-point

    Parameters
    ----------
    image_metadata
        Metadata dataframe such as that generated by 
        blimp.preprocessing.operetta_parse_metadata.get_image_metadata
    mip
        Whether or not the metadata represents maximum-intensity
        projections
    
    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = number of imaging sites (fields-of-view))

    """
    # get average absolute time for each unique imaging site
    mean_acq_time = image_metadata.groupby(["Row","Col","FieldID","TimepointID"])["AbsTime"].mean()
    n_planes = image_metadata.groupby(["Row","Col","FieldID","TimepointID"])["id"].count()
    n_planes.name = 'NPlanes'
    n_channels = image_metadata.groupby(["Row","Col","FieldID","TimepointID"])["ChannelID"].max()
    n_channels.name = 'NChannels'
    
    # drop columns that don't make sense after merging multiple files
    image_metadata_no_z = image_metadata.query("PlaneID==1 & ChannelID==1").drop(
        labels=["ChannelID","ChannelName","ChannelType",
                "MainExcitationWavelength","MainEmissionWavelength",
                "AbsTime","AbsPositionZ","PositionZ","PlaneID"],axis=1)
    
    # remove T, C, Z information from the filename 
    image_metadata_no_z['URL'] = image_metadata_no_z['URL'].apply(
        lambda x: _remove_TCZ_filename(operetta_regex_filename_pattern,x,mip))
 
    # merge all
    image_metadata_no_z = image_metadata_no_z.merge(
        right=mean_acq_time,
        on=["Row","Col","FieldID","TimepointID"],
        how='left').merge(
        right=n_channels,
        on=["Row","Col","FieldID","TimepointID"],
        how='left').merge(
        right=n_planes,
        on=["Row","Col","FieldID","TimepointID"],
        how='left'
    )
    return(image_metadata_no_z)
    

def operetta_to_ome_tiff(
    in_path : Union[str,Path],
    out_path : Union[str,Path],
    metadata_file : str,
    save_metadata_files : bool=True,
    mip : bool=False) -> None:
    """
    Reads a metadata file and loads individual TIFF files exported from 
    the Perkin-Elmer Operetta. These are combined into multi-channel,
    multi-z-plane stacks and saved as OME-TIFFs

    Parameters
    ----------
    in_path
        path to the "Images" directory (including "Images")
    out_path
        path where the OME-TIFFs should be saved
    metadata_file
        name of the xml metadata file inside "Images"
    save_metadata_files
        whether the metadata files should be saved after XML parsing

    """
    init_logging()
    log = logging.getLogger("operetta_to_ome_tiff")
    
    in_path = Path(in_path)
    out_path = Path(out_path)
    metadata_path = in_path / metadata_file
    
    if in_path==out_path:
        log.error("Input and output paths are the same.")
        os._exit(1)
    
    plate_metadata_file = in_path / "plate_metadata.csv" if save_metadata_files else None
    image_metadata_file = in_path / "image_metadata.csv" if save_metadata_files else None

    plate_metadata = get_plate_metadata(metadata_path,plate_metadata_file)
    image_metadata = get_image_metadata(metadata_path,image_metadata_file)

    # make output directories
    if not out_path.exists():
        out_path.mkdir(parents=True)
    if mip:
        out_path_mip = out_path.parent / (str(out_path.stem) + '-MIP')
        if not out_path_mip.exists():
            out_path_mip.mkdir(parents=True)
    
    # get unique sites (fields-of-view)
    timepoints = image_metadata[["TimepointID"]].drop_duplicates()
    # FIXME: test on a single site using iloc
    sites = image_metadata[["Row","Col","StandardFieldID"]].drop_duplicates().iloc[:1]
    channels = image_metadata[["ChannelID"]].drop_duplicates()
    
    # get channel names
    channel_names = image_metadata[["ChannelID","ChannelName"]].drop_duplicates().sort_values(
        by=["ChannelID"]).ChannelName.tolist()

    # generate new metadata dataframes
    # (remove Z and T information from metadata)
    image_metadata_ome_tiff = _aggregate_TCZ_metadata(image_metadata)    
    
    # save metadata files
    if (save_metadata_files):
        image_metadata_ome_tiff.to_csv(Path(out_path) / "image_metadata.csv", index=False)
        image_metadata_ome_tiff.to_pickle(Path(out_path) / "image_metadata.pkl")
        
    if (mip):
        image_metadata_mip_ome_tiff = _aggregate_TCZ_metadata(image_metadata,mip=True)
        if (save_metadata_files):
            image_metadata_mip_ome_tiff.to_csv(Path(out_path_mip) / "image_metadata.csv", index=False)
            image_metadata_mip_ome_tiff.to_pickle(Path(out_path_mip) / "image_metadata.pkl")
    
    # iterate over sites, combining time-points, channels and z-planes into OME-TIFFs
    for site_index, site in sites.iterrows():
        multi_timepoint_img=[]
        multi_timepoint_img_mip=[]
        
        for timepoint_index, timepoint in timepoints.iterrows():
            multi_channel_img=[]
            multi_channel_img_mip=[]
            
            for channel_index, channel in channels.iterrows():
                imgs = _read_images_single_site(
                    in_path,image_metadata,
                    site.Row,site.Col,site.StandardFieldID,channel.ChannelID,timepoint.TimepointID
                )
                
                # combine z-planes (axis 2)
                z_stack = np.concatenate([img.data for img in imgs],axis=2)
                multi_channel_img.append(z_stack)
                if (mip):
                    multi_channel_img_mip.append(z_stack.max(axis=2,keepdims=True))
                    
            # combine channels (axis 1)
            multi_timepoint_img.append(np.concatenate(multi_channel_img,axis=1))
            if (mip):
                multi_timepoint_img_mip.append(np.concatenate(multi_channel_img_mip,axis=1))
            
        # combine timepoints (axis 0)
        multi_timepoint_img = np.concatenate(multi_timepoint_img,axis=0)
        if (mip):
            multi_timepoint_img_mip = np.concatenate(multi_timepoint_img_mip,axis=0)
        
        # get voxel dimensions
        voxel_dimensions = _get_zyx_resolution(
            image_metadata,site.Row,site.Col,site.StandardFieldID,1,0)
        
        # get new filenames from restructured metadata dataframe
        out_file_path = image_metadata_ome_tiff.query(
            "Row==" + str(site.Row) + \
            " & Col==" + str(site.Col) + \
            " & StandardFieldID==" + str(site.StandardFieldID)).URL.iloc[0]
        if (mip):
            out_file_path_mip = Path(out_path_mip) / out_file_path
        out_file_path = Path(out_path) / out_file_path

        
        # write to OME TIFF (metadata provided is written to TIFF file)
        OmeTiffWriter.save(
            data = multi_timepoint_img,
            uri = out_file_path,
            dim_order="TCZYX",
            channel_names=channel_names,
            physical_pixel_sizes=voxel_dimensions,
            parser='lxml')
        
        # write MIP to OME TIFF
        if (mip):
            OmeTiffWriter.save(
                data = multi_timepoint_img_mip,
                uri = out_file_path_mip,
                dim_order="TCZYX",
                channel_names=channel_names,
                physical_pixel_sizes=voxel_dimensions,
                parser='lxml')
    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="operetta_to_ome_tiff")

    parser.add_argument(
        "-i",
        "--in_path",
        help="directory containing the input files",
        required=True
    )

    parser.add_argument(
        "-o",
        "--out_path",
        help="directory to write the output files",
        required=True
    )

    parser.add_argument(
        "-f",
        "--metadata_file",
        default="Index.idx.xml",
        help="name of the metadata file",
        required=True
    )

    parser.add_argument(
        "-s",
        "--save_metadata_files",
        default=False,
        action="store_true",
        help="flag to indicate that metadata files should be saved"
    )
    
    parser.add_argument(
        "-m", "--mip",
        default=False,
        action="store_true",
        help="whether to save maximum intensity projections"
    )

    args = parser.parse_args()

    operetta_to_ome_tiff(
        in_path=args.in_path,
        out_path=args.out_path,
        metadata_file=args.metadata_file,
        save_metadata_files=args.save_metadata_files,
        mip=args.mip
    )

