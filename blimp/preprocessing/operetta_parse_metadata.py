"""
Copyright 2023 (C) University of New South Wales
Original author:
Scott Berry <scott.berry@unsw.edu.au>
"""
import pandas as pd
import numpy as np
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

logger = logging.getLogger("operetta_parse_metadata")

image_metadata_dtypes = {'id': str,
 'State': str,
 'URL': str,
 'Row': np.uint8,
 'Col': np.uint8,
 'FieldID': np.uint16,
 'PlaneID': np.uint16,
 'TimepointID': np.int64,
 'ChannelID': np.uint8,
 'FlimID': np.uint8,
 'ChannelName': str,
 'ImageType': str,
 'AcquisitionType': str,
 'IlluminationType': str,
 'ChannelType': str,
 'ImageResolutionX': float,
 'ImageResolutionY': float,
 'ImageSizeX': np.uint16,
 'ImageSizeY': np.uint16,
 'BinningX': np.uint8,
 'BinningY': np.uint8,
 'MaxIntensity': int,
 'CameraType': str,
 'PositionX': float,
 'PositionY': float,
 'PositionZ': float,
 'AbsPositionZ': float,
 'MeasurementTimeOffset': float,
 'AbsTime': 'datetime64[ns]',
 'MainExcitationWavelength': np.uint16,
 'MainEmissionWavelength': np.uint16,
 'ObjectiveMagnification': np.uint8,
 'ObjectiveNA': float,
 'ExposureTime': float,
 'OrientationMatrix': str,
 'StandardFieldID': np.uint16}

def _remove_ns(s : str) -> str:
    """
    Strip text before "}"
    
    Parameters
    ----------
    s : string
        string to be modified
    
    Returns
    str
        string with text up to and including "}" removed
    
    Examples
    --------
    >>> _remove_ns("{http://www.perkinelmer.com/PEHH/HarmonyV5}Plates")
    'Plates'
    
    """
    return s.split("}")[1][0:]


def _xml_to_df(xmls,
               ns_key : str,
               ns_dict : dict) -> pd.DataFrame :
    """
    Convert a list of xmls into a pandas dataframe, 
    where each xml forms a row.

    Parameters
    ----------
    xmls
        XML element tree to parse
    ns_key
        namespace key to use
    ns_dict
        namespace dictionary

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = length(xmls), n_cols = # fields)

    Examples
    --------
    >>> idx_path = image_dir / 'Images' / 'Index.idx.xml'
    >>> idx_xml = ET.parse(idx_path).getroot()
    >>> ns = {'harmony': "http://www.perkinelmer.com/PEHH/HarmonyV5"}
    >>> plates_xml = idx_xml.find('harmony:Plates',namespaces=ns).findall('harmony:Plate',namespaces=ns)
    >>> _xml_to_df(plates_xml,"harmony",ns)
    
    """
    metadata=[]
    for xml in xmls:
        # get column names from tags
        xml_tags = [_remove_ns(field.tag) for field in xml]
        # read metadata using a dict comprehension
        metadata.append(
            {xml_tag : xml.find(ns_key + ":" + xml_tag, namespaces=ns_dict).text for xml_tag in xml_tags}
        )
    # convert to dataframe
    return(pd.DataFrame(metadata))


def _to_well_name(
    row : int,
    column : int) -> str :
    """
    Convert row and column numbers to well name

    Parameters
    ----------
    row : int
        Row of the plate
    column : int
        Column of the plate
    
    Returns
    -------
    str
        Well name

    Examples
    --------
    >>> _to_well_name(1,13)
    A13
    
    """
    return(chr(96 + row).upper() + "%0.2d" % column)


def get_plate_metadata(
    metadata_file : Union[str,Path],
    out_file : Union[str,Path,None]=None) -> pd.DataFrame:
    """
    Extracts plate metadata from the operetta xml file

    Parameters
    ----------
    metadata_file
        path to the xml metadata file
    out_file
        enter a file path if this dataframe should be written to file 
        (possible extensions are .csv or .pkl)

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = length(xmls), n_cols = # xml fields)
    """
    # define xml namespace
    ns = {'harmony': "http://www.perkinelmer.com/PEHH/HarmonyV5"}
    
    # get xml
    metadata_xml = ET.parse(metadata_file).getroot()
    plates_xml = metadata_xml.find('harmony:Plates',namespaces=ns).findall('harmony:Plate',namespaces=ns)

    # convert to dataframe
    plate_metadata = _xml_to_df(plates_xml,"harmony",ns)

    # write file if requested
    if out_file is not None:
        logger.info("Save plate metadata to file: {}".format(str(out_file)))
        if (Path(out_file).suffix==".csv"):
            plate_metadata.to_csv(out_file, index=False)
        elif(Path(out_file).suffix==".pkl"):
            plate_metadata.to_pickle(out_file)

    return(plate_metadata)


def get_image_metadata(
    metadata_file : Union[str,Path],
    out_file : Union[str,Path,None]=None) -> pd.DataFrame:
    """
    Extracts image metadata from the operetta xml file

    Parameters
    ----------
    metadata_file
        path to the xml metadata file
    out_file
        enter a file path if this dataframe should be written to file 
        (possible extensions are .csv or .pkl)

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = length(xmls), n_cols = # xml fields)
    """
    # define xml namespace
    ns = {'harmony': "http://www.perkinelmer.com/PEHH/HarmonyV5"}

    # get xml
    metadata_xml = ET.parse(metadata_file).getroot()
    images_xml = metadata_xml.find('harmony:Images',namespaces=ns).findall('harmony:Image',namespaces=ns)

    # convert to dataframe
    image_metadata = _xml_to_df(images_xml,"harmony",ns)

    # add the field index
    # add field indices using standard order (top-left to bottom right, incrementing columns first)
    image_metadata['XCoordinate']=(image_metadata['PositionX'].astype('float')*1e9).astype('int')
    image_metadata['YCoordinate']=(image_metadata['PositionY'].astype('float')*1e9).astype('int')
    image_metadata['XYCoordinates']= image_metadata[['XCoordinate','YCoordinate']].apply(tuple, axis=1)

    # Number fields from top-left to bottom-right (increase x first)
    unique_int_coords_sorted = sorted(list(set(image_metadata['XYCoordinates'])) , key=lambda k: [-k[1], k[0]])
    coord_index = dict(zip(unique_int_coords_sorted, ["%0d" %i for i in range(1,len(unique_int_coords_sorted)+1)])) 

    # keep this as StandardFieldID
    image_metadata['StandardFieldID'] = image_metadata['XYCoordinates'].map(coord_index)
    image_metadata = image_metadata.drop(columns=['XYCoordinates','XCoordinate','YCoordinate'])

    image_metadata = image_metadata.astype(image_metadata_dtypes)
    
    # add a "WellName" identifier
    image_metadata["WellName"] = image_metadata[["Row","Col"]].apply(lambda x: _to_well_name(x.Row, x.Col), axis=1)
    
    # write file if requested
    if out_file is not None:
        logger.info("Save plate metadata to file: {}".format(str(out_file)))
        if (Path(out_file).suffix==".csv"):
            image_metadata.to_csv(out_file, index=False)
        elif(Path(out_file).suffix==".pkl"):
            image_metadata.to_pickle(out_file)

    print(out_file)
    return(image_metadata)


def load_image_metadata(metadata_file: Union[str,Path]):
    """
    Loads image metadata previously saved during image conversion

    Parameters
    ----------
    metadata_file
        path to the pkl or csv metadata file

    Returns
    -------
    pandas.DataFrame
        Dimensions (n_rows = # fields-of-view, n_cols = # xml fields)
    """
    metadata_file = Path(metadata_file)
    if metadata_file.suffix==".pkl":
        metadata = pd.read_pickle(metadata_file)
    elif metadata_file.suffix==".csv":
        metadata = pd.read_csv(metadata_file)
    else:
        logger.error("Unknown metadata file: {}".format(str(metadata_file)))
    return(metadata)