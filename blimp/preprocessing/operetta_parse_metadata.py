"""
Copyright 2022 (C) University of New South Wales
Original author:
Scott Berry <scott.berry@unsw.edu.au>
"""
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

def _remove_ns(str):
    """
    Strip text before "}"
    
    Parameters
    ----------
    str : string
        string to be modified
    
    Returns
    -------
    string
        string with text up to and including "}" removed
    
    Examples
    --------
    >>> _remove_ns("{http://www.perkinelmer.com/PEHH/HarmonyV5}Plates")
    'Plates'
    
    """
    return str.split("}")[1][0:]


def _xml_to_df(xmls,ns_key,ns_dict):
    """
    Convert a list of xmls into a pandas dataframe, 
    where each xml forms a row.

    Parameters
    ----------
    xmls : xml.etree.ElementTree
        XML element tree to parse
    ns_key : string
        namespace key to use
    ns_dict : dict
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


def get_plate_metadata(metadata_file, out_file=""):
    """
    Extracts plate metadata from the operetta xml file

    Parameters
    ----------
    metadata_file : str or Path
        path to the xml metadata file
    out_file : str or Path, optional
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
    if (out_file!=""):
        if (Path(out_file).suffix==".csv"):
            plate_metadata.to_csv(out_file)
        elif(Path(out_file).suffix==".pkl")
            plate_metadata.to_pickle(out_file)

    return(plate_metadata)


def get_images_metadata(metadata_file, out_file=""):
    """
    Extracts image metadata from the operetta xml file

    Parameters
    ----------
    metadata_file : str or Path
        path to the xml metadata file
    out_file : str or Path, optional
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

    # write file if requested
    if (out_file!=""):
        if (Path(out_file).suffix==".csv"):
            image_metadata.to_csv(out_file)
        elif(Path(out_file).suffix==".pkl")
            image_metadata.to_pickle(out_file)

    return(image_metadata)