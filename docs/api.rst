API
===

Preprocessing
~~~~~~~~~~~~~

.. module:: blimp.preprocessing
.. currentmodule:: blimp

File handling
^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    preprocessing.convert_nd2
    preprocessing.nd2_to_ome_tiff
    preprocessing.convert_operetta
    preprocessing.operetta_to_ome_tiff
    preprocessing.operetta_parse_metadata.get_image_metadata
    preprocessing.operetta_parse_metadata.get_plate_metadata

Processing
~~~~~~~~~~

.. module:: blimp.processing
.. currentmodule:: blimp

.. autosummary::
    :toctree: api

    processing.segment.segment_nuclei_cellpose
    processing.quantify.quantify
