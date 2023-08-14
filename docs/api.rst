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
    preprocessing.get_image_metadata
    preprocessing.get_plate_metadata

Image registration
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    preprocessing.registration.register_2D
    preprocessing.registration.transform_2D
    preprocessing.registration.calculate_shifts
    preprocessing.registration.apply_shifts

Processing
~~~~~~~~~~

.. module:: blimp.processing
.. currentmodule:: blimp

.. autosummary::
    :toctree: api

    processing.segment_nuclei_cellpose
    processing.quantify
    processing.segment_and_quantify
