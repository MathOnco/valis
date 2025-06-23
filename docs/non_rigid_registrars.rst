Non-rigid registration
**********************

.. automodule:: valis.non_rigid_registrars

Base NonRigidRegistrar
----------------------
.. autoclass:: valis.non_rigid_registrars::NonRigidRegistrar
    :members: __init__, calc, register

Base NonRigidRegistrarXY
------------------------
.. autoclass:: valis.non_rigid_registrars::NonRigidRegistrarXY
    :members: __init__, calc, register
    :show-inheritance:

Base NonRigidRegistrarGroupwise
-------------------------------
.. autoclass:: valis.non_rigid_registrars::NonRigidRegistrarGroupwise
    :members: __init__, register
    :show-inheritance:

OpticalFlowWarper
-----------------
.. autoclass:: valis.non_rigid_registrars::OpticalFlowWarper
    :members: __init__, calc
    :show-inheritance:

SimpleElastixWarper
-------------------
.. autoclass:: valis.non_rigid_registrars::SimpleElastixWarper
    :members: __init__, get_default_params, calc
    :show-inheritance:

SimpleElastixGroupwiseWarper
----------------------------
.. autoclass:: valis.non_rigid_registrars::SimpleElastixGroupwiseWarper
    :members: __init__, get_default_params, calc
    :show-inheritance: