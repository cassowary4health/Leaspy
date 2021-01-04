:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members:
   :special-members:
   :show-inheritance:
   :inherited-members:

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
    {% for item in attributes %}
       ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    .. automethod:: __init__
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
       ~{{ name }}.{{ item }}
    {%- endfor %}
..    .. automethod:: __init__
    {% endif %}
    {% endblock %}

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>
