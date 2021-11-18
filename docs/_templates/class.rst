.. https://raw.githubusercontent.com/sphinx-doc/sphinx/master/sphinx/ext/autosummary/templates/autosummary/class.rst

:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :no-private-members:
   :no-undoc-members:
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
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
       ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>
