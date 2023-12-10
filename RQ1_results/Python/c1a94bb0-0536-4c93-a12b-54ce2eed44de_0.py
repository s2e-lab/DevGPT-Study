# __init__.py
from . import controllers

# __manifest__.py
{
    'name': 'Open Specific Tab',
    'version': '1.0',
    'category': 'Custom',
    'sequence': 1,
    'summary': 'Module to open a specific tab in form view',
    'depends': ['base', 'web', 'project'],
    'data': [
        'views/templates.xml',
    ],
    'installable': True,
    'application': False,
    'auto_install': False,
}
